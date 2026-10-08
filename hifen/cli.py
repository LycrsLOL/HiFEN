from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from .config import ensure_dirs, load_config
from .run_artifacts import (
    apply_training_artifact_routing,
    detect_tee_log_path,
    finalize_training_run_artifacts,
)
from .utils import setup_logging


def load_runtime_config(args: argparse.Namespace, fallback_name: str) -> dict:
    if fallback_name == "deploy-data":
        train_csv = getattr(args, "train_csv", None)
        output_dir = getattr(args, "output_dir", None)
        if bool(train_csv) != bool(output_dir):
            raise SystemExit("One-click deployment requires both --train-csv and --output-dir.")
        if train_csv and getattr(args, "dataset", None):
            raise SystemExit("--dataset cannot be combined with one-click --train-csv mode.")
    config = load_config(args.config, profile=getattr(args, "profile", None))
    if fallback_name == "deploy-data" and getattr(args, "train_csv", None):
        from .deployment.workflow import configure_one_click_deployment

        try:
            config = configure_one_click_deployment(
                config,
                train_csv=args.train_csv,
                output_dir=args.output_dir,
                esm_checkpoint=getattr(args, "esm_checkpoint", None),
            )
        except ValueError as exc:
            raise SystemExit(f"Invalid one-click deployment: {exc}") from exc
    tee_log_path = None
    if fallback_name == "train":
        tee_log_path = detect_tee_log_path((config.get("runtime_outputs", {}) or {}).get("log_dir"))
        if tee_log_path is not None:
            apply_training_artifact_routing(config, tee_log_path)
    run_name = (
        (config.get("run", {}) or {}).get("name")
        or config.get("_profile")
        or config.get("mode")
        or fallback_name
    )
    # When stdout/stderr already flow through tee, a second FileHandler only
    # creates a near-duplicate log. Keep logging on the stream and let tee own
    # the single canonical file. Commands without tee retain the internal log.
    internal_log_dir = None if tee_log_path is not None else (config.get("runtime_outputs", {}) or {}).get("log_dir")
    log_path = setup_logging(internal_log_dir, str(run_name))
    config.setdefault("run", {})["internal_log_path"] = str(log_path or "")
    logging.info(
        "Command initialized | command=%s config=%s profile=%s log=%s tee_log=%s artifact_tag=%s artifact_dir=%s",
        fallback_name,
        args.config,
        config.get("_profile", ""),
        log_path or "",
        tee_log_path or "",
        (config.get("run", {}) or {}).get("artifact_tag", ""),
        (config.get("run", {}) or {}).get("artifact_dir", ""),
    )
    return config


def cmd_deploy_data(args: argparse.Namespace) -> None:
    from .data.cache_index import build_manifest, format_summary, save_manifest, summarize_manifest
    from .deployment.workflow import DeploymentStatus, stage_train_csv

    config = load_runtime_config(args, "deploy-data")
    selected_datasets = ["train"] if config.get("_one_click_deployment") else args.dataset
    ensure_dirs(config)
    dry_run = args.dry_run or bool((config.get("deployment", {}) or {}).get("dry_run", False))
    tracker = DeploymentStatus(config, dry_run=dry_run)
    try:
        deployed_csv = stage_train_csv(config, dry_run=dry_run)
        tracker.record(
            "dataset_csv",
            {"path": str(deployed_csv), "copied": bool(config.get("_one_click_deployment") and not dry_run)},
        )
        manifest = build_manifest(config, dataset_names=selected_datasets)
        out_path = save_manifest(manifest, config)
        summary = summarize_manifest(manifest)
        print(format_summary(summary))
        logging.info("Manifest path: %s", out_path)
        tracker.record("initial_manifest", {"path": str(out_path)})
        if dry_run:
            logging.info("Dry run complete; no data generation attempted.")
            tracker.complete({"dry_run": True, "manifest_path": str(out_path)})
            return
        from .deployment.resource_manager import complete_dataset_structures, repair_missing_embeddings, repair_missing_predicted_structures

        tracker.record("embedding_repair")
        if bool((config.get("deployment", {}) or {}).get("skip_embedding_file_repair", False)):
            # Embeddings are generated transiently into graphs instead; writing a
            # full fp32 embedding file cache here would flood the shared disk.
            repair_stats = {"embedding_repair_skipped": 1}
            logging.info("Embedding file repair skipped (deployment.skip_embedding_file_repair=true).")
        else:
            repair_stats = repair_missing_embeddings(config, dataset_names=selected_datasets, limit=args.limit)
        tracker.record("embedding_repair", repair_stats)
        deployment = config.get("deployment", {}) or {}
        if bool(deployment.get("allow_completion", False)):
            structure_stats = {"structure_repair_skipped_for_completion_pipeline": 1}
            logging.info(
                "Predicted/crystal structure repair skipped because structure completion pipeline handles downloads."
            )
        else:
            structure_stats = repair_missing_predicted_structures(config, dataset_names=selected_datasets, limit=args.limit)
        tracker.record("structure_repair", structure_stats)
        pipeline_graph = bool(deployment.get("pipeline_graph_after_completion", False)) and bool(deployment.get("allow_completion", False))
        graph_session = None
        if pipeline_graph:
            from .data.graph_preparation import ThinGraphBuildSession

            graph_session = ThinGraphBuildSession(
                config,
                manifest=manifest,
                dataset_names=selected_datasets,
                dry_run=False,
                overwrite=args.overwrite,
            )
            logging.info("Pipeline graph building enabled after each completed structure.")
        tracker.record("structure_completion_and_graph")
        completion_stats = complete_dataset_structures(
            config,
            dataset_names=selected_datasets,
            limit=args.limit,
            on_complete=graph_session.build_row if graph_session is not None else None,
        )
        graph_pipeline_stats = graph_session.stats if graph_session is not None else {}
        result_stats = {
            "resource_repair": repair_stats,
            "structure_repair": structure_stats,
            "structure_completion": completion_stats,
            "graph_pipeline": graph_pipeline_stats,
        }
        tracker.record("structure_completion_and_graph", result_stats)
        print(result_stats)
        if (
            repair_stats.get("embedding_generated", 0) > 0
            or structure_stats.get("structure_downloaded", 0) > 0
            or completion_stats.get("completion_written", 0) > 0
            or graph_pipeline_stats.get("written", 0) > 0
            or graph_pipeline_stats.get("structure_archived", 0) > 0
        ):
            manifest = build_manifest(config, dataset_names=selected_datasets)
            out_path = save_manifest(manifest, config)
            logging.info("Manifest refreshed after resource repair: %s", out_path)
        tracker.record("final_manifest", {"path": str(out_path)})

        from .data.graph_preparation import prepare_thin_graphs

        if pipeline_graph:
            logging.info("Pipeline graph preparation summary: %s", graph_pipeline_stats)
            print(graph_pipeline_stats)
            tracker.complete(result_stats)
            return

        stats = prepare_thin_graphs(
            config,
            manifest=manifest,
            dataset_names=selected_datasets,
            limit=args.limit,
            dry_run=False,
            overwrite=args.overwrite,
        )
        logging.info("Thin graph preparation summary: %s", stats)
        print(stats)
        result_stats["graph"] = stats
        tracker.complete(result_stats)
    except BaseException as exc:
        tracker.fail(exc)
        raise


def cmd_train(args: argparse.Namespace) -> None:
    from .training.trainer import run_train

    config = load_runtime_config(args, "train")
    if os.environ.get("RANK", "0") == "0":
        logging.info(
            "Train runtime | pid=%s rank=%s dataset=%s dry_run=%s resume=%s init_checkpoint=%s",
            os.getpid(),
            os.environ.get("RANK", "0"),
            args.dataset,
            args.dry_run,
            args.resume,
            args.init_checkpoint,
        )
    try:
        result = run_train(
            config,
            dry_run=args.dry_run,
            limit=args.limit,
            epochs_override=args.epochs,
            device_override=args.device,
            repair_labels=args.repair_labels,
            distributed=args.distributed,
            resume_checkpoint=args.resume,
            init_checkpoint=args.init_checkpoint,
        )
    except Exception:
        if os.environ.get("RANK", "0") == "0":
            logging.exception("Train command failed | config=%s", args.config)
            sys.stdout.flush()
            sys.stderr.flush()
            time.sleep(0.25)
            finalize_training_run_artifacts(
                config,
                None,
                status="failed",
                error="See the tagged tee/internal logs for the exception traceback.",
            )
        raise
    if os.environ.get("RANK", "0") == "0":
        print(result)
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(0.25)
        artifacts = finalize_training_run_artifacts(
            config,
            result,
            status=str(result.get("status", "completed")),
        )
        if artifacts:
            logging.info("Tagged training artifacts finalized: %s", artifacts)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from .training.evaluator import run_evaluate

    config = load_runtime_config(args, "evaluate")
    run_evaluate(config, checkpoint=args.checkpoint, dry_run=args.dry_run)


def cmd_infer(args: argparse.Namespace) -> None:
    from .inference.pipeline import run_mock_inference

    config = load_runtime_config(args, "infer")
    if args.sequence:
        sequence = args.sequence
    elif args.fasta:
        from .inference.io import read_fasta

        _, sequence = read_fasta(args.fasta)
    else:
        raise SystemExit("infer requires --sequence or --fasta in the current scaffold")
    result = run_mock_inference(config, sequence, output_dir=args.output)
    print(f"Wrote inference result for {result['protein_id']}")


def cmd_export_repro(args: argparse.Namespace) -> None:
    from .reproducibility import export_reproducibility_metadata

    config = load_runtime_config(args, "reproducibility")
    out_dir = args.output or config["storage"]["reproducibility_dir"]
    out_path = export_reproducibility_metadata(config, out_dir)
    print(out_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hifen")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("deploy-data")
    p.add_argument("--config", default="configs/hifen.yaml")
    p.add_argument("--profile")
    p.add_argument("--train-csv", help="One-click mode: path to train_dataset.csv")
    p.add_argument("--output-dir", help="One-click mode: root directory for train embeddings, structures, graphs, and metadata")
    p.add_argument("--esm-checkpoint", help="Optional ESM-3 checkpoint override for one-click mode")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dataset", action="append", help="Dataset key from the loaded config, e.g. train, external_price149, new_437")
    p.add_argument("--limit", type=int)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_deploy_data)

    p = sub.add_parser("train")
    p.add_argument("--config", default="configs/hifen.yaml")
    p.add_argument("--profile")
    p.add_argument(
        "--dataset",
        choices=("train",),
        default="train",
        help="Training dataset key. The current training pipeline uses the configured 'train' dataset.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--device")
    p.add_argument("--repair-labels", action="store_true")
    p.add_argument("--distributed", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument(
        "--resume",
        nargs="?",
        const="last",
        default=None,
        help="Resume from a checkpoint path; with no value, resumes from storage.checkpoint_dir/last.pt.",
    )
    p.add_argument(
        "--init-checkpoint",
        help="Load model weights from this checkpoint and start a fresh training history at epoch 1.",
    )
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate")
    p.add_argument("--config", default="configs/hifen.yaml")
    p.add_argument("--profile")
    p.add_argument("--checkpoint")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("infer")
    p.add_argument("--config", default="configs/hifen.yaml")
    p.add_argument("--profile")
    p.add_argument("--checkpoint")
    p.add_argument("--sequence")
    p.add_argument("--fasta")
    p.add_argument("--output")
    p.set_defaults(func=cmd_infer)

    p = sub.add_parser("export-reproducibility")
    p.add_argument("--config", default="configs/hifen.yaml")
    p.add_argument("--profile")
    p.add_argument("--output")
    p.set_defaults(func=cmd_export_repro)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
