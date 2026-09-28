from __future__ import annotations

import csv
import importlib.util
import json
import logging
import os
import shlex
import sys
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable


def _norm(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return None
    return text


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def _graph_path_for_row(config: dict, dataset: str, row: dict[str, str]) -> Path | None:
    uniprot_id = _norm(row.get("uniprot_id"))
    if not uniprot_id:
        return None
    pdb_id = _norm(row.get("pdb_id"))
    output_subdir = config["datasets"][dataset].get("output_subdir", dataset)
    graph_name = f"{uniprot_id}_{pdb_id}.pt" if pdb_id else f"{uniprot_id}.pt"
    return Path(config["storage"]["save_dir"]) / output_subdir / "graph" / graph_name


def _has_prepared_graph(config: dict, dataset: str, row: dict[str, str]) -> bool:
    graph_path = _graph_path_for_row(config, dataset, row)
    return bool(graph_path and graph_path.exists() and graph_path.stat().st_size > 0)


def _clean_sequence(value: Any) -> str:
    text = _norm(value)
    if not text:
        return ""
    return "".join(text.split()).upper().replace("*", "")


def _normalize_embedding_shape(embedding: Any) -> torch.Tensor:
    if embedding is None:
        raise ValueError("ESM embedding output is None")
    if isinstance(embedding, np.ndarray):
        embedding = torch.from_numpy(embedding)
    if not isinstance(embedding, torch.Tensor):
        embedding = torch.as_tensor(embedding)
    if embedding.dim() == 3 and embedding.size(0) == 1:
        embedding = embedding.squeeze(0)
    if embedding.dim() != 2:
        raise ValueError(f"Expected 2D residue embeddings, got shape={tuple(embedding.shape)}")
    return embedding


def _embedding_validation_issue(embedding: torch.Tensor, fail_abs_threshold: float = 1e20) -> str | None:
    if embedding.numel() == 0:
        return "empty"
    if torch.isnan(embedding).any():
        return "nan"
    if torch.isinf(embedding).any():
        return "inf"
    if (embedding == 0).all():
        return "all_zero"
    if embedding.size(0) > 1 and torch.var(embedding, dim=0, unbiased=False).sum().item() < 1e-6:
        return "constant_sequence"
    max_abs = embedding.abs().max().item()
    if max_abs > fail_abs_threshold:
        return f"extreme_value:{max_abs:.2e}"
    return None


_HF_ESM_MARKER = "_hresannogat_hf_esm_backend"


def _load_esm_hf_model(config: dict, device: str):
    esm_config = config.get("esm", {}) or {}
    model_type = str(esm_config.get("model_type", "esm2")).strip().lower()
    default_name = {
        "esm2": "facebook/esm2_t33_650M_UR50D",
        "esm1b": "facebook/esm1b_t33_650M_UR50S",
        "esm1": "facebook/esm1b_t33_650M_UR50S",
    }[model_type]
    model_name = str(esm_config.get("hf_model_name") or default_name)
    local_checkpoint = _norm(esm_config.get("checkpoint"))
    logging.info("Initializing HF ESM model (%s): %s", model_type, model_name)
    # tensorboard ships protobuf descriptors that fail with protobuf>=3.20;
    # the pure-python implementation keeps the transformers import chain working
    # without touching the shared environment.
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    from transformers import AutoModel, AutoTokenizer

    if local_checkpoint and Path(local_checkpoint).exists():
        source = local_checkpoint
    else:
        if local_checkpoint:
            logging.warning("Configured HF ESM checkpoint does not exist, falling back to hub: %s", local_checkpoint)
        source = model_name
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModel.from_pretrained(source)
    model = model.to(device).eval()
    setattr(model, _HF_ESM_MARKER, {"tokenizer": tokenizer})
    return model


def _extract_residue_embedding_hf(model, sequence: str, device: str) -> tuple[torch.Tensor, str]:
    backend = getattr(model, _HF_ESM_MARKER)
    tokenizer = backend["tokenizer"]
    encoded = tokenizer(sequence, return_tensors="pt", return_attention_mask=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    special_ids = set(tokenizer.all_special_ids)
    with torch.no_grad():
        if str(device).startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
    keep = [index for index, token_id in enumerate(input_ids[0].tolist()) if token_id not in special_ids]
    if len(keep) != len(sequence):
        raise ValueError(
            f"HF ESM tokenization mismatch: sequence_length={len(sequence)} residue_tokens={len(keep)}"
        )
    embedding = hidden[0, keep, :].detach().float().cpu()
    return _normalize_embedding_shape(embedding), "hf_last_hidden_state"


def _load_esm3_model(config: dict, device: str):
    esm_config = config.get("esm", {}) or {}
    model_type = str(esm_config.get("model_type") or "esm3").strip().lower()
    if model_type in {"esm2", "esm1b", "esm1"}:
        return _load_esm_hf_model(config, device)

    source_path = _norm(esm_config.get("source_path") or esm_config.get("esm_source_path"))
    if source_path and Path(source_path).is_dir():
        sys.path.insert(0, source_path)

    from esm.models.esm3 import ESM3
    from esm.utils.constants.models import ESM3_OPEN_SMALL

    checkpoint = _norm(esm_config.get("checkpoint"))
    logging.info("Initializing ESM-3 model: %s", ESM3_OPEN_SMALL)
    try:
        with torch.device(device):
            model = ESM3.from_pretrained(ESM3_OPEN_SMALL)
    except ImportError as exc:
        # fair-esm and EvolutionaryScale's esm package share the same top-level
        # module name.  If fair-esm overwrites esm.pretrained.py, ESM3 remains
        # importable but from_pretrained() cannot import load_local_model.  The
        # configured checkpoint already contains the sequence model weights, so
        # construct the public ESM3-small architecture directly in that case.
        if "load_local_model" not in str(exc) or not (checkpoint and Path(checkpoint).exists()):
            raise
        logging.warning(
            "esm.pretrained is shadowed by fair-esm; constructing ESM-3-small "
            "directly before loading the configured local checkpoint."
        )
        from esm.tokenization import get_esm3_model_tokenizers
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

        # esm 3.2.1 exposes read-only convenience properties for these tokens,
        # while older supported transformers releases initialize them through
        # setattr().  Add setters that target SpecialTokensMixin's backing
        # attributes without changing normal token lookup behavior.
        for token_name in ("cls_token", "eos_token", "mask_token", "pad_token"):
            descriptor = getattr(EsmSequenceTokenizer, token_name)
            if isinstance(descriptor, property) and descriptor.fset is None:
                hidden_name = f"_{token_name}"

                def set_token(self, value, hidden_name=hidden_name):
                    setattr(self, hidden_name, value)

                setattr(
                    EsmSequenceTokenizer,
                    token_name,
                    property(descriptor.fget, set_token, descriptor.fdel, descriptor.__doc__),
                )

        # transformers 4.37 still stores special tokens on _<name> and has no
        # instance __getattr__ method, which esm 3.2.1's convenience getter
        # assumes.  Read the same backing values directly.
        def get_token(self, token_name):
            value = getattr(self, f"_{token_name}")
            return str(value)

        EsmSequenceTokenizer._get_token = get_token

        def unavailable_component(component_device):
            raise RuntimeError(
                "Structure/function decoders are unavailable in the sequence-embedding-only "
                "ESM-3 compatibility loader."
            )

        with torch.device(device):
            model = ESM3(
                d_model=1536,
                n_heads=24,
                v_heads=256,
                n_layers=48,
                structure_encoder_fn=unavailable_component,
                structure_decoder_fn=unavailable_component,
                function_decoder_fn=unavailable_component,
                tokenizers=get_esm3_model_tokenizers(ESM3_OPEN_SMALL),
            )
            if torch.device(device).type != "cpu":
                model = model.to(torch.bfloat16)

    if checkpoint and Path(checkpoint).exists():
        logging.info("Loading local ESM-3 weights: %s", checkpoint)
        state_dict = torch.load(checkpoint, map_location="cpu")
        if isinstance(state_dict, dict):
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
    elif checkpoint:
        logging.warning("Configured ESM-3 checkpoint does not exist: %s", checkpoint)

    return model.to(device).eval()


def _extract_residue_embedding(model, sequence: str, device: str) -> tuple[torch.Tensor, str]:
    if getattr(model, _HF_ESM_MARKER, None) is not None:
        return _extract_residue_embedding_hf(model, sequence, device)

    from esm.sdk.api import ESMProtein, LogitsConfig, SamplingConfig

    with torch.no_grad(), torch.device(device):
        protein = ESMProtein(sequence=sequence)
        protein_tensor = model.encode(protein)
        if hasattr(model, "logits"):
            output = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
            return _normalize_embedding_shape(output.embeddings), "logits_embeddings"
        output = model.forward_and_sample(protein_tensor, SamplingConfig(return_per_residue_embeddings=True))
        return _normalize_embedding_shape(output.per_residue_embedding), "sampling_embeddings_fallback"


def _expected_embedding_paths(embedding_dir: Path, uniprot_id: str, pdb_id: str | None) -> list[Path]:
    paths = []
    if pdb_id:
        paths.append(embedding_dir / f"{uniprot_id}_{pdb_id}.pt")
    paths.append(embedding_dir / f"{uniprot_id}.pt")
    return paths


def _target_embedding_path(embedding_dir: Path, uniprot_id: str, pdb_id: str | None) -> Path:
    if pdb_id:
        return embedding_dir / f"{uniprot_id}_{pdb_id}.pt"
    return embedding_dir / f"{uniprot_id}.pt"


def _has_embedding(root: Path, uniprot_id: str, pdb_id: str | None) -> bool:
    embedding_dir = root / "embedding"
    return any(path.exists() and path.stat().st_size > 0 for path in _expected_embedding_paths(embedding_dir, uniprot_id, pdb_id))


def repair_missing_embeddings(
    config: dict,
    dataset_names: Iterable[str] | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    deployment = config.get("deployment", {}) or {}
    esm_config = config.get("esm", {}) or {}
    allow_generate = bool(deployment.get("allow_generate_embedding", deployment.get("allow_recompute_embedding", False)))
    allow_recompute = bool(deployment.get("allow_recompute_embedding", False))
    save_embedding_in_graph = bool((config.get("storage", {}) or {}).get("save_embedding_in_graph", False))

    datasets = config.get("datasets", {}) or {}
    selected = list(dataset_names or datasets.keys())
    save_dir = Path(config["storage"]["save_dir"])
    device = _norm(esm_config.get("device") or deployment.get("embedding_device"))
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    fail_abs_threshold = float(esm_config.get("embedding_fail_abs_threshold", 1e20))

    stats = {
        "embedding_existing": 0,
        "embedding_generated": 0,
        "embedding_missing_generation_disabled": 0,
        "embedding_skipped_no_uniprot": 0,
        "embedding_skipped_no_sequence": 0,
        "embedding_skipped_existing_graph": 0,
        "embedding_skipped_self_contained_graph": 0,
        "embedding_invalid": 0,
        "embedding_failed": 0,
    }
    tasks: list[tuple[str, str, str | None, str, Path]] = []
    processed = 0

    if save_embedding_in_graph:
        stats["embedding_skipped_self_contained_graph"] = sum(
            len(_read_csv_rows(datasets[name]["csv"])) for name in selected if name in datasets
        )
        logging.info("Persistent embedding repair skipped because storage.save_embedding_in_graph=true.")
        logging.info("Embedding repair summary: %s", stats)
        return stats

    skip_for_existing_graph = bool(deployment.get("skip_resource_repair_for_existing_graph", False))

    for dataset in selected:
        spec = datasets[dataset]
        output_subdir = spec.get("output_subdir", dataset)
        embedding_dir = save_dir / output_subdir / "embedding"
        rows = _read_csv_rows(spec["csv"])
        if limit is not None:
            remaining = max(limit - processed, 0)
            rows = rows[:remaining]
        for row in rows:
            processed += 1
            if skip_for_existing_graph and _has_prepared_graph(config, dataset, row):
                stats["embedding_skipped_existing_graph"] += 1
                continue
            uniprot_id = _norm(row.get("uniprot_id"))
            if not uniprot_id:
                stats["embedding_skipped_no_uniprot"] += 1
                continue
            pdb_id = _norm(row.get("pdb_id"))
            if any(path.exists() and path.stat().st_size > 0 for path in _expected_embedding_paths(embedding_dir, uniprot_id, pdb_id)):
                stats["embedding_existing"] += 1
                continue
            if not allow_generate:
                stats["embedding_missing_generation_disabled"] += 1
                continue
            sequence = _clean_sequence(row.get("sequence"))
            if not sequence:
                stats["embedding_skipped_no_sequence"] += 1
                continue
            target_path = _target_embedding_path(embedding_dir, uniprot_id, pdb_id)
            if target_path.exists() and not allow_recompute:
                stats["embedding_existing"] += 1
                continue
            tasks.append((dataset, uniprot_id, pdb_id, sequence, target_path))
        if limit is not None and processed >= limit:
            break

    if not tasks:
        logging.info("Embedding repair summary: %s", stats)
        return stats
    if not allow_generate:
        logging.info("Embedding generation is disabled; missing embeddings were not generated.")
        logging.info("Embedding repair summary: %s", stats)
        return stats

    model = _load_esm3_model(config, device)
    sources: set[str] = set()
    progress = tqdm(tasks, desc="Generating missing ESM-3 embeddings", unit="seq", dynamic_ncols=True)
    for dataset, uniprot_id, pdb_id, sequence, target_path in progress:
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            embedding, source = _extract_residue_embedding(model, sequence, device)
            sources.add(source)
            embedding = embedding.detach().cpu()
            issue = _embedding_validation_issue(embedding, fail_abs_threshold=fail_abs_threshold)
            if issue:
                stats["embedding_invalid"] += 1
                logging.warning("Invalid embedding for %s/%s: %s", uniprot_id, pdb_id or "", issue)
                continue
            torch.save(embedding, target_path)
            stats["embedding_generated"] += 1
        except Exception as exc:
            stats["embedding_failed"] += 1
            logging.warning("Embedding generation failed for %s/%s in %s: %s", uniprot_id, pdb_id or "", dataset, exc)

    logging.info("Embedding repair sources: %s", sorted(sources))
    logging.info("Embedding repair summary: %s", stats)
    return stats


def _download_file(
    url: str,
    output_path: Path,
    timeout: float = 60.0,
    *,
    retries: int = 1,
    retry_backoff_seconds: float = 0.0,
    warn_on_failure: bool = True,
) -> bool:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    attempts = max(int(retries), 1)
    last_error: str | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, headers={"User-Agent": "HiFEN/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    last_error = f"HTTP {status}"
                    if status == 404:
                        tmp_path.unlink(missing_ok=True)
                        return False
                    raise RuntimeError(last_error)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with tmp_path.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            if tmp_path.stat().st_size == 0:
                last_error = "empty response"
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(last_error)
            tmp_path.replace(output_path)
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                tmp_path.unlink(missing_ok=True)
                return False
            last_error = f"HTTP {exc.code}"
            tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            last_error = str(exc)
            tmp_path.unlink(missing_ok=True)

        if attempt < attempts:
            logging.debug("Download attempt %d/%d failed for %s: %s", attempt, attempts, url, last_error)
            if retry_backoff_seconds > 0:
                time.sleep(retry_backoff_seconds)
            continue

    if warn_on_failure:
        logging.warning("Download failed for %s after %d attempt(s): %s", url, attempts, last_error)
    else:
        logging.debug("Download failed for %s after %d attempt(s): %s", url, attempts, last_error)
    return False


def _alphafold_cif_urls(uniprot_id: str, download_config: dict[str, Any], timeout: float) -> list[str]:
    urls: list[str] = []
    api_template = download_config.get(
        "alphafold_api_url_template",
        "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}",
    )
    if bool(download_config.get("use_alphafold_api", True)):
        try:
            request = urllib.request.Request(
                api_template.format(uniprot_id=uniprot_id),
                headers={"User-Agent": "HiFEN/0.1"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            for item in payload if isinstance(payload, list) else []:
                url = item.get("cifUrl") if isinstance(item, dict) else None
                if url:
                    urls.append(str(url))
        except Exception as exc:
            logging.debug("Could not query AlphaFold API for %s: %s", uniprot_id, exc)
    versions = download_config.get("alphafold_versions", [6, 5, 4, 3, 2, 1])
    template = download_config.get(
        "alphafold_cif_url_template",
        "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v{version}.cif",
    )
    for version in versions:
        urls.append(template.format(uniprot_id=uniprot_id, version=version))
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def _pdb_cif_urls(pdb_id: str, download_config: dict[str, Any]) -> list[str]:
    pdb_code = pdb_id.lower()
    templates = download_config.get(
        "pdb_cif_url_templates",
        [
            "https://files.rcsb.org/download/{pdb_id}.cif",
            "https://files.rcsb.org/download/{pdb_id_upper}.cif",
        ],
    )
    if isinstance(templates, str):
        templates = [templates]
    urls: list[str] = []
    seen: set[str] = set()
    for template in templates:
        url = str(template).format(pdb_id=pdb_code, pdb_id_upper=pdb_code.upper())
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _valid_structure_payload(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle):
                if line_number >= 200:
                    break
                text = line.strip()
                if not text:
                    continue
                if path.suffix.lower() in {".cif", ".mmcif"}:
                    return text.startswith("data_")
                if text.lower().startswith(("<!doctype html", "<html")):
                    return False
                if text.startswith(("ATOM", "HETATM", "HEADER", "TITLE", "MODEL")):
                    return True
    except OSError:
        return False
    return False


def _download_pdb_crystal_structure(
    *,
    root: Path,
    uniprot_id: str,
    pdb_id: str,
    download_config: dict[str, Any],
    timeout: float,
    retries: int,
    retry_backoff_seconds: float,
) -> bool:
    crystal_dir = root / "crystal"
    output_path = crystal_dir / f"{uniprot_id}_{pdb_id.lower()}.cif"
    for url in _pdb_cif_urls(pdb_id, download_config):
        if _download_file(
            url,
            output_path,
            timeout=timeout,
            retries=retries,
            retry_backoff_seconds=retry_backoff_seconds,
            warn_on_failure=False,
        ):
            if _valid_structure_payload(output_path):
                return True
            output_path.unlink(missing_ok=True)
    return False


def _safe_alphafold_target_id(uniprot_id: str, pdb_id: str | None = None) -> str:
    raw = f"{uniprot_id}_{pdb_id}" if pdb_id else uniprot_id
    safe = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in raw)
    return safe or "alphafold_target"


def _write_single_fasta(path: Path, target_id: str, sequence: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f">{target_id}\n")
        for start in range(0, len(sequence), 80):
            handle.write(sequence[start : start + 80] + "\n")


def _resolve_local_alphafold_runner(download_config: dict[str, Any]) -> tuple[str, Path] | tuple[None, str]:
    params_dir_text = _norm(download_config.get("local_alphafold_params_dir"))
    if not params_dir_text:
        return None, "local_alphafold_params_dir is not configured"
    params_dir = Path(params_dir_text).expanduser()
    if not params_dir.is_dir():
        return None, f"AlphaFold params directory does not exist: {params_dir}"

    backend = str(download_config.get("local_alphafold_backend", "alphafold")).strip().lower()
    if backend == "alphafold":
        model_name = str(download_config.get("local_alphafold_model_name", "model_1_ptm"))
        params_path = params_dir / f"params_{model_name}.npz"
        if not params_path.exists():
            return None, f"AlphaFold params file does not exist: {params_path}"
        if importlib.util.find_spec("alphafold") is None:
            return None, "Python package 'alphafold' is not installed"
        return sys.executable, params_dir

    if backend != "colabfold":
        return None, f"Unsupported local_alphafold_backend: {backend}"

    binary = _norm(download_config.get("local_alphafold_binary")) or "colabfold_batch"
    binary_path = shutil.which(binary)
    if binary_path is None:
        candidate = Path(binary).expanduser()
        if candidate.exists():
            binary_path = str(candidate)
    if binary_path is None:
        return None, f"Local AlphaFold runner is not available on PATH: {binary}"
    return binary_path, params_dir


def _local_alphafold_command(
    *,
    binary_path: str,
    params_dir: Path,
    fasta_path: Path,
    results_dir: Path,
    output_path: Path,
    download_config: dict[str, Any],
) -> list[str]:
    backend = str(download_config.get("local_alphafold_backend", "alphafold")).strip().lower()
    if backend == "alphafold":
        return [
            binary_path,
            "-m",
            "hifen.deployment.local_alphafold_predict",
            "--fasta",
            str(fasta_path),
            "--params-dir",
            str(params_dir),
            "--output",
            str(output_path),
            "--model-name",
            str(download_config.get("local_alphafold_model_name", "model_1_ptm")),
            "--num-recycle",
            str(int(download_config.get("local_alphafold_num_recycle", 3))),
            "--random-seed",
            str(int(download_config.get("local_alphafold_random_seed", 0))),
        ]

    cmd = [
        binary_path,
        "--data",
        str(params_dir),
        "--msa-mode",
        str(download_config.get("local_alphafold_msa_mode", "single_sequence")),
        "--model-type",
        str(download_config.get("local_alphafold_model_type", "alphafold2_ptm")),
        "--num-models",
        str(int(download_config.get("local_alphafold_num_models", 1))),
        "--num-recycle",
        str(int(download_config.get("local_alphafold_num_recycle", 3))),
        "--overwrite-existing-results",
    ]
    if bool(download_config.get("local_alphafold_amber", False)):
        cmd.append("--amber")
        cmd.extend(["--num-relax", str(int(download_config.get("local_alphafold_num_relax", 1)))])
    if bool(download_config.get("local_alphafold_disable_unified_memory", False)):
        cmd.append("--disable-unified-memory")
    cmd.extend([str(fasta_path), str(results_dir)])
    return cmd


def _find_local_alphafold_prediction(results_dir: Path, target_id: str) -> Path | None:
    preferred_names = (
        "ranked_0.pdb",
        "relaxed_model_1_pred_0.pdb",
        "unrelaxed_model_1_pred_0.pdb",
    )
    candidates: list[Path] = []
    for base in (results_dir / target_id, results_dir):
        candidates.extend(base / name for name in preferred_names)
    patterns = (
        f"{target_id}*rank_001*.pdb",
        f"{target_id}*ranked_0*.pdb",
        f"{target_id}*.pdb",
        f"{target_id}*.cif",
        f"{target_id}*.mmcif",
        "ranked_0.pdb",
        "*rank_001*.pdb",
        "*ranked_0*.pdb",
        "*.pdb",
        "*.cif",
        "*.mmcif",
    )
    for pattern in patterns:
        candidates.extend(sorted(results_dir.rglob(pattern)))

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists() and path.stat().st_size > 0 and _valid_structure_payload(path):
            return path
    return None


def _run_local_alphafold_prediction(
    *,
    root: Path,
    dataset: str,
    uniprot_id: str,
    pdb_id: str | None,
    sequence: str,
    predicted_dir: Path,
    download_config: dict[str, Any],
    binary_path: str,
    params_dir: Path,
) -> bool:
    target_id = _safe_alphafold_target_id(uniprot_id, pdb_id)
    configured_work_dir = _norm(download_config.get("local_alphafold_work_dir"))
    work_root = Path(configured_work_dir).expanduser() / dataset if configured_work_dir else root / "local_alphafold"
    target_work_dir = work_root / target_id
    results_dir = target_work_dir / "results"
    fasta_path = target_work_dir / f"{target_id}.fasta"
    log_path = target_work_dir / "local_alphafold.log"
    timeout = float(download_config.get("local_alphafold_timeout_seconds", 21600))
    backend = str(download_config.get("local_alphafold_backend", "alphafold")).strip().lower()
    target_path = predicted_dir / f"{uniprot_id}.pdb"

    existing_prediction = _find_local_alphafold_prediction(results_dir, target_id)
    if existing_prediction is None and target_path.exists() and target_path.stat().st_size > 0 and _valid_structure_payload(target_path):
        existing_prediction = target_path
    if existing_prediction is None:
        _write_single_fasta(fasta_path, target_id, sequence)
        results_dir.mkdir(parents=True, exist_ok=True)
        cmd = _local_alphafold_command(
            binary_path=binary_path,
            params_dir=params_dir,
            fasta_path=fasta_path,
            results_dir=results_dir,
            output_path=target_path,
            download_config=download_config,
        )
        env = None
        cuda_visible_devices = _norm(download_config.get("local_alphafold_cuda_visible_devices"))
        if cuda_visible_devices:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        if backend == "alphafold":
            env = os.environ.copy() if env is None else env
            project_root = str(Path(__file__).resolve().parents[2])
            previous_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = project_root if not previous_pythonpath else project_root + os.pathsep + previous_pythonpath
            env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        logging.info(
            "Running local AlphaFold fallback for %s (length=%d); log: %s",
            target_id,
            len(sequence),
            log_path,
        )
        try:
            with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
                log_handle.write("$ " + " ".join(shlex.quote(part) for part in cmd) + "\n")
                completed = subprocess.run(
                    cmd,
                    cwd=str(Path(__file__).resolve().parents[2] if backend == "alphafold" else target_work_dir),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout,
                    env=env,
                    check=False,
                )
        except subprocess.TimeoutExpired:
            logging.warning("Local AlphaFold fallback timed out for %s after %.0f seconds; log: %s", target_id, timeout, log_path)
            return False
        except Exception as exc:
            logging.warning("Local AlphaFold fallback failed to start for %s: %s", target_id, exc)
            return False
        if completed.returncode != 0:
            logging.warning("Local AlphaFold fallback failed for %s with exit code %s; log: %s", target_id, completed.returncode, log_path)
            return False
        existing_prediction = _find_local_alphafold_prediction(results_dir, target_id)
        if backend == "alphafold" and target_path.exists() and target_path.stat().st_size > 0 and _valid_structure_payload(target_path):
            existing_prediction = target_path

    if existing_prediction is None:
        logging.warning("Local AlphaFold fallback produced no valid structure for %s; log: %s", target_id, log_path)
        return False

    suffix = existing_prediction.suffix.lower()
    if suffix not in {".pdb", ".cif", ".mmcif"}:
        suffix = ".pdb"
    target_path = predicted_dir / f"{uniprot_id}{suffix}"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if existing_prediction.resolve() != target_path.resolve():
        shutil.copy2(existing_prediction, target_path)
    if _valid_structure_payload(target_path):
        return True
    target_path.unlink(missing_ok=True)
    logging.warning("Local AlphaFold fallback structure is invalid after import for %s: %s", target_id, existing_prediction)
    return False


def _expected_structure_paths(root: Path, uniprot_id: str, pdb_id: str | None) -> list[Path]:
    paths: list[Path] = []
    complete_dir = root / "complete"
    predicted_dir = root / "predicted"
    crystal_dir = root / "crystal"
    if pdb_id:
        for suffix in (".pdb", ".cif", ".mmcif"):
            paths.append(complete_dir / f"{uniprot_id}_{pdb_id}{suffix}")
            paths.append(predicted_dir / f"{uniprot_id}_{pdb_id}{suffix}")
            paths.append(crystal_dir / f"{pdb_id}_{uniprot_id}{suffix}")
            paths.append(crystal_dir / f"{uniprot_id}_{pdb_id}{suffix}")
    for suffix in (".cif", ".mmcif", ".pdb"):
        paths.append(predicted_dir / f"{uniprot_id}{suffix}")
        paths.append(complete_dir / f"{uniprot_id}{suffix}")
    return paths


def _first_valid_structure(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.stat().st_size > 0 and _valid_structure_payload(path):
            return path
    return None


def _complete_structure_candidates(root: Path, uniprot_id: str, pdb_id: str | None) -> list[Path]:
    complete_dir = root / "complete"
    base = f"{uniprot_id}_{pdb_id}" if pdb_id else uniprot_id
    return [complete_dir / f"{base}{suffix}" for suffix in (".pdb", ".cif", ".mmcif")]


def _crystal_structure_candidates(root: Path, uniprot_id: str, pdb_id: str | None) -> list[Path]:
    if not pdb_id:
        return []
    crystal_dir = root / "crystal"
    candidates: list[Path] = []
    for stem in (f"{uniprot_id}_{pdb_id.lower()}", f"{pdb_id.upper()}_{pdb_id.lower()}", f"{pdb_id}_{uniprot_id}", f"{pdb_id}"):
        for suffix in (".cif", ".mmcif", ".pdb"):
            candidates.append(crystal_dir / f"{stem}{suffix}")
    return candidates


def _predicted_structure_candidates(root: Path, uniprot_id: str, pdb_id: str | None) -> list[Path]:
    predicted_dir = root / "predicted"
    stems = [uniprot_id]
    if pdb_id:
        stems.insert(0, f"{uniprot_id}_{pdb_id}")
    candidates: list[Path] = []
    for stem in stems:
        for suffix in (".cif", ".mmcif", ".pdb"):
            candidates.append(predicted_dir / f"{stem}{suffix}")
    return candidates


def _download_predicted_structure(
    *,
    root: Path,
    dataset: str,
    uniprot_id: str,
    pdb_id: str | None,
    sequence: str,
    predicted_dir: Path,
    download_config: dict[str, Any],
    local_runner: tuple[str, Path] | None,
    max_local_alphafold_length: int,
    timeout: float,
    retries: int,
    retry_backoff_seconds: float,
) -> tuple[Path | None, str | None]:
    output_path = predicted_dir / f"{uniprot_id}.cif"
    for url in _alphafold_cif_urls(uniprot_id, download_config, timeout=timeout):
        if _download_file(
            url,
            output_path,
            timeout=timeout,
            retries=retries,
            retry_backoff_seconds=retry_backoff_seconds,
            warn_on_failure=False,
        ):
            if _valid_structure_payload(output_path):
                return output_path, "structure_predicted_downloaded"
            output_path.unlink(missing_ok=True)
    if local_runner is None:
        return None, "structure_not_found"
    if not sequence:
        return None, "structure_local_prediction_skipped_no_sequence"
    if max_local_alphafold_length > 0 and len(sequence) > max_local_alphafold_length:
        logging.info(
            "Local AlphaFold fallback skipped for %s/%s: sequence length %d exceeds limit %d",
            uniprot_id,
            pdb_id or "",
            len(sequence),
            max_local_alphafold_length,
        )
        return None, "structure_local_prediction_skipped_too_long"
    binary_path, params_dir = local_runner
    if _run_local_alphafold_prediction(
        root=root,
        dataset=dataset,
        uniprot_id=uniprot_id,
        pdb_id=pdb_id,
        sequence=sequence,
        predicted_dir=predicted_dir,
        download_config=download_config,
        binary_path=binary_path,
        params_dir=params_dir,
    ):
        predicted = _first_valid_structure(_predicted_structure_candidates(root, uniprot_id, pdb_id))
        return predicted, "structure_local_predicted"
    return None, "structure_local_prediction_failed"


def complete_dataset_structures(
    config: dict,
    dataset_names: Iterable[str] | None = None,
    limit: int | None = None,
    on_complete: Callable[[str, dict[str, str]], bool] | None = None,
) -> dict[str, int]:
    from .structure_fusion import complete_structure_if_needed

    deployment = config.get("deployment", {}) or {}
    download_config = config.get("structure_download", {}) or {}
    completion_config = config.get("structure_completion", {}) or {}
    allow_completion = bool(deployment.get("allow_completion", False))
    allow_download = bool(deployment.get("allow_download", False))
    allow_predicted = bool(deployment.get("download_predicted_structures", True))
    configured_datasets = deployment.get("complete_structure_datasets") or deployment.get("download_predicted_structure_datasets")
    if isinstance(configured_datasets, str):
        configured_datasets = [configured_datasets]
    configured_dataset_set = set(configured_datasets or [])

    datasets = config.get("datasets", {}) or {}
    requested = list(dataset_names or datasets.keys())
    selected = [name for name in requested if not configured_dataset_set or name in configured_dataset_set]
    save_dir = Path(config["storage"]["save_dir"])
    timeout = float(download_config.get("timeout_seconds", 60))
    download_retries = int(download_config.get("download_retries", 3))
    download_retry_backoff_seconds = float(download_config.get("download_retry_backoff_seconds", 2.0))
    structure_completion_workers = max(int(deployment.get("structure_completion_workers", 1) or 1), 1)
    local_runner: tuple[str, Path] | None = None
    if bool(download_config.get("local_alphafold_fallback", False)):
        resolved_binary, resolved_params_or_error = _resolve_local_alphafold_runner(download_config)
        if resolved_binary is None:
            logging.warning("Local AlphaFold fallback is disabled: %s", resolved_params_or_error)
        else:
            local_runner = (resolved_binary, resolved_params_or_error)
    max_local_alphafold_length = int(download_config.get("local_alphafold_max_length", 0) or 0)
    skip_structure_completion_for_existing_graph = bool(
        deployment.get("skip_structure_completion_for_existing_graph", True)
    )
    stats = {
        "completion_existing_graph": 0,
        "completion_existing": 0,
        "completion_written": 0,
        "completion_crystal_grafted": 0,
        "completion_predicted_only": 0,
        "completion_missing_predicted": 0,
        "completion_failed": 0,
        "completion_crystal_downloaded": 0,
        "completion_crystal_missing": 0,
        "completion_predicted_downloaded": 0,
        "completion_local_predicted": 0,
        "completion_skipped_disabled": 0,
        "completion_skipped_no_uniprot": 0,
        "completion_pipeline_graph_success": 0,
        "completion_pipeline_graph_failed": 0,
    }
    if not allow_completion:
        stats["completion_skipped_disabled"] = sum(
            len(_read_csv_rows(datasets[name]["csv"])) for name in selected if name in datasets
        )
        logging.info("Structure completion is disabled.")
        logging.info("Structure completion summary: %s", stats)
        return stats

    processed = 0
    for dataset in selected:
        spec = datasets[dataset]
        output_subdir = spec.get("output_subdir", dataset)
        root = save_dir / output_subdir
        complete_dir = root / "complete"
        predicted_dir = root / "predicted"
        rows = _read_csv_rows(spec["csv"])
        if limit is not None:
            rows = rows[: max(limit - processed, 0)]

        def complete_row(row: dict[str, str]) -> dict[str, int]:
            row_stats = {key: 0 for key in stats}
            uniprot_id = _norm(row.get("uniprot_id"))
            if not uniprot_id:
                row_stats["completion_skipped_no_uniprot"] += 1
                return row_stats
            pdb_id = _norm(row.get("pdb_id"))
            if skip_structure_completion_for_existing_graph and _has_prepared_graph(config, dataset, row):
                row_stats["completion_existing_graph"] += 1
                row_stats["_pdb_cap_success"] = 1
                return row_stats

            def run_complete_callback() -> bool:
                if on_complete is None:
                    return True
                try:
                    ok = bool(on_complete(dataset, row))
                except Exception as callback_exc:
                    logging.warning(
                        "Pipeline graph callback failed for %s/%s: %s",
                        uniprot_id,
                        pdb_id or "",
                        callback_exc,
                    )
                    ok = False
                if ok:
                    row_stats["completion_pipeline_graph_success"] += 1
                else:
                    row_stats["completion_pipeline_graph_failed"] += 1
                return ok

            try:
                existing_complete = _first_valid_structure(_complete_structure_candidates(root, uniprot_id, pdb_id))
                if existing_complete:
                    row_stats["completion_existing"] += 1
                    row_stats["_pdb_cap_success"] = 1 if run_complete_callback() else 0
                    return row_stats
                sequence = _clean_sequence(row.get("sequence"))
                crystal = _first_valid_structure(_crystal_structure_candidates(root, uniprot_id, pdb_id))
                predicted = _first_valid_structure(_predicted_structure_candidates(root, uniprot_id, pdb_id))
                need_crystal_download = bool(pdb_id and crystal is None and allow_download)
                need_predicted_download = bool(predicted is None and allow_download and allow_predicted)
                crystal_downloaded = False
                predicted_source_key: str | None = None

                def download_crystal_task() -> bool:
                    assert pdb_id is not None
                    return _download_pdb_crystal_structure(
                        root=root,
                        uniprot_id=uniprot_id,
                        pdb_id=pdb_id,
                        download_config=download_config,
                        timeout=timeout,
                        retries=download_retries,
                        retry_backoff_seconds=download_retry_backoff_seconds,
                    )

                def download_predicted_task() -> tuple[Path | None, str | None]:
                    return _download_predicted_structure(
                        root=root,
                        dataset=dataset,
                        uniprot_id=uniprot_id,
                        pdb_id=pdb_id,
                        sequence=sequence,
                        predicted_dir=predicted_dir,
                        download_config=download_config,
                        local_runner=local_runner,
                        max_local_alphafold_length=max_local_alphafold_length,
                        timeout=timeout,
                        retries=download_retries,
                        retry_backoff_seconds=download_retry_backoff_seconds,
                    )

                if need_crystal_download and need_predicted_download:
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        futures = {
                            executor.submit(download_crystal_task): "crystal",
                            executor.submit(download_predicted_task): "predicted",
                        }
                        for future in as_completed(futures):
                            task_name = futures[future]
                            try:
                                if task_name == "crystal":
                                    crystal_downloaded = bool(future.result())
                                else:
                                    predicted, predicted_source_key = future.result()
                            except Exception as exc:
                                logging.warning(
                                    "Parallel structure download failed for %s/%s (%s): %s",
                                    uniprot_id,
                                    pdb_id or "",
                                    task_name,
                                    exc,
                                )
                elif need_crystal_download:
                    try:
                        crystal_downloaded = download_crystal_task()
                    except Exception as exc:
                        logging.warning("Crystal download failed for %s/%s: %s", uniprot_id, pdb_id or "", exc)
                elif need_predicted_download:
                    try:
                        predicted, predicted_source_key = download_predicted_task()
                    except Exception as exc:
                        logging.warning("Predicted structure download failed for %s/%s: %s", uniprot_id, pdb_id or "", exc)

                if need_crystal_download:
                    if crystal_downloaded:
                        crystal = _first_valid_structure(_crystal_structure_candidates(root, uniprot_id, pdb_id))
                        row_stats["completion_crystal_downloaded"] += 1
                    else:
                        row_stats["completion_crystal_missing"] += 1
                elif pdb_id and crystal is None:
                    row_stats["completion_crystal_missing"] += 1

                if predicted_source_key == "structure_predicted_downloaded":
                    row_stats["completion_predicted_downloaded"] += 1
                elif predicted_source_key == "structure_local_predicted":
                    row_stats["completion_local_predicted"] += 1
                if predicted is None:
                    row_stats["completion_missing_predicted"] += 1
                    return row_stats

                base = f"{uniprot_id}_{pdb_id}" if pdb_id else uniprot_id
                suffix = ".pdb" if crystal is not None else predicted.suffix.lower()
                if suffix not in {".pdb", ".cif", ".mmcif"}:
                    suffix = ".pdb"
                output_path = complete_dir / f"{base}{suffix}"
                result = complete_structure_if_needed(
                    crystal_path=crystal,
                    predicted_path=predicted,
                    output_path=output_path,
                    allow_completion=True,
                    overwrite=False,
                    full_sequence=sequence,
                    min_predicted_plddt=float(completion_config.get("min_predicted_plddt", 50.0)),
                    min_anchor_residues=int(completion_config.get("min_anchor_residues", 8)),
                    max_anchor_rmsd=float(completion_config.get("max_anchor_rmsd", 8.0)),
                )
                if result is None:
                    row_stats["completion_failed"] += 1
                    return row_stats
                row_stats["completion_written"] += 1
                if result.mode == "crystal_with_alphafold_grafts":
                    row_stats["completion_crystal_grafted"] += 1
                elif result.mode.startswith("alphafold_only"):
                    row_stats["completion_predicted_only"] += 1
                row_stats["_pdb_cap_success"] = 1 if run_complete_callback() else 0
                return row_stats
            except Exception as exc:
                logging.warning("Structure completion failed for %s/%s: %s", uniprot_id, pdb_id or "", exc)
                row_stats["completion_failed"] += 1
                return row_stats

        def merge_stats(row_stats: dict[str, int]) -> None:
            for key, value in row_stats.items():
                if key in stats:
                    stats[key] += value

        if structure_completion_workers <= 1 or len(rows) <= 1:
            progress = tqdm(rows, desc=f"Completing {dataset} structures", unit="protein", dynamic_ncols=True)
            for row in progress:
                merge_stats(complete_row(row))
        else:
            logging.info(
                "Completing %s structures with %d workers (%d rows)",
                dataset,
                structure_completion_workers,
                len(rows),
            )
            with ThreadPoolExecutor(max_workers=structure_completion_workers) as executor:
                futures = [executor.submit(complete_row, row) for row in rows]
                progress = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Completing {dataset} structures ({structure_completion_workers} workers)",
                    unit="protein",
                    dynamic_ncols=True,
                )
                for future in progress:
                    merge_stats(future.result())
        processed += len(rows)
        if limit is not None and processed >= limit:
            break
    logging.info("Structure completion summary: %s", stats)
    return stats


def _load_split_graph_keep_set(config: dict) -> set[str] | None:
    graph_config = config.get("graph", {}) or {}
    if not bool(graph_config.get("write_split_graphs_only", False)):
        return None
    split_dir = Path(str((config.get("storage", {}) or {}).get("split_dir", "")))
    if not str(split_dir) or not split_dir.exists():
        return None
    keep: set[str] = set()
    for path in sorted(split_dir.glob("*.txt")):
        for line in path.read_text(errors="ignore").splitlines():
            name = line.strip()
            if not name:
                continue
            if "," in name:
                name = name.split(",")[0].strip()
            name = Path(name).name
            if name.endswith(".pt"):
                name = name[:-3]
            if name and name.lower() not in {"name", "protein_id", "id", "uniprot_id", "graph"}:
                keep.add(name)
    logging.info("Structure repair split filter enabled: %d keep names from %s", len(keep), split_dir)
    return keep


def repair_missing_predicted_structures(
    config: dict,
    dataset_names: Iterable[str] | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    deployment = config.get("deployment", {}) or {}
    download_config = config.get("structure_download", {}) or {}
    allow_download = bool(deployment.get("allow_download", False))
    allow_predicted = bool(deployment.get("download_predicted_structures", True))
    configured_datasets = deployment.get("download_predicted_structure_datasets", ["external_price149"])
    if isinstance(configured_datasets, str):
        configured_datasets = [configured_datasets]
    configured_dataset_set = set(configured_datasets)

    datasets = config.get("datasets", {}) or {}
    selected = [name for name in list(dataset_names or datasets.keys()) if name in configured_dataset_set]
    save_dir = Path(config["storage"]["save_dir"])
    split_keep_set = _load_split_graph_keep_set(config)
    timeout = float(download_config.get("timeout_seconds", 60))
    download_retries = int(download_config.get("download_retries", 3))
    download_retry_backoff_seconds = float(download_config.get("download_retry_backoff_seconds", 2.0))
    local_alphafold_enabled = bool(download_config.get("local_alphafold_fallback", False))
    local_runner: tuple[str, Path] | None = None
    if local_alphafold_enabled:
        resolved_binary, resolved_params_or_error = _resolve_local_alphafold_runner(download_config)
        if resolved_binary is None:
            logging.warning("Local AlphaFold fallback is disabled: %s", resolved_params_or_error)
        else:
            local_runner = (resolved_binary, resolved_params_or_error)
    max_local_alphafold_length = int(download_config.get("local_alphafold_max_length", 0) or 0)
    skip_for_existing_graph = bool(deployment.get("skip_resource_repair_for_existing_graph", False))

    stats = {
        "structure_existing": 0,
        "structure_downloaded": 0,
        "structure_crystal_downloaded": 0,
        "structure_crystal_not_found": 0,
        "structure_predicted_downloaded": 0,
        "structure_local_predicted": 0,
        "structure_local_prediction_failed": 0,
        "structure_local_prediction_skipped_missing_embedding": 0,
        "structure_local_prediction_skipped_no_sequence": 0,
        "structure_local_prediction_skipped_too_long": 0,
        "structure_missing_download_disabled": 0,
        "structure_not_found": 0,
        "structure_failed_invalid": 0,
        "structure_skipped_no_uniprot": 0,
        "structure_skipped_existing_graph": 0,
        "structure_skipped_not_in_split": 0,
    }
    if not selected:
        logging.info("Predicted structure repair skipped; no selected datasets match %s", sorted(configured_dataset_set))
        logging.info("Predicted structure repair summary: %s", stats)
        return stats
    if not allow_download or not allow_predicted:
        for dataset in selected:
            spec = datasets[dataset]
            output_subdir = spec.get("output_subdir", dataset)
            root = save_dir / output_subdir
            rows = _read_csv_rows(datasets[dataset]["csv"])
            if limit is not None:
                rows = rows[:limit]
            for row in rows:
                if skip_for_existing_graph and _has_prepared_graph(config, dataset, row):
                    stats["structure_skipped_existing_graph"] += 1
                    continue
                uniprot_id = _norm(row.get("uniprot_id"))
                if not uniprot_id:
                    stats["structure_skipped_no_uniprot"] += 1
                    continue
                pdb_id = _norm(row.get("pdb_id"))
                existing_paths = _expected_structure_paths(root, uniprot_id, pdb_id)
                if any(path.exists() and path.stat().st_size > 0 for path in existing_paths):
                    stats["structure_existing"] += 1
                else:
                    stats["structure_missing_download_disabled"] += 1
        logging.info("Predicted structure download is disabled.")
        logging.info("Predicted structure repair summary: %s", stats)
        return stats

    processed = 0
    for dataset in selected:
        spec = datasets[dataset]
        output_subdir = spec.get("output_subdir", dataset)
        root = save_dir / output_subdir
        predicted_dir = root / "predicted"
        rows = _read_csv_rows(spec["csv"])
        if limit is not None:
            rows = rows[: max(limit - processed, 0)]
        progress = tqdm(rows, desc=f"Downloading {dataset} AlphaFold structures", unit="protein", dynamic_ncols=True)
        for row in progress:
            processed += 1
            if skip_for_existing_graph and _has_prepared_graph(config, dataset, row):
                stats["structure_skipped_existing_graph"] += 1
                continue
            uniprot_id = _norm(row.get("uniprot_id"))
            if not uniprot_id:
                stats["structure_skipped_no_uniprot"] += 1
                continue
            pdb_id = _norm(row.get("pdb_id"))
            if split_keep_set is not None and (f"{uniprot_id}_{pdb_id}" if pdb_id else uniprot_id) not in split_keep_set:
                stats["structure_skipped_not_in_split"] += 1
                continue
            existing_paths = _expected_structure_paths(root, uniprot_id, pdb_id)
            if any(path.exists() and path.stat().st_size > 0 for path in existing_paths):
                stats["structure_existing"] += 1
                continue
            if pdb_id:
                if _download_pdb_crystal_structure(
                    root=root,
                    uniprot_id=uniprot_id,
                    pdb_id=pdb_id,
                    download_config=download_config,
                    timeout=timeout,
                    retries=download_retries,
                    retry_backoff_seconds=download_retry_backoff_seconds,
                ):
                    stats["structure_downloaded"] += 1
                    stats["structure_crystal_downloaded"] += 1
                    continue
                stats["structure_crystal_not_found"] += 1
            output_path = predicted_dir / f"{uniprot_id}.cif"
            downloaded = False
            for url in _alphafold_cif_urls(uniprot_id, download_config, timeout=timeout):
                if _download_file(
                    url,
                    output_path,
                    timeout=timeout,
                    retries=download_retries,
                    retry_backoff_seconds=download_retry_backoff_seconds,
                    warn_on_failure=False,
                ):
                    if _valid_structure_payload(output_path):
                        stats["structure_downloaded"] += 1
                        stats["structure_predicted_downloaded"] += 1
                        downloaded = True
                        break
                    output_path.unlink(missing_ok=True)
                    stats["structure_failed_invalid"] += 1
            if not downloaded:
                if local_runner is not None:
                    sequence = _clean_sequence(row.get("sequence"))
                    if not sequence:
                        stats["structure_local_prediction_skipped_no_sequence"] += 1
                    elif max_local_alphafold_length > 0 and len(sequence) > max_local_alphafold_length:
                        stats["structure_local_prediction_skipped_too_long"] += 1
                        logging.info(
                            "Local AlphaFold fallback skipped for %s/%s: sequence length %d exceeds limit %d",
                            uniprot_id,
                            pdb_id or "",
                            len(sequence),
                            max_local_alphafold_length,
                        )
                    else:
                        binary_path, params_dir = local_runner
                        if _run_local_alphafold_prediction(
                            root=root,
                            dataset=dataset,
                            uniprot_id=uniprot_id,
                            pdb_id=pdb_id,
                            sequence=sequence,
                            predicted_dir=predicted_dir,
                            download_config=download_config,
                            binary_path=binary_path,
                            params_dir=params_dir,
                        ):
                            stats["structure_downloaded"] += 1
                            stats["structure_local_predicted"] += 1
                            downloaded = True
                        else:
                            stats["structure_local_prediction_failed"] += 1
                if not downloaded:
                    stats["structure_not_found"] += 1
        if limit is not None and processed >= limit:
            break

    logging.info("Predicted structure repair summary: %s", stats)
    return stats
