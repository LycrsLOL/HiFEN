from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np


def _read_fasta(path: Path) -> tuple[str, str]:
    description = path.stem
    sequence_parts: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                description = text[1:].strip() or description
            else:
                sequence_parts.append(text)
    sequence = "".join(sequence_parts).upper().replace("*", "")
    if not sequence:
        raise ValueError(f"No sequence found in FASTA: {path}")
    return description, sequence


def _make_empty_template_features(num_res: int) -> dict[str, np.ndarray]:
    return {
        "template_aatype": np.zeros((0, num_res, 22), dtype=np.float32),
        "template_all_atom_masks": np.zeros((0, num_res, 37), dtype=np.float32),
        "template_all_atom_positions": np.zeros((0, num_res, 37, 3), dtype=np.float32),
        "template_domain_names": np.array([], dtype=np.object_),
        "template_sequence": np.array([], dtype=np.object_),
        "template_sum_probs": np.array([], dtype=np.float32),
    }


def _load_params(params_dir: Path, model_name: str):
    from alphafold.model import utils

    params_path = params_dir / f"params_{model_name}.npz"
    if not params_path.exists():
        raise FileNotFoundError(f"AlphaFold params file does not exist: {params_path}")
    params = np.load(params_path, allow_pickle=False)
    return utils.flat_params_to_haiku(params, fuse=True, to_jnp=True)


def predict_single_sequence(
    *,
    fasta_path: Path,
    params_dir: Path,
    output_path: Path,
    model_name: str = "model_1_ptm",
    num_recycle: int = 3,
    random_seed: int = 0,
) -> None:
    from alphafold.common import protein
    from alphafold.data import parsers
    from alphafold.data import pipeline
    from alphafold.model import config as model_config_lib
    from alphafold.model import model as model_lib

    target_id, sequence = _read_fasta(fasta_path)
    num_res = len(sequence)
    raw_features = {
        **pipeline.make_sequence_features(sequence, target_id, num_res),
        **pipeline.make_msa_features(
            [
                parsers.Msa(
                    sequences=[sequence],
                    deletion_matrix=[[0] * num_res],
                    descriptions=[target_id],
                )
            ]
        ),
        **_make_empty_template_features(num_res),
    }

    model_config = model_config_lib.model_config(model_name)
    model_config.model.num_recycle = int(num_recycle)
    model_config.data.eval.num_ensemble = 1
    params = _load_params(params_dir, model_name)
    runner = model_lib.RunModel(model_config, params)
    processed_features = runner.process_features(raw_features, random_seed=random_seed)
    prediction, _ = runner.predict(processed_features, random_seed=random_seed)
    b_factors = prediction["plddt"][:, None] * prediction["structure_module"]["final_atom_mask"]
    unrelaxed = protein.from_prediction(processed_features, prediction, b_factors=b_factors)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(protein.to_pdb(unrelaxed), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local AlphaFold single-sequence prediction with local params.")
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--params-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="model_1_ptm")
    parser.add_argument("--num-recycle", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    predict_single_sequence(
        fasta_path=Path(args.fasta),
        params_dir=Path(args.params_dir),
        output_path=Path(args.output),
        model_name=args.model_name,
        num_recycle=args.num_recycle,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
