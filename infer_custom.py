import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from rdkit.Chem import Mol
from tqdm import tqdm

from dataset import get_dataset_dataloader, load
from models import DynamicBind
from utils_dynamic_bind import DynamicBindConfig, dic_to_device, initialize_model

# In training/inference pipeline, the model score is converted to pKd by
# dividing by this factor (see also inference.py: pred_affinity = score / -1.36).
SCORE_TO_PKD_DIVISOR = -1.36


def resolve_row_key(row: Dict[str, str], idx: int) -> str:
    """Return a stable key for a CSV row, generating one when missing."""
    raw_key = (row.get("key") or "").strip()
    if raw_key:
        return raw_key
    auto_key = row.get("_auto_key")
    if auto_key:
        return auto_key
    auto_key = f"row_{idx}"
    row["_auto_key"] = auto_key
    return auto_key


def read_pairs_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"key", "ligand_path", "target_path"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must contain columns: {sorted(required_cols)}. "
                f"Got: {reader.fieldnames}"
            )
        return list(reader)


def build_screening_list(rows: List[Dict[str, str]]) -> Tuple[List[Tuple[Mol, Mol, str, str]], Dict[str, str]]:
    screening_list = []
    status_map: Dict[str, str] = {}

    for idx, row in enumerate(rows):
        key = resolve_row_key(row, idx)
        ligand_path = row["ligand_path"].strip()
        target_path = row["target_path"].strip()

        if not os.path.exists(ligand_path):
            status_map[key] = f"fail: ligand_not_found ({ligand_path})"
            continue
        if not os.path.exists(target_path):
            status_map[key] = f"fail: target_not_found ({target_path})"
            continue

        try:
            ref_mol, tar_mol, _, _ = load(ligand_path, target_path)
        except Exception as e:
            status_map[key] = f"fail: load_exception ({e})"
            continue

        if not isinstance(ref_mol, Mol) or not isinstance(tar_mol, Mol):
            status_map[key] = "fail: rdkit_sanitize_failed"
            continue

        screening_list.append((ref_mol, tar_mol, key, ""))
        status_map[key] = "ok"

    return screening_list, status_map


def run_prediction(train_dir: str, screening_list: List[Tuple[Mol, Mol, str, str]], batch_size: int, num_workers: int) -> Dict[str, float]:
    model_conf_path = os.path.join(train_dir, "configs.yml")
    if not os.path.exists(model_conf_path):
        raise FileNotFoundError(f"Missing model config: {model_conf_path}")

    model_weight = os.path.join(train_dir, "save_best.pt")
    if not os.path.exists(model_weight):
        model_weight = os.path.join(train_dir, "save_last.pt")
    if not os.path.exists(model_weight):
        raise FileNotFoundError(f"Missing model weights in {train_dir}")

    model_cfg: DynamicBindConfig = OmegaConf.structured(DynamicBindConfig)
    model_cfg = OmegaConf.merge(model_cfg, OmegaConf.load(model_conf_path))

    _, dataloader = get_dataset_dataloader(
        screening_list=screening_list,
        id_to_y={},
        batch_size=batch_size,
        num_workers=num_workers,
        train=False,
    )

    device = torch.device(f"cuda:{model_cfg.ngpu}" if torch.cuda.is_available() else "cpu")
    model = DynamicBind(model_cfg)
    model = initialize_model(model, device, model_weight, strict=True)
    model.eval()

    pred_map: Dict[str, float] = {}
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Predicting")
    for _, sample in pbar:
        if sample[0] is None:
            continue
        sample_score = dic_to_device(sample[0], device)
        # MD-trained models may require trajectory coordinates when
        # whole_traj_learning=True. For custom inference without real MD
        # trajectories, we replicate static coordinates across frames.
        if getattr(model_cfg, "whole_traj_learning", False):
            n_frames = int(getattr(model_cfg, "n_total_frames", 50))
            lig_pos = sample_score["ligand_pos"]      # [B, N_lig, 3]
            tar_pos = sample_score["target_pos"]      # [B, N_tar, 3]
            sample_score["lig_traj_coords"] = lig_pos.unsqueeze(1).repeat(1, n_frames, 1, 1)
            sample_score["target_traj_coords"] = tar_pos.unsqueeze(1).repeat(1, n_frames, 1, 1)
        keys = sample_score["key"]
        with torch.no_grad():
            pred = model(sample_score)[0].detach().cpu().numpy()
        for i, key in enumerate(keys):
            pred_map[key] = float(np.sum(pred[i]))

    return pred_map


def write_output(out_csv: str, rows: List[Dict[str, str]], pred_map: Dict[str, float], status_map: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "score", "pKd_est", "status"])
        for idx, row in enumerate(rows):
            key = resolve_row_key(row, idx)
            if key in pred_map:
                score = pred_map[key]
                writer.writerow([key, f"{score:.6f}", f"{(score / SCORE_TO_PKD_DIVISOR):.6f}", "ok"])
            else:
                writer.writerow([key, "", "", status_map.get(key, "fail: unknown")])


def main():
    parser = argparse.ArgumentParser(description="Batch inference for custom protein-ligand pairs.")
    parser.add_argument("train_dir", type=str, help="Trained model directory, e.g. results/casf_no_md")
    parser.add_argument("--pairs_csv", type=str, required=True, help="CSV with columns: key,ligand_path,target_path")
    parser.add_argument("--out_csv", type=str, default="custom_scores.csv", help="Output prediction CSV path")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")
    args = parser.parse_args()

    rows = read_pairs_csv(args.pairs_csv)
    screening_list, status_map = build_screening_list(rows)

    print(f"[INFO] Input rows: {len(rows)}")
    print(f"[INFO] Valid pairs: {len(screening_list)}")
    print(f"[INFO] Invalid pairs: {len(rows) - len(screening_list)}")

    pred_map: Dict[str, float] = {}
    if len(screening_list) > 0:
        pred_map = run_prediction(
            train_dir=args.train_dir,
            screening_list=screening_list,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    write_output(args.out_csv, rows, pred_map, status_map)
    print(f"[DONE] Wrote predictions to: {args.out_csv}")


if __name__ == "__main__":
    main()
