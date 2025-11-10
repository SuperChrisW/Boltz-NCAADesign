from __future__ import annotations
import json
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from .config import RunConfig
from .preprocess import load_ccd, prepare_inputs
from .model_loader import (
    init_env, build_processed_inputs, make_model_args,
    load_structure_model, load_affinity_model,
)
from .run_structure import run_structure_once
from .run_affinity import pred_res_affinity_once
from .io_utils import set_record_affinity


def _render_yaml_with_binder(template_yaml: Path, out_yaml: Path, binder_seq: str) -> None:
    """
    Load a template YAML and overwrite chain B's sequence with `binder_seq`.
    Assumes sequences[1] is the binder (protein id: B).
    """
    with open(template_yaml, "r") as f:
        data = yaml.safe_load(f)

    # Basic guard: sequences[1] exists and is a protein with id 'B'
    if not isinstance(data.get("sequences"), list) or len(data["sequences"]) < 2:
        raise ValueError("YAML template must have at least two sequences (A and B).")

    # Mutate chain B sequence
    seqB = data["sequences"][1]
    if "protein" not in seqB:
        raise ValueError("YAML template sequences[1] must be a protein entry for chain B.")
    if "id" in seqB["protein"] and seqB["protein"]["id"] not in ("B", "b"):
        # Still allow replacing the second entry regardless of ID, but warn.
        pass
    seqB["protein"]["sequence"] = str(binder_seq)

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _load_trajectory(jsonl_path: Path) -> pd.DataFrame:
    """Load your optimization log into a tidy DataFrame (same as your snippet)."""
    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(records)

    def mean_or_nan(x):
        if isinstance(x, (list, np.ndarray)):
            arr = np.array(x, dtype=float)
            return float(np.nanmean(arr)) if arr.size > 0 else np.nan
        return np.nan

    if "mean_pae" not in df.columns:
        df["mean_pae"] = df.get("pae_raw", np.nan).apply(mean_or_nan)
    if "mean_plddt" not in df.columns:
        df["mean_plddt"] = df.get("plddt_mean", np.nan)
    if "iptm" not in df.columns:
        df["iptm"] = df.get("iptm", np.nan)
    df = df[["step", "binder_seq", "iptm", "mean_plddt", "mean_pae"]].sort_values("step").reset_index(drop=True)
    return df


def run_trajectory_affinity(
    *,
    cfg: RunConfig,
    traj_jsonl: Path,
    yaml_template: Path,
    every_n: int = 1,           # set to 5 to run every 5th step
    max_steps: Optional[int] = None,  # limit how many steps to run
) -> pd.DataFrame:
    """
    For each selected step in the sequence optimization trajectory:
      1) write a YAML with chain B = binder_seq,
      2) preprocess → run Boltz2 structure once,
      3) run residue-wise affinity,
      4) save per-step raw outputs and collect summary rows.

    Returns a DataFrame with affinity metrics per step.
    """
    # ---- one-time init / models (reuse across steps) ----
    init_env(cfg)
    ccd = load_ccd(cfg.mol_dir, cfg.boltz2)

    # Load original trajectory
    df = _load_trajectory(Path(traj_jsonl))
    if max_steps is not None:
        df = df.head(max_steps)

    # Prepare output dirs
    tmp_yaml_dir = cfg.out_dir / "tmp_yaml"
    tmp_yaml_dir.mkdir(parents=True, exist_ok=True)

    # Load model args + models once
    diff, pf, msa_args = make_model_args(cfg)
    model_struct = load_structure_model(cfg, cfg.predict_args_structure, diff, pf, msa_args)
    model_aff = load_affinity_model(cfg, cfg.predict_args_affinity, diff, pf, msa_args)
    device = torch.device("cuda")

    # Per-step results
    result_rows: List[Dict[str, Any]] = []

    # Iterate trajectory
    for _, row in df.iterrows():
        step = int(row["step"])
        binder_seq = str(row["binder_seq"])

        # Skip if not on stride
        if (step % every_n) != 0:
            continue

        # Prepare step-specific YAML
        step_yaml = tmp_yaml_dir / f"step_{step}.yaml"
        _render_yaml_with_binder(yaml_template, step_yaml, binder_seq)

        # Preprocess for this YAML (manifest/processed)
        manifest = prepare_inputs(
            data_yaml=step_yaml,
            ccd=ccd,
            mol_dir=cfg.mol_dir,
            msa_dir=cfg.msa_dir,
            processed_msa_dir=cfg.processed_dir / "msa",
            constraints_dir=cfg.constraints_dir,
            templates_dir=cfg.templates_dir,
            mols_out_dir=cfg.mols_out_dir,
            structure_dir=cfg.structure_dir,
            records_dir=cfg.records_dir,
            logger=None,
        )

        # Processed inputs for this manifest
        filtered_manifest, processed = build_processed_inputs(cfg, manifest)

        # Data module for structure (recreate to pick up new manifest)
        from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
        data_module = Boltz2InferenceDataModule(
            manifest=processed.manifest,
            target_dir=processed.targets_dir,
            msa_dir=processed.msa_dir,
            mol_dir=str(cfg.mol_dir),
            num_workers=2,
            constraints_dir=processed.constraints_dir,
            template_dir=processed.template_dir,
            extra_mols_dir=processed.extra_mols_dir,
            override_method=None,
        )

        # -- STRUCTURE --
        feats, dict_out = run_structure_once(
            model=model_struct,
            data_module=data_module,
            device=device,
            recycling_steps=cfg.predict_args_structure["recycling_steps"],
            num_sampling_steps=cfg.predict_args_structure["sampling_steps"],
            diffusion_samples=cfg.predict_args_structure["diffusion_samples"],
            max_parallel_samples=cfg.predict_args_structure["max_parallel_samples"],
            structure_dir=cfg.structure_dir,
            logger=None,
        )
        processed.manifest.records[0] = set_record_affinity(processed.manifest.records[0], chain_id=1)

        # -- AFFINITY (residue sweep) --
        # adjust sweep range based on actual binder length
        binder_len = len(binder_seq)
        start_idx = max(1, cfg.residue_min)
        end_idx = min(binder_len, cfg.residue_max)
        if start_idx > end_idx:
            print(f"[WARN] Binder length {binder_len} shorter than residue_min={cfg.residue_min}; skipping step {step}.")
            continue

        # make a shallow copy of cfg for per-step overrides
        from dataclasses import replace as dataclass_replace
        cfg_step = dataclass_replace(cfg, residue_min=start_idx, residue_max=end_idx)

        results = pred_res_affinity_once(
            cfg=cfg_step,
            processed=processed,
            feats=feats,
            dict_out=dict_out,
            model_struct=model_struct,
            model_aff=model_aff,
            device=device,
        )

        # Detach/cpu for saving
        def _cpu_detach(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu()
            if isinstance(obj, dict):
                return {k: _cpu_detach(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_cpu_detach(v) for v in obj]
            return obj

        dict_out_cpu = _cpu_detach(dict_out)
        results_cpu = _cpu_detach(results)

        # Save raw dumps for this step
        with open(cfg.out_dir / f"dict_out_step{step}.pkl", "wb") as f:
            pickle.dump(dict_out_cpu, f)
        with open(cfg.out_dir / f"affinity_out_step{step}.pkl", "wb") as f:
            pickle.dump(results_cpu, f)

        # Summarize affinity over residues (you can customize how you aggregate)
        # Here we record per-residue mean values.
        for ridx, vals in results_cpu.items():
            row_out = {
                "step": step,
                "binder_seq": binder_seq,
                "res_idx": int(ridx),
                "affinity_pred_value_mean": float(vals["affinity_pred_value"].mean().item()),
                "affinity_probability_binary_mean": float(vals["affinity_probability_binary"].mean().item()),
                "iptm_logged": float(row.get("iptm", np.nan)) if not pd.isna(row.get("iptm", np.nan)) else np.nan,
                "mean_plddt_logged": float(row.get("mean_plddt", np.nan)) if not pd.isna(row.get("mean_plddt", np.nan)) else np.nan,
                "mean_pae_logged": float(row.get("mean_pae", np.nan)) if not pd.isna(row.get("mean_pae", np.nan)) else np.nan,
            }
            result_rows.append(row_out)

    # Final table
    out_df = pd.DataFrame(result_rows).sort_values(["step", "res_idx"]).reset_index(drop=True)

    # Write CSV
    out_df.to_csv(cfg.out_dir / "trajectory_affinity_summary.csv", index=False)
    return out_df
