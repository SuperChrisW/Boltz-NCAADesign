from __future__ import annotations
import os
import yaml
import torch
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import replace as dataclass_replace
from .config import load_run_config
from .logging_utils import init_logging
from .paths import ensure_dirs
from .preprocess import load_ccd, prepare_inputs
from .model_loader import (
    init_env, build_processed_inputs, trainer_and_writers,
    make_model_args, load_structure_model, load_affinity_model, create_ligand_template_module,
)
from boltz.data.types import Manifest
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from .run_structure import run_structure_once
from .run_affinity import pred_res_affinity_once
from .io_utils import set_record_affinity
from .traj_runner import run_trajectory_affinity

def prepare_environment(cfg, logger):
        ensure_dirs(
            cfg.out_dir, cfg.msa_dir, cfg.records_dir, cfg.structure_dir,
            cfg.processed_msa_dir, cfg.constraints_dir, cfg.templates_dir,
            cfg.mols_out_dir, cfg.predictions_dir
        )
        init_env(cfg)
        ccd = load_ccd(cfg.mol_dir, cfg.boltz2)
        manifest = prepare_inputs(
            data_yaml=cfg.data_yaml, ccd=ccd, mol_dir=cfg.mol_dir,
            msa_dir=cfg.msa_dir, processed_msa_dir=cfg.processed_msa_dir,
            constraints_dir=cfg.constraints_dir, templates_dir=cfg.templates_dir,
            mols_out_dir=cfg.mols_out_dir, structure_dir=cfg.structure_dir, records_dir=cfg.records_dir,
            logger=logger
        )
        return manifest

def build_models_and_data(cfg, processed):
    # Build processed inputs and trainer (writers kept for parity)
    trainer, writer = trainer_and_writers(cfg, processed)

    # Data module for structure
    data_module = Boltz2InferenceDataModule(
        manifest=processed.manifest, target_dir=processed.targets_dir, msa_dir=processed.msa_dir,
        mol_dir=str(cfg.mol_dir), num_workers=2, constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir, extra_mols_dir=processed.extra_mols_dir, override_method=None,
    )

    device = torch.device("cuda")
    template_module = create_ligand_template_module()

    # Models + args
    diff, pf, msa_args = make_model_args(cfg)
    model_struct = load_structure_model(cfg, cfg.predict_args_structure, diff, pf, msa_args)
    model_aff = load_affinity_model(cfg, cfg.predict_args_affinity, diff, pf, msa_args)

    return trainer, writer, data_module, device, template_module, model_struct, model_aff

def infer_structure(cfg, model_struct, data_module, device, logger):
    feats, dict_out = run_structure_once(
        model=model_struct, data_module=data_module, device=device,
        recycling_steps=cfg.predict_args_structure["recycling_steps"],
        num_sampling_steps=cfg.predict_args_structure["sampling_steps"],
        diffusion_samples=cfg.predict_args_structure["diffusion_samples"],
        max_parallel_samples=cfg.predict_args_structure["max_parallel_samples"],
        structure_dir=cfg.structure_dir, logger=logger,
    )
    return feats, dict_out

def run_affinity_pipeline(cfg, logger):
    manifest = prepare_environment(cfg, logger)
    filtered_manifest, processed = build_processed_inputs(cfg, manifest)
    trainer, writer, data_module, device, template_module, model_struct, model_aff = build_models_and_data(cfg, processed)
    feats, dict_out = infer_structure(cfg, model_struct, data_module, device, logger)

    # mark ligand chain 1 for pre-affinity
    processed.manifest.records[0] = set_record_affinity(processed.manifest.records[0], chain_id=1)
    binder_len = processed.manifest.records[0].chains[1].num_residues
    if cfg.residue_min <= 0 or cfg.residue_max <= 0:
        start_idx = 1
        end_idx = binder_len
        print(f"[INFO] Auto-scanning all {binder_len} residues (1-{binder_len})")
    else:
        start_idx = max(1, cfg.residue_min)
        end_idx = min(binder_len, cfg.residue_max)
        if start_idx > end_idx:
            raise ValueError(f"Binder length {binder_len} shorter than residue_min={cfg.residue_min}.")

    print(f"[INFO] Processing residues {start_idx} to {end_idx} (total: {end_idx - start_idx + 1})")

    # Make a shallow copy of cfg for per-step overrides
    cfg_step = dataclass_replace(cfg, residue_min=start_idx, residue_max=end_idx)

    # Per-residue affinity (keeps your original sweep)
    results = pred_res_affinity_once(
        cfg=cfg_step, processed=processed, feats=feats, dict_out=dict_out,
        model_struct=model_struct, model_aff=model_aff, device=device, template_module=template_module, save_structure=True # provide template_module
    )
    logger.info(
        "Residue-wise affinity results: %s", 
        {k: {kk: float(vv.mean().item()) for kk, vv in val.items()} for k, val in results.items()}
    )
    return results

def launch_folder_multitask(
    target_dir: str | Path,
    base_cfg_loader=load_run_config,
    affinity_pipeline=run_affinity_pipeline,
    logger=None,
    *,
    parallel: bool = False,
    dry_run: bool = False,
):
    """
    Launch affinity pipeline for all (pdb_id/mut_label) folders in target_dir.
    Each mut_label contains complex.yaml (for sequence input) and optionally run_config.yaml (for cfg override).
    """
    from copy import deepcopy

    target_dir = Path(target_dir)
    # Collect tasks by traversing two-levels deep
    jobs = []
    for pdb_dir in sorted(target_dir.iterdir()):
        if not pdb_dir.is_dir():
            continue
        for mut_dir in sorted(pdb_dir.iterdir()):
            if not mut_dir.is_dir():
                continue
            complex_yaml = mut_dir / "complex.yaml"
            run_config_yaml = mut_dir / "run_config.yaml"
            if not complex_yaml.exists():
                if logger:
                    logger.warning(f"[SKIP] {mut_dir}: missing complex.yaml")
                continue
            jobs.append((pdb_dir.name, mut_dir.name, str(complex_yaml), run_config_yaml if run_config_yaml.exists() else None))

    #logger = logger or print
    logger.info(f"Found {len(jobs)} subfolders for processing.")
    
    def _load_override(run_config_yaml):
        if not run_config_yaml:
            return {}
        with open(run_config_yaml, "r") as f:
            return yaml.safe_load(f)
    
    def _process_job(job):
        pdb_id, mut_label, complex_yaml, run_config_yaml = job
        overrides = _load_override(run_config_yaml)

        cfg = base_cfg_loader(config_path=None, overrides=dict(**(overrides or {})))
        logger.info(f"[LAUNCH] {pdb_id}/{mut_label} → out_dir: {cfg.out_dir}")
        try:
            return affinity_pipeline(cfg, logger)
        except Exception as exc:
            logger.error(f"[ERROR] {pdb_id}/{mut_label} failed: {exc}")
            return None
    
    if parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(_process_job, job) for job in jobs]
            for fut in as_completed(futures):
                fut.result()  # propagate errors
    else:
        results = {}
        for job in jobs:
            pdb_id, mut_label, complex_yaml, run_config_yaml = job
            affinity_out = _process_job(job)

            # Convert any tensor values in affinity_out to numpy, detach, and move to cpu
            def tensor_to_numpy(val):
                if isinstance(val, torch.Tensor):
                    return val.detach().cpu().numpy()
                return val

            if isinstance(affinity_out, dict):
                affinity_out = {
                    k: tensor_to_numpy(v) if not isinstance(v, dict) else {
                        sk: tensor_to_numpy(sv) for sk, sv in v.items()
                    }
                    for k, v in affinity_out.items()
                }

            results[(pdb_id, mut_label)] = affinity_out
        return results

def main():
    logger = init_logging()
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    target_dir = "/home/lwang/models/boltz_inference/scripts/affinity_eval/test/PPI_predict/run_files"
    save_dir = "/home/lwang/models/boltz_inference/scripts/affinity_eval/test/PPI_predict"
    results = launch_folder_multitask(
        target_dir,
        logger=logger,
        parallel=False,
    )

    # Save the `results` dictionary as a pickle file for the whole job
    with open(f"{save_dir}/affinity_results.pkl", "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Saved full results to {save_dir}/affinity_results.pkl")

    rows = []
    for (pdb_id, mut_label), affinity_out in (results or {}).items():
        if affinity_out is None:
            continue
        for residue, resvals in (affinity_out or {}).items():
            row = {
                "pdb_id": pdb_id,
                "mut_label": mut_label,
                "residue": residue,
            }
            if isinstance(resvals, dict):
                row.update(resvals)
            rows.append(row)
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(f"{save_dir}/affinity_results.csv", index=False)
    logger.info(f"Saved affinity results to {save_dir}/affinity_results.csv")

if __name__ == "__main__":
    main()
