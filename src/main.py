from __future__ import annotations
import torch
import pickle
from pathlib import Path
from .config import RunConfig
from .logging_utils import init_logging
from .paths import ensure_dirs
from .preprocess import load_ccd, prepare_inputs
from .model_loader import (
    init_env, build_processed_inputs, trainer_and_writers,
    make_model_args, load_structure_model, load_affinity_model
)
from boltz.data.types import Manifest
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from .run_structure import run_structure_once
from .run_affinity import pred_res_affinity_once
from .io_utils import set_record_affinity
from .traj_runner import run_trajectory_affinity

def main():
    logger = init_logging()
    cfg = RunConfig()

    ensure_dirs(
        cfg.out_dir, cfg.msa_dir, cfg.records_dir, cfg.structure_dir,
        cfg.processed_msa_dir, cfg.constraints_dir, cfg.templates_dir,
        cfg.mols_out_dir, cfg.predictions_dir
    )
    init_env(cfg)

    # Preprocess → manifest
    ccd = load_ccd(cfg.mol_dir, cfg.boltz2)
    manifest = prepare_inputs(
        data_yaml=cfg.data_yaml, ccd=ccd, mol_dir=cfg.mol_dir,
        msa_dir=cfg.msa_dir, processed_msa_dir=cfg.processed_msa_dir,
        constraints_dir=cfg.constraints_dir, templates_dir=cfg.templates_dir,
        mols_out_dir=cfg.mols_out_dir, structure_dir=cfg.structure_dir, records_dir=cfg.records_dir,
        logger=logger
    )

    # Build processed inputs and trainer (writers kept for parity)
    filtered_manifest, processed = build_processed_inputs(cfg, manifest)
    trainer, writer = trainer_and_writers(cfg, processed)

    # Data module for structure
    data_module = Boltz2InferenceDataModule(
        manifest=processed.manifest, target_dir=processed.targets_dir, msa_dir=processed.msa_dir,
        mol_dir=str(cfg.mol_dir), num_workers=2, constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir, extra_mols_dir=processed.extra_mols_dir, override_method=None,
    )

    # Models + args
    diff, pf, msa_args = make_model_args(cfg)
    model_struct = load_structure_model(cfg, cfg.predict_args_structure, diff, pf, msa_args)
    model_aff = load_affinity_model(cfg, cfg.predict_args_affinity, diff, pf, msa_args)

    device = torch.device("cuda")
    feats, dict_out = run_structure_once(
        model=model_struct, data_module=data_module, device=device,
        recycling_steps=cfg.predict_args_structure["recycling_steps"],
        num_sampling_steps=cfg.predict_args_structure["sampling_steps"],
        diffusion_samples=cfg.predict_args_structure["diffusion_samples"],
        max_parallel_samples=cfg.predict_args_structure["max_parallel_samples"],
        structure_dir=cfg.structure_dir, logger=logger,
    )

    # mark ligand chain 1 for pre-affinity
    processed.manifest.records[0] = set_record_affinity(processed.manifest.records[0], chain_id=1)

    # Per-residue affinity (keeps your original sweep)
    results = pred_res_affinity_once(
        cfg=cfg, processed=processed, feats=feats, dict_out=dict_out,
        model_struct=model_struct, model_aff=model_aff, device=device,
    )
    logger.info("Residue-wise affinity results: %s", {k: {kk: float(vv.mean().item()) for kk, vv in val.items()} for k, val in results.items()})

if __name__ == "__main__":
    main()
