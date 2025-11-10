from __future__ import annotations
import os
from dataclasses import asdict
from typing import List, Union, Tuple
import platform
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from boltz.model.models.boltz2 import Boltz2
from boltz.main import (
    Boltz2DiffusionParams, PairformerArgsV2, MSAModuleArgs,
    BoltzWriter, BoltzAffinityWriter, BoltzSteeringParams,
    filter_inputs_structure, filter_inputs_affinity, BoltzProcessedInput,
)
from .config import RunConfig
from pathlib import Path
from typing import Optional
from .template_constraints import LigandTemplateModule, LigandTemplateLoader

def init_env(cfg: RunConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    seed_everything(42)

def build_processed_inputs(cfg: RunConfig, manifest):
    filtered_manifest = filter_inputs_structure(manifest=manifest, outdir=cfg.out_dir, override=True)
    processed = BoltzProcessedInput(
        manifest=filtered_manifest,
        targets_dir=cfg.structure_dir,
        msa_dir=cfg.processed_msa_dir,
        constraints_dir=cfg.constraints_dir if cfg.constraints_dir.exists() else None,
        template_dir=cfg.templates_dir if cfg.templates_dir.exists() else None,
        extra_mols_dir=cfg.mols_out_dir if cfg.mols_out_dir.exists() else None,
    )
    return filtered_manifest, processed

def trainer_and_writers(cfg: RunConfig, processed):
    writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=cfg.predictions_dir,
        output_format="mmcif",
        boltz2=True,
        write_embeddings=True,
    )
    trainer = Trainer(
        default_root_dir=cfg.out_dir,
        strategy="auto",
        callbacks=[writer],
        accelerator="gpu",
        devices=cfg.devices,
        precision=cfg.precision,
    )
    return trainer, writer

def make_model_args(cfg: RunConfig):
    diff = Boltz2DiffusionParams()
    diff.step_scale = 1.5
    pf = PairformerArgsV2()
    msa_args = MSAModuleArgs(subsample_msa=False, num_subsampled_msa=1024, use_paired_feature=True)
    return diff, pf, msa_args

def load_structure_model(cfg: RunConfig, predict_args: dict, diff, pf, msa_args):
    model = Boltz2.load_from_checkpoint(
        cfg.conf_ckpt, strict=True, predict_args=predict_args, map_location="cpu",
        diffusion_process_args=asdict(diff), ema=False, use_kernels=True,
        pairformer_args=asdict(pf), msa_args=asdict(msa_args), steering_args=asdict(BoltzSteeringParams(fk_steering=False, physical_guidance_update=False)),
    )
    model.eval()
    return model

def load_affinity_model(cfg: RunConfig, predict_args: dict, diff, pf, msa_args):
    aff_ckpt = cfg.aff_ckpt or (cfg.cache / "boltz2_aff.ckpt")
    model = Boltz2.load_from_checkpoint(
        aff_ckpt, strict=True, predict_args=predict_args, map_location="cpu",
        diffusion_process_args=asdict(diff), ema=False,
        pairformer_args=asdict(pf), msa_args=asdict(msa_args),
        steering_args=asdict(BoltzSteeringParams(fk_steering=False, physical_guidance_update=False, contact_guidance_update=False)),
        affinity_mw_correction=True,
    )
    model.eval()
    return model

def create_ligand_template_module(
    token_z: int = 128,
    template_dim: int = 128,
    template_blocks: int = 2,
    device: Optional[torch.device] = None,
) -> LigandTemplateModule:
    """
    Create and initialize a ligand template module.

    Parameters
    ----------
    token_z : int
        Token pairwise embedding dimension (should match model's token_z).
    template_dim : int
        Template feature processing dimension.
    template_blocks : int
        Number of pairformer blocks for template processing.
    device : torch.device, optional
        Device to place the module on.

    Returns
    -------
    LigandTemplateModule
        Initialized ligand template module.
    """
    module = LigandTemplateModule(
        token_z=token_z,
        template_dim=template_dim,
        template_blocks=template_blocks,
    )
    if device is not None:
        module = module.to(device)
    module.eval()
    return module

def load_template_coords(
    template_path: Path,
    template_dir: Optional[Path] = None,
    protein_key: str = "protein_coords",
    ligand_key: str = "ligand_coords",
) -> dict:
    """
    Load template coordinates from .npz file.

    Parameters
    ----------
    template_path : Path
        Path to the template .npz file.
    template_dir : Path, optional
        Directory to resolve relative paths from.
    protein_key : str
        Key for protein coordinates in .npz file.
    ligand_key : str
        Key for ligand coordinates in .npz file.

    Returns
    -------
    dict
        Dictionary containing loaded coordinates and masks.
    """
    loader = LigandTemplateLoader(
        template_dir=template_dir,
        protein_key=protein_key,
        ligand_key=ligand_key,
    )
    return loader.load_template(template_path)
