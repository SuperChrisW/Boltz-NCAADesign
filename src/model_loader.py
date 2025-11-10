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
