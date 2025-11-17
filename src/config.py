from __future__ import annotations
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Optional, Union, List, Mapping, Any

import yaml

@dataclass(frozen=True)
class RunConfig:
    # Core paths
    cache: Path = Path("~/.boltz").expanduser()
    data_yaml: Path = Path("/home/lwang/models/boltz_inference/boltz_work/examples/IL23_design.yaml")
    out_dir: Path = Path("/home/lwang/models/boltz_inference/scripts/affinity_eval/test/BindCraft_traj/designs/DerF21/opt15_traj").expanduser()
    mol_dir: Path = Path("/home/lwang/.boltz/mols")

    # Hardware
    cuda_visible_devices: str = "0"
    devices: Union[int, List[int]] = field(default_factory=lambda: [0])
    precision: str = "bf16-mixed"

    # Toggles
    boltz2: bool = True

    # Predict (structure)
    predict_args_structure: dict = field(default_factory=lambda: dict(
        recycling_steps=5, sampling_steps=200, diffusion_samples=1,
        max_parallel_samples=1, write_confidence_summary=True,
        write_full_pae=False, write_full_pde=False,
    ))

    # Predict (affinity)
    predict_args_affinity: dict = field(default_factory=lambda: dict(
        recycling_steps=5, sampling_steps=200, diffusion_samples=1,
        max_parallel_samples=1, write_confidence_summary=False,
        write_full_pae=False, write_full_pde=False,
    ))

    # Checkpoints
    conf_ckpt: Path = Path("~/.boltz/boltz2_conf.ckpt").expanduser()
    aff_ckpt: Optional[Path] = None  # if None → ~/.boltz/boltz2_aff.ckpt

    # Crop limits
    affinity_max_tokens: int = 256
    affinity_max_atoms: int = 2048

    # Ligand chain & window
    ligand_chain_id: int = 1  # hardcoded binder chain index (unchanged)
    receptor_chain_id: int = 0
    tokenize_res_window: int = 1

    # Residue sweep
    residue_min: int = 0
    residue_max: int = 0

    # MSA/processed dirs
    @property
    def msa_dir(self) -> Path: return self.cache / "msa"
    @property
    def processed_dir(self) -> Path: return self.out_dir / "processed"
    @property
    def records_dir(self) -> Path: return self.processed_dir / "records"
    @property
    def structure_dir(self) -> Path: return self.processed_dir / "structures"
    @property
    def processed_msa_dir(self) -> Path: return self.processed_dir / "msa"
    @property
    def constraints_dir(self) -> Path: return self.processed_dir / "constraints"
    @property
    def templates_dir(self) -> Path: return self.processed_dir / "templates"
    @property
    def mols_out_dir(self) -> Path: return self.processed_dir / "mols"
    @property
    def predictions_dir(self) -> Path: return self.out_dir / "predictions"


_PATH_FIELDS = {"cache", "data_yaml", "out_dir", "mol_dir", "conf_ckpt", "aff_ckpt"}
_PREDICT_ARGS = {"predict_args_structure", "predict_args_affinity"}


def _coerce_value(field_name: str, value: Any) -> Any:
    if field_name in _PATH_FIELDS and isinstance(value, str):
        return Path(value).expanduser()
    return value


def _merge_predict_args(current: dict, override: Mapping[str, Any]) -> dict:
    merged = dict(current)
    for key, val in override.items():
        merged[key] = val
    return merged


def load_run_config(config_path: Optional[Union[str, Path]] = None,
                    overrides: Optional[Mapping[str, Any]] = None) -> RunConfig:
    """
    Build a RunConfig, optionally overriding fields via YAML file and/or mapping.

    If config_path is not provided the loader will look for a BOLTZ_RUN_CONFIG
    environment variable.
    """
    cfg = RunConfig()
    path = config_path or os.environ.get("BOLTZ_RUN_CONFIG")
    update_data: dict[str, Any] = {}

    if path:
        path_obj = Path(path).expanduser()
        if not path_obj.exists():
            raise FileNotFoundError(f"Config override file not found: {path_obj}")
        with path_obj.open("r", encoding="utf-8") as handle:
            file_overrides = yaml.safe_load(handle) or {}
        if not isinstance(file_overrides, Mapping):
            raise TypeError(f"Override file must contain a mapping, got {type(file_overrides)}")
        update_data.update(file_overrides)

    if overrides:
        update_data.update(overrides)

    if not update_data:
        return cfg

    valid_fields = {f.name for f in fields(RunConfig)}
    apply_kwargs: dict[str, Any] = {}
    for key, value in update_data.items():
        if key not in valid_fields:
            raise KeyError(f"Unknown RunConfig field '{key}' in overrides")
        if key in _PREDICT_ARGS and isinstance(value, Mapping):
            current = getattr(cfg, key)
            value = _merge_predict_args(current, value)
        value = _coerce_value(key, value)
        apply_kwargs[key] = value

    return replace(cfg, **apply_kwargs)
