from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List

@dataclass(frozen=True)
class RunConfig:
    # Core paths
    cache: Path = Path("~/.boltz").expanduser()
    data_yaml: Path = Path("/home/lwang/models/boltz_inference/boltz_work/examples/IL23_design.yaml")
    out_dir: Path = Path("/home/lwang/models/boltz_inference/boltz_work/output/test/boltz_results_affinityEval").expanduser()
    mol_dir: Path = Path("/home/lwang/.boltz/mols")

    # Hardware
    cuda_visible_devices: str = "3"
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
    tokenize_res_window: int = 0

    # Residue sweep
    residue_min: int = 6
    residue_max: int = 6

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
