from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from .config import RunConfig

def customize(cfg: RunConfig) -> RunConfig:
    """
    Override selected fields of RunConfig.
    Edit the paths/values below to your environment.
    """
    return replace(
        cfg,
        data_yaml=Path("/ABS/PATH/TO/your_design.yaml"),
        out_dir=Path("/ABS/PATH/TO/output/boltz_results_custom").expanduser(),
        cuda_visible_devices="0",          # e.g., "0", "1", "0,1"
        tokenize_res_window=2,             # e.g., 1 or 2
        residue_min=1,                     # inclusive
        residue_max=20,                    # inclusive
    )
