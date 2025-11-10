# affinity_eval/main.py
from __future__ import annotations
from pathlib import Path
from .config import RunConfig
from .logging_utils import init_logging
from .traj_runner import run_trajectory_affinity

# Optional config override
try:
    from .config_overrides import customize as customize_config
except Exception:
    customize_config = None


def main() -> None:
    logger = init_logging()
    cfg = RunConfig()

    # Apply optional per-user overrides
    if customize_config is not None:
        cfg = customize_config(cfg)

    # ---- Fixed trajectory parameters ----
    traj_jsonl = Path("/home/lwang/models/BindCraft/BindCraft_fork/IL23_pepBinder2/opt_log_15.jsonl")
    yaml_template = cfg.data_yaml       # use template in cfg
    every_n = 5                         # run every 5th step
    max_steps = 60                    # or set an int limit

    # ---- Run trajectory affinity prediction ----
    df = run_trajectory_affinity(
        cfg=cfg,
        traj_jsonl=traj_jsonl,
        yaml_template=yaml_template,
        every_n=every_n,
        max_steps=max_steps,
    )

    out_csv = cfg.out_dir / "trajectory_affinity_l15.csv"
    logger.info("Done. Summary CSV saved to: %s", out_csv)
    if not df.empty:
        logger.info("Preview:\n%s", df.head().to_string(index=False))

if __name__ == "__main__":
    main()
