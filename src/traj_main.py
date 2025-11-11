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
    # Update these paths to match your trajectory file location
    traj_jsonl = Path("/home/lwang/models/BindCraft/BindCraft_fork/IL23_pepBinder2/opt_log_15.jsonl")
    yaml_template = cfg.data_yaml       # use template in cfg
    every_n = 5                         # run every 5th step
    max_steps = 60                      # or set an int limit (None for all steps)

    logger.info("Starting trajectory affinity prediction")
    logger.info("Trajectory file: %s", traj_jsonl)
    logger.info("YAML template: %s", yaml_template)
    logger.info("Processing every %d steps, max_steps=%s", every_n, max_steps)

    # ---- Run trajectory affinity prediction ----
    df = run_trajectory_affinity(
        cfg=cfg,
        traj_jsonl=traj_jsonl,
        yaml_template=yaml_template,
        every_n=every_n,
        max_steps=max_steps,
    )

    # Summary CSV is saved by run_trajectory_affinity
    out_csv = cfg.out_dir / "trajectory_affinity_summary.csv"
    logger.info("Done. Summary CSV saved to: %s", out_csv)
    
    if not df.empty:
        logger.info("Total rows: %d", len(df))
        logger.info("Preview:\n%s", df.head(10).to_string(index=False))
        logger.info("Summary statistics:")
        logger.info("  Steps processed: %d", df["step"].nunique())
        logger.info("  Residues per step: %d", df.groupby("step")["res_idx"].count().iloc[0] if len(df) > 0 else 0)
        logger.info("  Mean affinity value: %.3f", df["affinity_pred_value_mean"].mean())
        logger.info("  Mean affinity probability: %.3f", df["affinity_probability_binary_mean"].mean())
    else:
        logger.warning("No results generated. Check trajectory file and configuration.")

if __name__ == "__main__":
    main()
