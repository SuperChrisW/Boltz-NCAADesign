# affinity_eval/main.py
from __future__ import annotations
from pathlib import Path
from .config import RunConfig
from .logging_utils import init_logging
from .traj_runner import run_trajectory_affinity
from .paths import ensure_dirs

# Optional config override
try:
    from .config_overrides import customize as customize_config
except Exception:
    customize_config = None


def main() -> None:
    logger = init_logging()
    cfg = RunConfig()
    ensure_dirs(
        cfg.out_dir, cfg.msa_dir, cfg.records_dir, cfg.structure_dir,
        cfg.processed_msa_dir, cfg.constraints_dir, cfg.templates_dir,
        cfg.mols_out_dir, cfg.predictions_dir
    )

    # Apply optional per-user overrides
    if customize_config is not None:
        cfg = customize_config(cfg)

    # ---- Fixed trajectory parameters ----
    # Update these paths to match your trajectory file location
    traj_jsonl = Path("/home/lwang/models/BindCraft/BindCraft_fork/IL23_pepBinder2/opt_log_15.jsonl")
    yaml_template = cfg.data_yaml       # use template in cfg
    every_n = 10                         # run every Nth step (1 = all steps)
    max_steps = None                    # or set an int limit (None for all steps)
    
    # Override residue range to scan ALL residues (set to 0 or negative to auto-detect)
    # If you want specific range, set: cfg.residue_min = 1, cfg.residue_max = <max_length>
    from dataclasses import replace
    cfg = replace(cfg, residue_min=0, residue_max=0)  # 0 means auto-detect from sequence length

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
