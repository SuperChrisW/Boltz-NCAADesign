import logging, warnings

def init_logging() -> logging.Logger:
    warnings.filterwarnings("ignore", category=UserWarning, module="rdkit")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    )
    return logging.getLogger("affinity_eval")
