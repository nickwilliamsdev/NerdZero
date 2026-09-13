from sefer.config import V30DGXConfig
from sefer.experiments.synthetic_v30 import run

if __name__ == "__main__":
    cfg = V30DGXConfig(resume_from_algebra=True)
    run(cfg)
