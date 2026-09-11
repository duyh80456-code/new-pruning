from __future__ import annotations

import argparse
from pathlib import Path

from geometry.analysis import analyze_seed
from research_utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute geometry analysis for one completed seed")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_dir = Path(config["experiment"]["output_dir"]) / f"seed_{args.seed}"
    analyze_seed(seed_dir, config, args.seed)


if __name__ == "__main__":
    main()
