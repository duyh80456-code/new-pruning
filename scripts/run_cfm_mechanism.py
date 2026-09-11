from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from baseline_artifacts import find_confirmatory_root
from cfm_mechanism import run_cfm_mechanism


def main():
    parser = argparse.ArgumentParser(description="Run cached-feature CFM mechanism tests")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with Path(args.config).open() as handle:
        config = yaml.safe_load(handle)
    baseline_root = find_confirmatory_root(config["baseline_artifacts"]["input_root"])
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    results = run_cfm_mechanism(baseline_root, config, output_dir)
    print(f"Completed {len(results)} CFM/baseline evaluations in {output_dir}")


if __name__ == "__main__":
    main()
