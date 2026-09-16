"""Internal single-GPU worker for one HT trajectory."""

from __future__ import annotations

import argparse

from rq2_quick_trajectory_diagnostic import probe_trajectory_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ht-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--path", required=True, choices=("geo_ht", "resource_ht"))
    parser.add_argument("--dataset-root", required=True)
    args = parser.parse_args()
    probe_trajectory_path(
        args.ht_root, args.output, args.path, args.dataset_root, device="cuda:0"
    )


if __name__ == "__main__":
    main()
