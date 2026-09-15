"""Internal single-GPU shard for the cross-subnet interaction probe."""

from __future__ import annotations

import argparse

from rq2_cross_subnet_interaction import run_cross_subnet_interaction_probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--geometry-marginals", required=True)
    parser.add_argument("--resource-marginals", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--batch-offset", required=True, type=int)
    parser.add_argument("--num-batches", required=True, type=int)
    parser.add_argument("--one-step", action="store_true")
    args = parser.parse_args()
    run_cross_subnet_interaction_probe(
        checkpoint=args.checkpoint,
        config_path=args.config,
        geometry_marginals_path=args.geometry_marginals,
        resource_marginals_path=args.resource_marginals,
        output_dir=args.output,
        dataset_root=args.dataset_root,
        device="cuda:0",
        num_batches=args.num_batches,
        batch_offset=args.batch_offset,
        run_one_step=args.one_step,
    )


if __name__ == "__main__":
    main()
