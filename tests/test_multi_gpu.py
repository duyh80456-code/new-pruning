from pathlib import Path

import yaml

import scripts.run_multi_gpu as multi_gpu


def test_multi_gpu_scheduler_runs_every_seed_and_merges_outputs(tmp_path, monkeypatch):
    config = yaml.safe_load(Path("configs/kaggle_three_seed_oracle.yaml").read_text())
    root_output = tmp_path / "combined"
    config["experiment"]["output_dir"] = str(root_output)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    gpu_assignments = []

    def fake_subprocess_run(command, cwd, env, check):
        worker_config = yaml.safe_load(Path(command[-1]).read_text())
        seed = int(worker_config["experiment"]["seeds"][0])
        worker_root = Path(worker_config["experiment"]["output_dir"])
        (worker_root / f"seed_{seed}").mkdir(parents=True)
        (worker_root / f"seed_{seed}" / "marker.txt").write_text(str(seed))
        gpu_assignments.append(env["CUDA_VISIBLE_DEVICES"])

    def fake_finalize(root, final_config):
        assert final_config["experiment"]["seeds"] == [0, 1, 2]
        for seed in [0, 1, 2]:
            assert (root / f"seed_{seed}" / "marker.txt").read_text() == str(seed)
        return root / "reports" / "smoke_test_report.md"

    monkeypatch.setattr(multi_gpu.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(multi_gpu, "_finalize", fake_finalize)

    report = multi_gpu.run_multi_gpu(config_path, gpu_ids=[0, 1])

    assert report == root_output / "reports" / "smoke_test_report.md"
    assert len(gpu_assignments) == 3
    assert set(gpu_assignments) == {"0", "1"}

