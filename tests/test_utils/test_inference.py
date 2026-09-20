"""Tests for the shared model-building and checkpoint-loading helpers.

These run without Hydra and without a real checkpoint: the helpers were extracted
out of the entry points precisely so this logic could be exercised directly.
"""

import os
import time

import pytest
import torch
from omegaconf import OmegaConf

from cyfm.utils.inference import (
    build_model,
    find_latest_checkpoint,
    load_weights,
    reject_unsupported_sampling_model,
    resolve_checkpoint,
)

CPU = torch.device("cpu")


class TestBuildModel:
    def test_builds_plain_unet_with_correct_velocity_shape(self) -> None:
        cfg = OmegaConf.create({"model": {"name": "c_unet", "base_channels": 8}})
        model = build_model(cfg, CPU)

        x = torch.rand(2, 3, 16, 16)
        t = torch.rand(2)
        out = model(x, t)

        # Velocity contract: 2 channels (v_amp, v_phi), spatial dims preserved.
        assert out.shape == (2, 2, 16, 16)

    def test_builds_attention_unet(self) -> None:
        cfg = OmegaConf.create(
            {
                "model": {
                    "name": "c_unet_attention",
                    "base_channels": 8,
                    "channel_mults": [1, 2],
                    "use_attention": False,
                }
            }
        )
        model = build_model(cfg, CPU)

        # channel_mults of length 2 means one pooling level, so 16x16 is safe.
        out = model(torch.rand(1, 3, 16, 16), torch.rand(1))
        assert out.shape == (1, 2, 16, 16)

    def test_defaults_to_c_unet_when_unspecified(self) -> None:
        model = build_model(OmegaConf.create({}), CPU)
        assert type(model).__name__ == "CylindricalUNet"

    def test_unknown_model_name_raises(self) -> None:
        cfg = OmegaConf.create({"model": {"name": "not_a_real_model"}})
        with pytest.raises(ValueError, match="Unknown config model_name"):
            build_model(cfg, CPU)

    def test_builds_cross_slice_unet_returning_center_velocity(self) -> None:
        cfg = OmegaConf.create(
            {
                "model": {
                    "name": "c_unet_cross_slice",
                    "base_channels": 16,
                    "channel_mults": [1, 2],
                    "attn_heads": 2,
                }
            }
        )
        model = build_model(cfg, CPU)

        # Window of 3 slices in, one center velocity out.
        out = model(torch.rand(2, 3, 3, 16, 16), torch.rand(2))
        assert out.shape == (2, 2, 16, 16)

    def test_cross_slice_model_takes_the_manifold_state_width(self) -> None:
        """in_channels must reach the 2.5D model, so the Euclidean baseline can
        be run with cross-slice attention rather than rejected."""
        cfg = OmegaConf.create(
            {
                "model": {
                    "name": "c_unet_cross_slice",
                    "base_channels": 8,
                    "channel_mults": [1, 2],
                    "attn_heads": 2,
                }
            }
        )
        model = build_model(cfg, CPU, in_channels=2, out_channels=2)

        out = model(torch.rand(2, 3, 2, 16, 16), torch.rand(2))
        assert out.shape == (2, 2, 16, 16)


class TestRejectUnsupportedSamplingModel:
    def test_cross_slice_model_is_rejected_with_an_explanation(self) -> None:
        cfg = OmegaConf.create({"model": {"name": "c_unet_cross_slice"}})
        # Matched on the reason, not on a class name: the blocker is the solver's
        # shape contract, which holds for every geometry's solver.
        with pytest.raises(NotImplementedError, match="multi-slice state"):
            reject_unsupported_sampling_model(cfg)

    @pytest.mark.parametrize("name", ["c_unet", "c_unet_attention"])
    def test_2d_models_pass(self, name: str) -> None:
        reject_unsupported_sampling_model(OmegaConf.create({"model": {"name": name}}))

    def test_missing_model_config_defaults_to_a_supported_model(self) -> None:
        reject_unsupported_sampling_model(OmegaConf.create({}))


class TestFindLatestCheckpoint:
    def test_returns_none_for_empty_dir(self, tmp_path) -> None:
        assert find_latest_checkpoint(str(tmp_path)) is None

    def test_returns_newest_by_mtime_across_subdirs(self, tmp_path) -> None:
        old = tmp_path / "run_a" / "checkpoints"
        new = tmp_path / "run_b" / "checkpoints"
        old.mkdir(parents=True)
        new.mkdir(parents=True)

        old_ckpt = old / "checkpoint_epoch_10.pt"
        new_ckpt = new / "checkpoint_epoch_20.pt"
        old_ckpt.write_bytes(b"old")
        time.sleep(0.01)
        new_ckpt.write_bytes(b"new")

        assert find_latest_checkpoint(str(tmp_path)) == str(new_ckpt)

    def test_ignores_non_pt_files(self, tmp_path) -> None:
        (tmp_path / "notes.txt").write_text("not a checkpoint")
        assert find_latest_checkpoint(str(tmp_path)) is None


class TestResolveCheckpoint:
    def _make_checkpoint(self, tmp_path, run_name: str) -> str:
        ckpt_dir = tmp_path / "outputs" / "train" / run_name / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        ckpt = ckpt_dir / "checkpoint_epoch_1.pt"
        ckpt.write_bytes(b"weights")
        return str(ckpt)

    def test_uses_section_run_name(self, tmp_path) -> None:
        expected = self._make_checkpoint(tmp_path, "my_run")
        cfg = OmegaConf.create({"evaluate": {"run_name": "my_run"}})

        assert resolve_checkpoint(cfg, "evaluate", str(tmp_path)) == expected

    def test_falls_back_to_experiment_name(self, tmp_path) -> None:
        expected = self._make_checkpoint(tmp_path, "fallback_run")
        cfg = OmegaConf.create(
            {"evaluate": {"run_name": None}, "logging": {"experiment_name": "fallback_run"}}
        )

        assert resolve_checkpoint(cfg, "evaluate", str(tmp_path)) == expected

    def test_section_name_appears_in_the_error(self, tmp_path) -> None:
        cfg = OmegaConf.create({})
        with pytest.raises(ValueError, match="No run_name in generate config"):
            resolve_checkpoint(cfg, "generate", str(tmp_path))

    def test_missing_checkpoint_names_the_searched_dir(self, tmp_path) -> None:
        cfg = OmegaConf.create({"evaluate": {"run_name": "ghost_run"}})
        with pytest.raises(FileNotFoundError, match="ghost_run"):
            resolve_checkpoint(cfg, "evaluate", str(tmp_path))


class TestLoadWeights:
    def test_round_trips_and_sets_eval_mode(self, tmp_path) -> None:
        cfg = OmegaConf.create({"model": {"name": "c_unet", "base_channels": 8}})
        source = build_model(cfg, CPU)

        path = os.path.join(str(tmp_path), "ckpt.pt")
        torch.save(source.state_dict(), path)

        target = build_model(cfg, CPU)
        target.train()  # so eval() is observably applied
        load_weights(target, path, CPU)

        assert not target.training
        x, t = torch.rand(1, 3, 16, 16), torch.rand(1)
        with torch.no_grad():
            assert torch.allclose(source.eval()(x, t), target(x, t))

    def test_strips_orig_mod_prefix_from_compiled_checkpoints(self, tmp_path) -> None:
        """torch.compile prefixes every key; such a checkpoint must still load."""
        cfg = OmegaConf.create({"model": {"name": "c_unet", "base_channels": 8}})
        source = build_model(cfg, CPU)

        compiled_state = {f"_orig_mod.{k}": v for k, v in source.state_dict().items()}
        path = os.path.join(str(tmp_path), "compiled.pt")
        torch.save(compiled_state, path)

        target = build_model(cfg, CPU)
        load_weights(target, path, CPU)  # would raise on unexpected keys

        x, t = torch.rand(1, 3, 16, 16), torch.rand(1)
        with torch.no_grad():
            assert torch.allclose(source.eval()(x, t), target(x, t))
