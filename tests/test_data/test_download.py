"""Unit tests for dataset download and verification utilities."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from cyfm.data.download import (
    ensure_dataset_exists,
    ensure_fastmri,
    ensure_skm_tea_mini,
    load_env,
)


class TestDownloadUtilities:
    def test_load_env_nonexistent(self, tmp_path) -> None:
        missing = tmp_path / ".env"
        assert load_env(missing) == {}

    def test_load_env_parses_properly(self, tmp_path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# Comment line\n"
            "FASTMRI_MINI_URL='https://example.com/mini.tar.gz'\n"
            'FASTMRI_FULL_URLS="https://example.com/1.tar.gz, https://example.com/2.tar.gz"\n'
            "\n"
            "EMPTY_LINE=true\n",
            encoding="utf-8",
        )
        parsed = load_env(env_file)
        assert parsed["FASTMRI_MINI_URL"] == "https://example.com/mini.tar.gz"
        assert (
            parsed["FASTMRI_FULL_URLS"]
            == "https://example.com/1.tar.gz, https://example.com/2.tar.gz"
        )
        assert parsed["EMPTY_LINE"] == "true"

    def test_ensure_dataset_exists_unsupported(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="Unsupported dataset"):
            ensure_dataset_exists("invalid_cohort", tmp_path)

    def test_ensure_skm_tea_existing_data_skips(self, tmp_path) -> None:
        (tmp_path / "slice_001.h5").touch()
        with patch("cyfm.data.download.snapshot_download") as mock_hf:
            ensure_dataset_exists("skm_tea", tmp_path)
            mock_hf.assert_not_called()

    def test_ensure_skm_tea_calls_snapshot_download(self, tmp_path) -> None:
        with patch("cyfm.data.download.snapshot_download") as mock_hf:
            ensure_dataset_exists("skm_tea", tmp_path)
            mock_hf.assert_called_once_with(
                repo_id="arjundd/skm-tea-mini",
                repo_type="dataset",
                local_dir=str(tmp_path),
            )

    def test_ensure_fastmri_existing_data_skips(self, tmp_path) -> None:
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "vol_001.h5").touch()
        with patch("cyfm.data.download.download_and_extract_tar") as mock_dl:
            ensure_dataset_exists("fastmri", tmp_path, mode="local")
            mock_dl.assert_not_called()

    def test_ensure_fastmri_local_missing_env_url(self, tmp_path) -> None:
        with patch("cyfm.data.download.load_env", return_value={}):
            with pytest.raises(ValueError, match="Missing FASTMRI_MINI_URL"):
                ensure_dataset_exists("fastmri", tmp_path, mode="local")

    def test_ensure_fastmri_full_missing_env_url(self, tmp_path) -> None:
        with patch("cyfm.data.download.load_env", return_value={}):
            with pytest.raises(ValueError, match="Missing FASTMRI_FULL_URLS in .env"):
                ensure_dataset_exists("fastmri", tmp_path, mode="full")

    def test_ensure_fastmri_invalid_mode(self, tmp_path) -> None:
        with patch("cyfm.data.download.load_env", return_value={}):
            with pytest.raises(ValueError, match="Unknown fastMRI mode"):
                ensure_dataset_exists("fastmri", tmp_path, mode="invalid_mode")

    def test_ensure_fastmri_local_downloads_tar(self, tmp_path) -> None:
        mock_env = {"FASTMRI_MINI_URL": "https://example.com/mini.tar.gz"}
        with patch("cyfm.data.download.load_env", return_value=mock_env):
            with patch("cyfm.data.download.download_and_extract_tar") as mock_dl:
                ensure_dataset_exists("fastmri", tmp_path, mode="local")
                mock_dl.assert_called_once_with("https://example.com/mini.tar.gz", tmp_path)

    def test_ensure_fastmri_full_downloads_all_tars(self, tmp_path) -> None:
        mock_env = {"FASTMRI_FULL_URLS": "https://ex.com/1.tar.gz, https://ex.com/2.tar.gz"}
        with patch("cyfm.data.download.load_env", return_value=mock_env):
            with patch("cyfm.data.download.download_and_extract_tar") as mock_dl:
                ensure_dataset_exists("fastmri", tmp_path, mode="full")
                assert mock_dl.call_count == 2

    def test_backward_compatible_wrappers(self, tmp_path) -> None:
        (tmp_path / "dummy.h5").touch()
        with patch("cyfm.data.download.ensure_dataset_exists") as mock_ensure:
            ensure_skm_tea_mini(tmp_path)
            mock_ensure.assert_called_once_with(dataset_name="skm_tea", data_dir=tmp_path)

        with patch("cyfm.data.download.ensure_dataset_exists") as mock_ensure:
            ensure_fastmri(tmp_path, mode="full")
            mock_ensure.assert_called_once_with(
                dataset_name="fastmri", data_dir=tmp_path, mode="full"
            )
