"""Dataset download and verification utilities for SKM-TEA and fastMRI."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def load_env(env_path: str | Path = ".env") -> dict[str, str]:
    """Parse key-value pairs from a .env file without external dependencies.

    Args:
        env_path: Path to the .env file. Defaults to '.env'.

    Returns:
        Dictionary containing the parsed environment variables.
    """
    path = Path(env_path)
    if not path.exists():
        return {}

    env: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key, val = stripped.split("=", 1)
                env[key.strip()] = val.strip(' \'"')
    return env


def download_and_extract_tar(url: str, dest_dir: str | Path) -> None:
    """Download an archive using wget and extract it via tar, removing the archive.

    Args:
        url: Remote URL of the tar or tar.gz archive.
        dest_dir: Target directory where the archive contents will be extracted.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    filename = url.split("?")[0].split("/")[-1]
    filepath = dest / filename

    logger.info("Downloading: %s", filename)
    print(f"[Download] Fetching: {filename}...")
    subprocess.run(["wget", "-c", url, "-O", str(filepath)], check=True)

    logger.info("Extracting archive: %s", filename)
    print(f"[Download] Extracting archive: {filename}...")
    subprocess.run(["tar", "-xf", str(filepath), "-C", str(dest)], check=True)

    logger.info("Removing temporary archive: %s", filename)
    print(f"[Download] Cleaning up archive: {filename}...")
    filepath.unlink(missing_ok=True)


def ensure_dataset_exists(
    dataset_name: str,
    data_dir: str | Path,
    mode: str | None = None,
) -> None:
    """Ensure that the required dataset files exist locally, downloading if absent.

    Args:
        dataset_name: Dataset identifier ('skm_tea', 'skm-tea-mini', or 'fastmri').
        data_dir: Path to directory where dataset files should reside.
        mode: Optional sub-mode or split configuration ('local' or 'full' for fastMRI).

    Raises:
        ValueError: If required configuration/URLs are missing, or if an unsupported
            dataset name is provided.
    """
    path = Path(data_dir)
    canonical_name = dataset_name.lower().replace("-", "_")

    if canonical_name in ("skm_tea", "skm_tea_mini"):
        if path.exists() and len(list(path.glob("*.h5"))) > 0:
            return

        if snapshot_download is None:
            raise ImportError(
                "huggingface_hub is required to download SKM-TEA data. "
                "Please install it via `pip install huggingface_hub`."
            )

        logger.info("No SKM-TEA data found in %s. Downloading from Hugging Face Hub...", path)
        print(f"[Auto-Download] Missing SKM-TEA data in {path}. Downloading from Hugging Face Hub...")
        snapshot_download(
            repo_id="arjundd/skm-tea-mini",
            repo_type="dataset",
            local_dir=str(path),
        )
        logger.info("SKM-TEA download complete.")
        print(f"[Auto-Download] SKM-TEA dataset successfully downloaded to {path}.")

    elif canonical_name == "fastmri":
        if path.exists() and len(list(path.glob("**/*.h5"))) > 0:
            return

        logger.info("No fastMRI data found in %s. Checking environment configuration...", path)
        print(
            f"[Auto-Download] Missing fastMRI data in {path}. Checking environment configuration..."
        )
        env = load_env()
        mode_resolved = mode or "local"

        if mode_resolved == "local":
            url = env.get("FASTMRI_MINI_URL")
            if not url:
                raise ValueError(
                    "Missing FASTMRI_MINI_URL in .env! "
                    "Please copy .env.example to .env and configure FASTMRI_MINI_URL."
                )
            logger.info("Mode 'local': Downloading single archive from FASTMRI_MINI_URL...")
            print("[Auto-Download] Mode 'local': Downloading single archive from FASTMRI_MINI_URL...")
            download_and_extract_tar(url, path)

        elif mode_resolved == "full":
            urls_str = env.get("FASTMRI_FULL_URLS")
            if not urls_str:
                raise ValueError(
                    "Missing FASTMRI_FULL_URLS in .env! "
                    "Please copy .env.example to .env and configure FASTMRI_FULL_URLS."
                )
            urls = [u.strip() for u in urls_str.split(",") if u.strip()]
            logger.info("Mode 'full': Downloading %d archives from FASTMRI_FULL_URLS...", len(urls))
            print(
                f"[Auto-Download] Mode 'full': Found {len(urls)} archives to download sequentially..."
            )
            for i, url in enumerate(urls, 1):
                logger.info("Processing archive %d/%d...", i, len(urls))
                print(f"\n[Auto-Download] Processing archive {i}/{len(urls)}...")
                download_and_extract_tar(url, path)

        else:
            raise ValueError(
                f"Unknown fastMRI mode: {mode!r}. Expected 'local' or 'full'."
            )

        logger.info("fastMRI packages successfully integrated into %s.", path)
        print(f"[Auto-Download] All fastMRI packages successfully integrated into {path}.")

    else:
        raise ValueError(
            f"Unsupported dataset for auto-download: {dataset_name!r}. "
            "Supported datasets are 'skm_tea' and 'fastmri'."
        )


def ensure_skm_tea_mini(data_dir: str | Path) -> None:
    """Ensure SKM-TEA-mini dataset is downloaded and available.

    Args:
        data_dir: Directory where dataset files should reside.
    """
    ensure_dataset_exists(dataset_name="skm_tea", data_dir=data_dir)


def ensure_fastmri(data_dir: str | Path, mode: str = "local") -> None:
    """Ensure fastMRI dataset packages are downloaded and available.

    Args:
        data_dir: Directory where dataset files should reside.
        mode: Dataset mode, either 'local' or 'full'. Defaults to 'local'.
    """
    ensure_dataset_exists(dataset_name="fastmri", data_dir=data_dir, mode=mode)
