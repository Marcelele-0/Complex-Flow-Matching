"""Dataset download and verification utilities for fastMRI."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import dotenv

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def load_env(env_path: str | Path = ".env") -> dict[str, str | None]:
    """Parse key-value pairs from a .env file using python-dotenv.
    Supports multi-line variables if enclosed in quotes.
    """
    path = Path(env_path)
    if not path.exists():
        return {}
    return dotenv.dotenv_values(str(path))


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
        dataset_name: Dataset identifier; only 'fastmri' is supported.
        data_dir: Path to directory where dataset files should reside.
        mode: Optional sub-mode or split configuration ('local' or 'full' for fastMRI).

    Raises:
        ValueError: If required configuration/URLs are missing, or if an unsupported
            dataset name is provided.
    """
    path = Path(data_dir)
    canonical_name = dataset_name.lower().replace("-", "_")

    if canonical_name == "fastmri":
        if path.exists() and len(list(path.glob("**/*.h5"))) > 0:
            return

        logger.info("No fastMRI data found in %s. Checking environment configuration...", path)
        print(
            f"[Auto-Download] Missing fastMRI data in {path}. Checking environment configuration..."
        )
        env = load_env()
        mode_resolved = mode or "local"

        if mode_resolved == "local":
            urls_str = env.get("FASTMRI_MINI_URLS") or env.get("FASTMRI_MINI_URL")
            if not urls_str:
                raise ValueError(
                    "Missing FASTMRI_MINI_URLS in .env! "
                    "Please copy .env.example to .env and configure FASTMRI_MINI_URLS."
                )
            urls = [u.strip() for u in urls_str.replace("\n", ",").split(",") if u.strip()]
            logger.info(
                "Mode 'local': Downloading %d archive(s) from FASTMRI_MINI_URLS...",
                len(urls),
            )
            print(
                f"[Auto-Download] Mode 'local': Found {len(urls)} archive(s) "
                "to download sequentially..."
            )
            for i, url in enumerate(urls, 1):
                logger.info("Processing mini archive %d/%d...", i, len(urls))
                print(f"\n[Auto-Download] Processing mini archive {i}/{len(urls)}...")
                download_and_extract_tar(url, path)

        elif mode_resolved == "full":
            urls_str = env.get("FASTMRI_FULL_URLS")
            if not urls_str:
                raise ValueError(
                    "Missing FASTMRI_FULL_URLS in .env! "
                    "Please copy .env.example to .env and configure FASTMRI_FULL_URLS."
                )
            urls = [u.strip() for u in urls_str.replace("\n", ",").split(",") if u.strip()]
            logger.info("Mode 'full': Downloading %d archives from FASTMRI_FULL_URLS...", len(urls))
            print(
                f"[Auto-Download] Mode 'full': Found {len(urls)} archives "
                "to download sequentially..."
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
            "The only supported dataset is 'fastmri'."
        )


def ensure_fastmri(data_dir: str | Path, mode: str = "local") -> None:
    """Ensure fastMRI dataset packages are downloaded and available.

    Args:
        data_dir: Directory where dataset files should reside.
        mode: Dataset mode, either 'local' or 'full'. Defaults to 'local'.
    """
    ensure_dataset_exists(dataset_name="fastmri", data_dir=data_dir, mode=mode)
