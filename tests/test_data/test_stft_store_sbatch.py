"""The audio intake job, exercised without a cluster and without downloading anything.

The download is stubbed, but the archive, the extraction and the build are real: what
this has to catch is the corpus layout, and a mock of `tar` would assert the layout the
test already assumes instead of the one LibriSpeech actually ships.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tarfile

import pytest
import soundfile
import torch

from cfm.data.stft import DEFAULT_STFT

SBATCH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "wcss" / "build_stft_store.sbatch"
)
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _a_cohort_archive(path: pathlib.Path, cohort: str, speakers: list[str]) -> None:
    """A tar.gz shaped exactly like LibriSpeech: LibriSpeech/<cohort>/<speaker>/<chapter>/."""
    staging = path.parent / f"staging-{cohort}"
    for speaker in speakers:
        chapter = staging / "LibriSpeech" / cohort / speaker / "1"
        chapter.mkdir(parents=True)
        for utterance in range(2):
            generator = torch.Generator().manual_seed(int(speaker) + utterance)
            wave = (torch.randn(16_000, generator=generator) * 0.1).numpy()
            soundfile.write(
                str(chapter / f"{speaker}-1-{utterance:04d}.flac"),
                wave,
                DEFAULT_STFT.sample_rate,
                format="FLAC",
            )
    with tarfile.open(path, "w:gz") as archive:
        archive.add(staging / "LibriSpeech", arcname="LibriSpeech")
    shutil.rmtree(staging)


def _a_fake_cluster(tmp_path: pathlib.Path, cohorts: dict[str, list[str]]) -> pathlib.Path:
    """A stub env.sh plus wget/curl that serve prebuilt archives instead of the network."""
    root = pathlib.Path(__file__).resolve().parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    served = tmp_path / "served"
    served.mkdir()
    for cohort, speakers in cohorts.items():
        _a_cohort_archive(served / f"{cohort}.tar.gz", cohort, speakers)

    (bin_dir / "curl").write_text("#!/usr/bin/env bash\necho 206\n")
    # wget's last argument is -O <target>; copy the prebuilt archive there if we have it.
    (bin_dir / "wget").write_text(
        "#!/usr/bin/env bash\n"
        'target="${!#}"\n'
        'for a in "$@"; do case "$a" in *.tar.gz) url="$a";; esac; done\n'
        f'src="{served}/$(basename "$url")"\n'
        '[ -f "$src" ] || exit 1\n'
        'cp "$src" "$target"\n'
    )
    for stub in ("curl", "wget"):
        (bin_dir / stub).chmod(0o755)

    (tmp_path / "env.sh").write_text(
        f'export CYFM_ROOT="{root}"\n'
        f'export PATH="{bin_dir}:$PATH"\n'
        # the job calls `uv run --no-sync python <script> ...`, so drop three words
        f'uv() {{ shift 3; "{sys.executable}" "$@"; }}\n'
        "export -f uv\n"
    )
    return tmp_path / "env.sh"


def _run(tmp_path: pathlib.Path, cohorts: dict[str, list[str]], **env: str):
    env_sh = _a_fake_cluster(tmp_path, cohorts)
    out = tmp_path / "store" / "librispeech.h5"
    return (
        subprocess.run(
            ["bash", str(SBATCH)],
            capture_output=True,
            text=True,
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": str(tmp_path),
                "TMPDIR": str(tmp_path / "work"),
                "SLURM_JOB_ID": "1",
                "CYFM_ENV": str(env_sh),
                "COHORTS": " ".join(cohorts),
                "OUT": str(out),
                **env,
            },
        ),
        out,
    )


def test_both_cohorts_are_pooled_into_one_flat_speaker_root(tmp_path: pathlib.Path) -> None:
    """The defect this job exists to avoid.

    LibriSpeech nests as LibriSpeech/<cohort>/<speaker>/, and the builder reads the
    speaker from the first component below --raw-dir. Handing it the archive root would
    yield one "speaker" per cohort, and the split by speaker would silently become a
    split by cohort. Nothing downstream would complain.
    """
    (tmp_path / "work").mkdir()
    done, out = _run(
        tmp_path, {"dev-clean": ["1272", "1462", "1673"], "test-clean": ["1089", "1221"]}
    )
    assert done.returncode == 0, done.stderr
    assert "speakers=5" in done.stdout, done.stdout

    import json

    manifest = json.loads(out.with_suffix(".json").read_text())
    assert sorted(manifest["speakers"]) == ["1089", "1221", "1272", "1462", "1673"]


def test_one_cohort_alone_still_works(tmp_path: pathlib.Path) -> None:
    (tmp_path / "work").mkdir()
    done, out = _run(tmp_path, {"dev-clean": ["1272", "1462"]})
    assert done.returncode == 0, done.stderr
    assert "speakers=2" in done.stdout

    import json

    assert sorted(json.loads(out.with_suffix(".json").read_text())["speakers"]) == ["1272", "1462"]


def test_a_dead_link_stops_before_downloading(tmp_path: pathlib.Path) -> None:
    """A 403 on an archive must fail loudly rather than build an empty store."""
    (tmp_path / "work").mkdir()
    env_sh = _a_fake_cluster(tmp_path, {"dev-clean": ["1272"]})
    (tmp_path / "bin" / "curl").write_text("#!/usr/bin/env bash\necho 403\n")
    (tmp_path / "bin" / "curl").chmod(0o755)
    done = subprocess.run(
        ["bash", str(SBATCH)],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "TMPDIR": str(tmp_path / "work"),
            "SLURM_JOB_ID": "1",
            "CYFM_ENV": str(env_sh),
            "COHORTS": "dev-clean",
            "OUT": str(tmp_path / "store" / "librispeech.h5"),
        },
    )
    assert done.returncode == 1
    assert "answered 403" in done.stdout
    assert not (tmp_path / "store" / "librispeech.h5").exists()


def test_the_raw_audio_is_gone_when_the_job_finishes(tmp_path: pathlib.Path) -> None:
    """--delete-consumed is what keeps the corpus off the shared filesystem."""
    (tmp_path / "work").mkdir()
    done, _ = _run(tmp_path, {"dev-clean": ["1272", "1462"]})
    assert done.returncode == 0, done.stderr
    assert list((tmp_path / "work").rglob("*.flac")) == []
