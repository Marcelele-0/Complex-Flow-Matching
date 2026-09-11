"""The paper experiments in ``conf/experiment/`` and the runner that executes them.

Checks that every paper experiment has a well-formed ``paper:`` block pointing at
configs and scripts that exist, that each U-Net table config composes to its
training protocol, that the U-Net experiments enumerate exactly the runs archived in
``docs/reproduce/paper_results/``, and that ``--seeds`` / ``--sides`` / ``--tag``
narrow an experiment as documented.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
from types import ModuleType

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONF = ROOT / "conf"
RESULTS = ROOT / "docs" / "reproduce" / "paper_results"
RUNNER = ROOT / "scripts" / "paper" / "reproduce.py"

# archive file -> (run-name prefix of its runs, experiment config that produced them)
ARCHIVES = {
    "unet_eval_metrics_l2u.json": ("l2u_", "paper_unet"),
    "unet_eval_metrics_l1u_cylindrical.json": ("l1u_", "paper_unet_l1"),
    "unet_eval_metrics_l1w_cartesian.json": ("l1w_", "paper_unet_v1loss"),
}


def load_runner() -> ModuleType:
    """Import scripts/paper/reproduce.py, which is not part of a package."""
    spec = importlib.util.spec_from_file_location("reproduce", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["reproduce"] = module
    spec.loader.exec_module(module)
    return module


reproduce = load_runner()
EXPERIMENTS = list(reproduce.PAPER_ORDER)
UNET = [name for name in EXPERIMENTS if reproduce.load_spec(name)["kind"] == "unet_grid"]


def test_paper_order_lists_every_paper_experiment() -> None:
    """A config with a paper: block is a paper experiment, and all are in the order."""
    with_block = {
        path.stem
        for path in (CONF / "experiment").glob("*.yaml")
        if "paper" in (OmegaConf.load(path) or {})
    }
    assert with_block == set(EXPERIMENTS)


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_paper_block_is_well_formed(name: str) -> None:
    spec = reproduce.load_spec(name)
    for key in ("name", "section", "runtime", "kind"):
        assert key in spec, key
    scripts = [entry[0] for entry in spec.get("commands", []) + spec.get("report", [])]
    for script in scripts:
        assert (ROOT / script).is_file(), script
    if spec["kind"] == "unet_grid":
        assert spec["sides"] and spec["seeds"] and spec["arms"]
        for arm in spec["arms"]:
            assert (CONF / "experiment" / f"{arm['experiment']}.yaml").is_file()
            assert arm["geometries"] and arm["couplings"] and arm["prefix"]


@pytest.mark.parametrize("name", UNET)
def test_table_config_composes_to_its_protocol(name: str) -> None:
    """``+experiment=<table>`` trains exactly like ``+experiment=paper_unet``."""
    cell = ["manifold=cylindrical", "training.coupling=ot", "dataset.crop_size=[32,32]"]
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        table = compose(config_name="config", overrides=[f"+experiment={name}", *cell])
        protocol = compose(config_name="config", overrides=["+experiment=paper_unet", *cell])
    composed = OmegaConf.to_container(table, resolve=False)
    assert isinstance(composed, dict)
    assert composed.pop("paper")["kind"] == "unet_grid"
    assert composed == OmegaConf.to_container(protocol, resolve=False)


def test_unet_experiments_cover_exactly_the_archived_runs() -> None:
    """The union of the table experiments' runs is the union of the three v2 archives."""
    specified: set[tuple[str, str]] = set()
    for name in UNET:
        if name == "ablation_loss32":
            continue
        grid = reproduce.grid(reproduce.load_spec(name))
        specified |= {(f"{run.name}_eval", run.experiment) for run in grid}
    archived: set[tuple[str, str]] = set()
    for filename, (prefix, experiment) in ARCHIVES.items():
        names = json.loads((RESULTS / filename).read_text())
        archived |= {(prefix + name, experiment) for name in names}
    assert specified == archived


def test_table3_contains_table2() -> None:
    table2 = {run.name for run in reproduce.grid(reproduce.load_spec("table2_unet64"))}
    table3 = {run.name for run in reproduce.grid(reproduce.load_spec("table3_unet_scaling"))}
    assert table2 < table3


def test_overrides_narrow_the_grid() -> None:
    spec = reproduce.load_spec("table3_unet_scaling")
    runs = reproduce.grid(spec, tag="quick_", seeds=[0], sides=[16])
    assert len(runs) == 4
    assert {run.seed for run in runs} == {0}
    assert all(run.name.startswith("quick_l2u_unsz_") and run.side == 16 for run in runs)


def test_ablation_names_match_the_ablation_report() -> None:
    runs = reproduce.grid(reproduce.load_spec("ablation_loss32"), seeds=[0])
    assert {run.name for run in runs} == {
        "abl_cylindrical_l1_weighted_32_s0",
        "abl_cylindrical_l1_unweighted_32_s0",
        "abl_cylindrical_l2_weighted_32_s0",
        "abl_cylindrical_l2_unweighted_32_s0",
        "abl_euclidean_l1_32_s0",
        "abl_euclidean_l2_32_s0",
    }


def test_commands_carry_the_run_arguments() -> None:
    run = reproduce.Run("x", "paper_unet_l1", 32, "cylindrical", "ot", 3)
    train = reproduce.train_command(run, epochs=None)
    assert train[1:3] == ["-m", "cfm.train"]
    assert "+experiment=paper_unet_l1" in train and "training.seed=3" in train
    assert not any(arg.startswith("training.epochs") for arg in train)
    assert "training.epochs=1" in reproduce.train_command(run, epochs=1)
    evaluate = reproduce.evaluate_command(run)
    assert "evaluate.run_name=x" in evaluate and "evaluate.seed=3" in evaluate


def test_placeholders_expand() -> None:
    command = reproduce.script_command(
        ["scripts/paper/loss_protocols.py", "--tag", "{tag}", "--seeds", "{seeds}"],
        tag="quick_",
        seeds=[0, 1],
        sides=[16],
    )
    assert command[2:] == ["--tag", "quick_", "--seeds", "0", "1"]


def test_protocol_configs_are_not_paper_experiments() -> None:
    with pytest.raises(ValueError, match="no paper: block"):
        reproduce.load_spec("paper_unet")


def test_dry_run_lists_every_experiment() -> None:
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--all", "--dry-run", "--seeds", "0", "--tag", "dry_"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.count("\n=== ") == len(EXPERIMENTS)
