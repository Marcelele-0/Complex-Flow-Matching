# Dokumentacja setupu projektowego

Cel: jednolity, powtarzalny setup projektowy wykorzystujący Hydra, Weights & Biases (wandb), `pyproject.toml`/zarządzanie zależnościami, `pre-commit` oraz minimalne praktyki CI/testów. Materiał jest przygotowany tak, aby łatwo przenieść go do innego projektu (np. zadanie klasyfikacji MNIST).

**Zakres**: konfiguracja środowiska, struktura katalogów, przykłady plików konfiguracyjnych dla Hydry, integracja z wandb, pre-commit, testy i uruchamianie eksperymentów.

**Wymagania wstępne**
- Python >= 3.10 (dopasuj do projektu)
- virtualenv / venv / conda (wybierz jedną metodę izolacji)
- narzędzie do zarządzania zależnościami: `pip` + `requirements.txt` lub `pip-tools`, albo `poetry` (preferowane projekty nowoczesne)

1. Tworzenie środowiska

Przykład z `venv` i `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Jeśli używasz `poetry`:

```bash
poetry install
poetry shell
```

2. Struktura projektu (zalecana)

- `src/` — kod źródłowy pakietu (np. `src/myproj/`)
- `conf/` lub `configs/` — konfiguracje Hydra (np. `conf/config.yaml`, `conf/training/*.yaml`)
- `scripts/` — helpery uruchomieniowe
- `data/` — surowe dane (nie trzymać dużych danych w repo)
- `outputs/` — wyniki eksperymentów (Hydra może nadpisywać `outputs` lub kierować logi)
- `tests/` — testy jednostkowe
- `docs/` — materiały i instrukcje (tu właśnie)

3. Konfiguracja Hydra

- Zainstaluj: `pip install hydra-core` (dopasuj wersję)
- Umieść konfigi w `conf/` (format YAML). Użyj hierarchii np.:

```
conf/
  config.yaml        # główny config (łączy sekcje)
  dataset/
    mnist.yaml
  model/
    simple_cnn.yaml
  training/
    default.yaml
```

- W `src` udekoruj punkt wejścia:

```py
from hydra import main
from omegaconf import DictConfig

@main(config_path="../conf", config_name="config")
def run(cfg: DictConfig):
    # inicjalizacja: seed, dataset, model, trainer
    ...

if __name__ == "__main__":
    run()
```

- Korzystanie z override'ów w linii poleceń:

```bash
python -m src.train dataset=mnist training.lr=0.001
```

- Hydra conveniences:
  - `hydra.run.dir` kontroluje katalog outputów
  - `hydra.sweep.dir` dla sweepów
  - zapis użytego configu do katalogu eksperymentu — zawsze zapisuj `hydra` config

4. Integracja z Weights & Biases (wandb)

- Instalacja: `pip install wandb`
- Zalecana praktyka: inicjalizuj wandb w skrypcie treningowym z użyciem parametrów z Hydry.

Przykład:

```py
import wandb

def setup_wandb(cfg):
    wandb.init(project=cfg.wandb.project, config=cfg, reinit=True)
```

- Ustawienia przy uruchomieniu:

```bash
WANDB_PROJECT=myproj WANDB_MODE=online python -m src.train
```

- Dla reproducibility: zapisuj `wandb.config` bazujący na cfg Hydry, taguj runy odpowiednio, włącz `save_code=True` jeśli chcesz uploadować pliki źródłowe.

5. `pre-commit` i formatowanie

- Dodaj do repo `.pre-commit-config.yaml` z co najmniej: `black`, `isort`, `ruff`/`flake8`.
- Instalacja hooków:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Przykładowy `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.0
    hooks:
      - id: black
  - repo: https://github.com/PyCQA/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/charliermarsh/ruff
    rev: 0.1.0
    hooks:
      - id: ruff
```

6. Testy i CI

- Używaj `pytest` i `pytest-cov` do uruchamiania testów i generowania raportów.
- Przykładowy lokalny test:

```bash
pytest -q
```

- W CI (GitHub Actions) uruchamiaj: instalacja środowiska, `pre-commit run --all-files`, `pytest -q` i ewentualnie `flake8`/`ruff`.

7. Zapisywanie checkpointów, outputs

- Konwencja: `outputs/train/<experiment-name>/<timestamp>/checkpoints` i `outputs/train/<experiment-name>/<timestamp>/logs`.
- Hydra może tworzyć unikalne katalogi; ustaw `hydra.run.dir=outputs/train/${now:%Y-%m-%d_%H-%M-%S}` lub użyj `hydra` domyślnego.

8. Reproducibility i najlepsze praktyki

- Zapisuj seed w configu i w wandb
- Zapisuj wersję kodu (commit hash) w metadanych runu
- Włącz deterministyczne ustawienia w PyTorch jeśli to konieczne:

```py
import torch
torch.manual_seed(cfg.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

9. `pyproject.toml` / zarządzanie zależnościami

- Minimalny `pyproject.toml` (przykład używając Poetry):

```toml
[tool.poetry]
name = "myproj"
version = "0.1.0"
description = "Example project"
authors = ["You <you@example.com>"]

[tool.poetry.dependencies]
python = ">=3.10"
hydra-core = "^1.2"
torch = "^2.0"
wandb = "^1.0"

[tool.poetry.dev-dependencies]
pytest = "^7.0"
pre-commit = "^3.0"
black = "^24.1"
isort = "^5.12"
```

10. Przykład minimalnego configa Hydra dla MNIST

`conf/config.yaml` (główny):

```yaml
defaults:
  - dataset: mnist
  - model: simple_cnn
  - training: default

hydra:
  run:
    dir: outputs/train/${now:%Y-%m-%d_%H-%M-%S}

wandb:
  project: mnist-example

seed: 42
```

`conf/dataset/mnist.yaml`:

```yaml
name: mnist
path: ${oc.env:MNIST_PATH, data/mnist}
batch_size: 64
```

`conf/model/simple_cnn.yaml`:

```yaml
name: simple_cnn
num_classes: 10

architecture:
  channels: [32, 64]
  kernel: 3
```

`conf/training/default.yaml`:

```yaml
lr: 0.01
epochs: 10
optimizer:
  name: sgd
  momentum: 0.9
```

Uruchamianie:

```bash
python -m src.train
# lub z override'ami
python -m src.train training.epochs=20 dataset.batch_size=128
```

11. Checklist do przeniesienia tego setupu do nowego projektu
- Utwórz `pyproject.toml`/`requirements.txt` z podstawowymi dependency
- Skopiuj `conf/` z konfiguracjami Hydry i dostosuj wartości domyślne
- Dodaj `src/train.py` punkt wejścia wykorzystujący `@hydra.main`
- Dodaj hookup do `wandb` (funkcja `setup_wandb(cfg)`)
- Dodaj `.pre-commit-config.yaml` i zainstaluj hooki
- Dodaj podstawowe testy w `tests/` i CI workflow
- Ustaw `outputs/` i strategię zapisu checkpointów

12. Dodatkowe uwagi i dobre praktyki
- Trzymaj konfiguracje, które często zmieniasz, poza kodem — w Hydrze
- Minimalizuj magiczne ścieżki — używaj parametrów configu
- Loguj metadane eksperymentu (git commit, branch, note)
- Dla eksperymentów długich: rozważ enqueue/run na HPC lub scheduler

---

Plik ten można skopiować do innych projektów jako punkt startowy. Jeśli chcesz, mogę:
- dodać przykładowy `src/train.py` minimalnego trenera (MNIST),
- wygenerować `.pre-commit-config.yaml` w repozytorium,
- przygotować `pyproject.toml` z listą zależności.
