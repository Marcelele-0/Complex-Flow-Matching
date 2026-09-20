import sys

files = [
    ".env.example",
    "conf/evaluate/default.yaml",
    "conf/hydra/default.yaml",
    "src/cyfm/core/reconstructor.py",
    "src/cyfm/data/__init__.py",
    "src/cyfm/data/dataset.py",
    "src/cyfm/data/download.py",
    "src/cyfm/data/fastmri.py",
    "src/cyfm/evaluate.py",
    "src/cyfm/manifolds/complex_diffusion.py",
    "src/cyfm/manifolds/cylindrical.py",
    "src/cyfm/manifolds/euclidean.py",
    "src/cyfm/models/varnet.py",
    "src/cyfm/suite.py",
    "tests/test_data/test_download.py",
]

has_errors = False
for f in files:
    try:
        with open(f, encoding="utf-8") as file:
            for i, line in enumerate(file, 1):
                # check for non-ascii
                non_ascii = [c for c in line if ord(c) > 127]
                if non_ascii:
                    print(f"{f}:{i}: Non-ASCII found: {''.join(non_ascii)}")
                    has_errors = True
    except Exception as e:
        print(f"Error reading {f}: {e}")

if has_errors:
    sys.exit(1)
else:
    print("All clean!")
