import re
import sys

files = [
    ".env.example",
    "conf/evaluate/default.yaml",
    "conf/hydra/default.yaml",
    "src/cfm/core/reconstructor.py",
    "src/cfm/data/__init__.py",
    "src/cfm/data/dataset.py",
    "src/cfm/data/download.py",
    "src/cfm/data/fastmri.py",
    "src/cfm/evaluate.py",
    "src/cfm/manifolds/complex_diffusion.py",
    "src/cfm/manifolds/cylindrical.py",
    "src/cfm/manifolds/euclidean.py",
    "src/cfm/models/varnet.py",
    "src/cfm/suite.py",
    "tests/test_data/test_download.py"
]

has_errors = False
for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as file:
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
