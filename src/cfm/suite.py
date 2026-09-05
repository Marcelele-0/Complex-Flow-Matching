import argparse
import subprocess
import sys
import os
import json
import glob

def run_evaluation(multirun_dir):
    """Automatically run evaluation for all checkpoints and VarNet."""
    print(f"\n🧪 Automatyczna ewaluacja do pliku JSON w {multirun_dir}...")
    summary = {}
    
    datasets_found = set()
    
    job_dirs = sorted(glob.glob(os.path.join(multirun_dir, "[0-9]*")))
    for job_dir in job_dirs:
        checkpoints = glob.glob(os.path.join(job_dir, "checkpoints", "*.pt"))
        if not checkpoints:
            continue
            
        latest_ckpt = max(checkpoints, key=os.path.getmtime)
        
        with open(os.path.join(job_dir, ".hydra", "overrides.yaml"), "r") as f:
            overrides = f.read().splitlines()
            
        params = {}
        for override in overrides:
            if "=" in override:
                k, v = override.lstrip("- ").split("=", 1)
                params[k] = v
                
        dataset = params.get("dataset", "unknown")
        manifold = params.get("manifold", "unknown")
        model = params.get("model", "unknown")
        epochs = params.get("training.epochs", "unknown")
        seed = params.get("training.seed", "0")
        datasets_found.add(dataset)
        
        if dataset not in summary:
            summary[dataset] = {"epochs_trained": epochs, "metrics": {}}
            
        key = f"{manifold}_s{seed}" if "training.seed" in params else manifold
        print(f"  [EVAL] {key} na zbiorze {dataset}...")
        
        eval_cmd = [
            sys.executable, "src/cfm/evaluate.py",
            f"dataset={dataset}",
            f"manifold={manifold}",
            f"model={model}",
            "evaluate.max_samples=2",
            "evaluate.t_start=0.5",
            f"evaluate.run_name={os.path.abspath(job_dir)}",
            f"reconstruct.checkpoint_path={os.path.abspath(latest_ckpt)}"
        ]
        
        subprocess.run(eval_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        eval_dirs = glob.glob("outputs/evaluate/*/*/")
        if eval_dirs:
            latest_eval = max(eval_dirs, key=os.path.getmtime)
            metrics_file = os.path.join(latest_eval, "metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                summary[dataset]["metrics"][key] = metrics

    for dataset in datasets_found:
        print(f"  [EVAL] varnet na zbiorze {dataset}...")
        eval_cmd = [
            sys.executable, "src/cfm/evaluate.py",
            f"dataset={dataset}",
            "model=varnet",
            "evaluate.max_samples=2",
            "evaluate.run_name=varnet_smoke"
        ]
        subprocess.run(eval_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        eval_dirs = glob.glob("outputs/evaluate/*/*/")
        if eval_dirs:
            latest_eval = max(eval_dirs, key=os.path.getmtime)
            metrics_file = os.path.join(latest_eval, "metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                summary[dataset]["metrics"]["varnet"] = metrics
            
    summary_path = os.path.join(multirun_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n✅ Zakończono! Wyniki ewaluacji zapisane w: {summary_path}")
    print(json.dumps(summary, indent=2))

def main():
    parser = argparse.ArgumentParser(description="torch-cfmri Mission Control")
    parser.add_argument("--matrix", action="store_true", help="Uruchom pełną macierz modeli i zbiorów danych")
    parser.add_argument("--seeds", action="store_true", help="Dodaj wariancję na seedach (42, 123, 999)")
    parser.add_argument("--smoke", action="store_true", help="Uruchom w trybie testu (3 epoki, batch 2)")
    parser.add_argument("--eval", action="store_true", help="Automatycznie ewaluuj modele po treningu")
    parser.add_argument("--slurm", action="store_true", help="Wyślij zadania na klastry SLURM")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="Dodatkowe flagi do Hydry", default=[])
    args = parser.parse_args()

    cmd = [sys.executable, "src/cfm/train.py"]
    is_multirun = False
    
    if args.matrix:
        cmd.extend(["dataset=skm_tea,fastmri_local", "manifold=cylindrical,euclidean,complex_diffusion", "model=c_unet"])
        is_multirun = True
        
    if args.seeds:
        cmd.append("training.seed=42,123,999")
        is_multirun = True
        
    if args.smoke:
        cmd.extend(["training.epochs=3", "training.batch_size=2", "evaluate.max_samples=2", "dataset.use_cache=false"])
        
    if args.slurm:
        cmd.append("+hydra/launcher=submitit_slurm")
        is_multirun = True
        
    if is_multirun:
        cmd.insert(2, "-m")
        
    cmd.extend(args.extra)
    
    print(f"🚀 Launching CFM Suite: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0 and args.eval and is_multirun:
        multirun_dirs = glob.glob("outputs/multirun/*/")
        if multirun_dirs:
            latest_multirun = max(multirun_dirs, key=os.path.getmtime)
            run_evaluation(latest_multirun)
            
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
