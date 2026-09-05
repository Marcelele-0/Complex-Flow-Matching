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
    
    # 1. Evaluate trained generative models from multirun
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
        datasets_found.add(dataset)
        
        if dataset not in summary:
            summary[dataset] = {"epochs_trained": epochs, "metrics": {}}
            
        print(f"  [EVAL] {manifold} na zbiorze {dataset}...")
        
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
                summary[dataset]["metrics"][manifold] = metrics

    # 2. Evaluate VarNet (which wasn't trained in train.py because it's supervised)
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
            
    summary_path = os.path.join(multirun_dir, "smoke_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n✅ Full smoke matrix evaluated!")
    print(f"📄 Wyniki zapisane w: {summary_path}")
    print(json.dumps(summary, indent=2))

def main():
    parser = argparse.ArgumentParser(description="torch-cfmri Mission Control")
    parser.add_argument("suite", choices=["smoke", "full-matrix", "seeds"], help="Suite to run")
    parser.add_argument("--slurm", action="store_true", help="Launch on SLURM via submitit")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra arguments for Hydra", default=[])
    args = parser.parse_args()

    cmd = [sys.executable, "src/cfm/train.py", "-m"]
    
    if args.suite == "smoke":
        cmd.append("+experiment=matrix_smoke")
    elif args.suite == "full-matrix":
        cmd.append("+experiment=matrix_train")
    elif args.suite == "seeds":
        cmd.extend(["training.seed=42,123,999"])
        
    if args.slurm:
        cmd.append("+hydra/launcher=submitit_slurm")
        
    cmd.extend(args.extra)
    
    print(f"🚀 Launching CFM Suite: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0 and args.suite == "smoke":
        multirun_dirs = glob.glob("outputs/multirun/*/")
        if multirun_dirs:
            latest_multirun = max(multirun_dirs, key=os.path.getmtime)
            run_evaluation(latest_multirun)
            
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
