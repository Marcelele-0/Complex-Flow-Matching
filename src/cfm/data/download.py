import glob
import os
import subprocess

from huggingface_hub import snapshot_download


def load_env():
    """Manual .env parser to avoid heavy dependencies."""
    if not os.path.exists(".env"):
        return {}
    env = {}
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    key, val = line.split("=", 1)
                    env[key] = val.strip(' \'"')
    return env

def download_and_extract_tar(url: str, dest_dir: str):
    """Downloads a file using wget and extracts it using tar, then cleans up."""
    os.makedirs(dest_dir, exist_ok=True)
    filename = url.split("?")[0].split("/")[-1]
    filepath = os.path.join(dest_dir, filename)
    
    print(f"🌐 Zaczynam pobieranie: {filename}...")
    subprocess.run(["wget", "-c", url, "-O", filepath], check=True)
    
    print(f"📦 Rozpakowywanie archiwum: {filename}...")
    subprocess.run(["tar", "-xf", filepath, "-C", dest_dir], check=True)
    
    print(f"🧹 Sprzątanie pobranego archiwum: {filename}...")
    os.remove(filepath)

def ensure_skm_tea_mini(data_dir: str) -> None:
    """Sprawdza i ewentualnie pobiera oficjalne skm-tea-mini (16GB)."""
    if not os.path.exists(data_dir) or len(glob.glob(os.path.join(data_dir, "*.h5"))) == 0:
        print(f"📦 [Auto-Download] Brak danych w {data_dir}. Pobieranie skm-tea-mini z chmury (HuggingFace)...")
        snapshot_download(repo_id="arjundd/skm-tea-mini", repo_type="dataset", local_dir=data_dir)
        print("✅ Pobieranie SKM-TEA zakończone.")

def ensure_fastmri(data_dir: str, mode: str) -> None:
    """Sprawdza i pobiera paczki fastMRI używając linków z .env."""
    if os.path.exists(data_dir) and len(glob.glob(os.path.join(data_dir, "**/*.h5"), recursive=True)) > 0:
        return  # Data already exists
        
    print(f"📦 [Auto-Download] Brak danych fastmri w folderze: {data_dir}.")
    env = load_env()
    
    if mode == "local":
        url = env.get("FASTMRI_MINI_URL")
        if not url:
            raise ValueError("🚨 Brak linku w .env! Skopiuj .env.example do .env i uzupełnij zmienną FASTMRI_MINI_URL.")
        print("⚙️ Tryb local: Pobieram pojedynczy zestaw z FASTMRI_MINI_URL.")
        download_and_extract_tar(url, data_dir)
        
    elif mode == "full":
        urls_str = env.get("FASTMRI_FULL_URLS")
        if not urls_str:
            raise ValueError("🚨 Brak linków w .env! Skopiuj .env.example do .env i uzupełnij zmienną FASTMRI_FULL_URLS.")
        urls = [u.strip() for u in urls_str.split(",") if u.strip()]
        
        print(f"⚙️ Tryb full: Znaleziono {len(urls)} linków do pobrania. Będę pobierał, rozpakowywał i usuwał archiwa jedno po drugim.")
        for i, url in enumerate(urls, 1):
            print(f"\n--- Przetwarzanie paczki {i}/{len(urls)} ---")
            download_and_extract_tar(url, data_dir)
            
    print("✅ Wszystkie paczki fastMRI zostały pomyślnie zintegrowane.")
