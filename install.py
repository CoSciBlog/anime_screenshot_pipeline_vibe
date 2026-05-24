import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def install_with_uv(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "uv", "pip", "install", *args], check=True)


def ensure_project_environment() -> None:
    if sys.prefix == sys.base_prefix or Path(sys.prefix).name.lower() != "env":
        raise SystemExit(
            "Run this installer through the project environment: "
            "env/bin/python install.py or env\\Scripts\\python.exe install.py"
        )


def bootstrap_installer() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "uv"],
        check=True,
    )


def prepare_environment() -> None:
    torch_args = ["torch", "torchvision", "torchaudio"]
    if shutil.which("nvidia-smi"):
        torch_args.extend(["--index-url", "https://download.pytorch.org/whl/cu128"])
    install_with_uv(*torch_args)
    install_with_uv("-r", str(ROOT / "requirements.txt"))
    install_with_uv("-e", str(ROOT / "waifuc"))
    install_with_uv("-e", str(ROOT))


if __name__ == "__main__":
    ensure_project_environment()
    bootstrap_installer()
    prepare_environment()
