from pathlib import Path
import subprocess
import sys

Path("data").mkdir(exist_ok=True)
Path("data/input.txt").write_text(("hello loop graph language model\n" * 200), encoding="utf-8")
subprocess.check_call([sys.executable, "train.py", "--data", "data/input.txt", "--out", "runs/test", "--steps", "3", "--batch-size", "4", "--block-size", "32", "--d-model", "48", "--heads", "4", "--max-loops", "3", "--device", "cpu"])
subprocess.check_call([sys.executable, "generate.py", "--ckpt", "runs/test/ckpt.pt", "--prompt", "hello", "--tokens", "20", "--device", "cpu"])
