import subprocess
import time
from pathlib import Path
from PIL import Image

OUT_DIR = Path("PTRS/Industry/outputs").resolve()
EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

# 1. Capture standard views
views = [
    ("website_auth.png", "http://127.0.0.1:8000/?view=auth", 2, 1440, 960, 2.0),
    ("website_dashboard.png", "http://127.0.0.1:8000/?view=dashboard", 2, 1440, 960, 2.0),
    ("website_inspection.png", "http://127.0.0.1:8000/?view=inspection", 4, 1440, 960, 2.0),
]

for filename, url, delay, w, h, scale in views:
    target = OUT_DIR / filename
    print(f"Capturing {filename} from {url}...")
    args = [
        EDGE_EXE,
        "--headless=new",
        "--disable-gpu",
        f"--window-size={w},{h}",
        f"--force-device-scale-factor={scale}",
        f"--virtual-time-budget={delay * 1000}",
        f"--screenshot={str(target)}",
        url
    ]
    subprocess.run(args, check=True)
    print(f"  -> Success: {target.name} ({target.stat().st_size / 1024:.1f} KB)")

# 2. Capture Ultra-HD unblurred Inspection History (focused on active tab + table)
print("Capturing Ultra-HD unblurred website_history.png...")
temp_hist = OUT_DIR / "temp_hist.png"
args_hist = [
    EDGE_EXE,
    "--headless=new",
    "--disable-gpu",
    "--window-size=1420,950",
    "--force-device-scale-factor=2.5",
    "--virtual-time-budget=2500",
    f"--screenshot={str(temp_hist)}",
    "http://127.0.0.1:8000/?view=history"
]
subprocess.run(args_hist, check=True)
im_hist = Image.open(temp_hist)
crop_hist = im_hist.crop((35, 170, 3515, 1500))
target_hist = OUT_DIR / "website_history.png"
crop_hist.save(target_hist, quality=95)
temp_hist.unlink(missing_ok=True)
print(f"  -> Success: website_history.png ({crop_hist.size}, {target_hist.stat().st_size / 1024:.1f} KB)")

# 3. Capture Ultra-HD unblurred Modern Benchmark Analysis (focused on active tab + benchmark cards)
print("Capturing Ultra-HD unblurred website_benchmarks.png...")
temp_bench = OUT_DIR / "temp_bench.png"
args_bench = [
    EDGE_EXE,
    "--headless=new",
    "--disable-gpu",
    "--window-size=1420,1350",
    "--force-device-scale-factor=2.5",
    "--virtual-time-budget=2500",
    f"--screenshot={str(temp_bench)}",
    "http://127.0.0.1:8000/?view=analytics"
]
subprocess.run(args_bench, check=True)
im_bench = Image.open(temp_bench)
crop_bench = im_bench.crop((35, 170, 3515, 2900))
target_bench = OUT_DIR / "website_benchmarks.png"
crop_bench.save(target_bench, quality=95)
temp_bench.unlink(missing_ok=True)
print(f"  -> Success: website_benchmarks.png ({crop_bench.size}, {target_bench.stat().st_size / 1024:.1f} KB)")
