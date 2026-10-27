import sys
import webbrowser
from pathlib import Path
import uvicorn

# Ensure PTRS/Industry is on sys.path
INDUSTRY_DIR = Path(__file__).resolve().parent
if str(INDUSTRY_DIR) not in sys.path:
    sys.path.insert(0, str(INDUSTRY_DIR))

from webapp.server import app  # noqa: E402


def main(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True):
    url = f"http://{host}:{port}"
    print("=" * 68)
    print(f"  LOCO-Inspect AI Web Application running at: {url}")
    print("  Demo Login: username 'demo' (or demo@nus.edu.sg) / password 'demo1234'")
    print("=" * 68)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main(open_browser=False)
