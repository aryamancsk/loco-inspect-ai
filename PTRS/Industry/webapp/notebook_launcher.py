import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

INDUSTRY_DIR = Path(__file__).resolve().parent.parent
if str(INDUSTRY_DIR) not in sys.path:
    sys.path.insert(0, str(INDUSTRY_DIR))


def _is_port_listening(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, port)) == 0


def _start_uvicorn_in_thread(host: str, port: int) -> None:
    if _is_port_listening(host, port):
        return
    try:
        import uvicorn
        from webapp.server import app

        cfg = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(cfg)
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        for _ in range(25):
            if _is_port_listening(host, port):
                break
            time.sleep(0.1)
    except Exception as exc:
        print(f"[LOCO-Inspect AI] Note starting server on {host}:{port}: {exc}")


def open_browser_window(url: str) -> None:
    """Reliably pop up the browser window in the foreground on Windows."""
    opened = False
    if os.name == "nt":
        try:
            os.startfile(url)
            opened = True
        except Exception:
            pass
        if not opened:
            try:
                subprocess.Popen(
                    ["cmd", "/c", "start", "", url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                opened = True
            except Exception:
                pass
    if not opened:
        try:
            webbrowser.open(url, new=1, autoraise=True)
        except Exception:
            pass


class _AppHandle:
    def __init__(self, url: str):
        self.local_url = url

    def close(self):
        pass


def launch_loco_inspect_from_notebook(pop_browser: bool = True) -> _AppHandle:
    host = "127.0.0.1"
    primary_port = 8000
    legacy_port = 7860

    _start_uvicorn_in_thread(host, primary_port)
    _start_uvicorn_in_thread(host, legacy_port)

    url = f"http://{host}:{primary_port}"
    if pop_browser:
        open_browser_window(url)
    return _AppHandle(url)
