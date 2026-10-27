import sys
from pathlib import Path

INDUSTRY_DIR = Path(__file__).resolve().parent / "PTRS" / "Industry"
if str(INDUSTRY_DIR) not in sys.path:
    sys.path.insert(0, str(INDUSTRY_DIR))

from app import main  # noqa: E402

if __name__ == "__main__":
    main(open_browser=True)
