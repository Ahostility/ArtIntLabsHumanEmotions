from pathlib import Path
import sys
MAINPATH = Path(__file__).absolute().parent.parent
sys.path.append(MAINPATH.as_posix())