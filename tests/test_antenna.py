# tests/gen_tests.py
import sys
from pathlib import Path
# Ensure project root is importable when this test is run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.antenna_gen as antenna_gen


def test_localPosition():
    arr = antenna_gen.antenna()
    localPos = arr.getLocalPosition()
    assert localPos.shape[0] == 3
    assert localPos.shape[1] == arr.x_elements * arr.y_elements


def test_localPosition_circular_masks_corners():
    arr = antenna_gen.antenna(shape="circular", x_elements=5, y_elements=5)
    localPos = arr.getLocalPosition()
    assert localPos.shape[0] == 3
    assert localPos.shape[1] < arr.x_elements * arr.y_elements


def test_localPosition_circular_even_grid_not_overtrimmed():
    arr = antenna_gen.antenna(shape="circular", x_elements=6, y_elements=6)
    localPos = arr.getLocalPosition()
    # 6x6 should look like a rounded square (typically 32 elems), not collapse to 4x4 (16 elems).
    assert localPos.shape[1] > 16
