import sys
import types
from pathlib import Path


class _DummyAutoAttack:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.modules.setdefault("autoattack", types.SimpleNamespace(AutoAttack=_DummyAutoAttack))

from data_analysis.autoattack_array_eval import select_balanced_indices


class _Dataset:
    def __init__(self, samples):
        self.samples = samples
        self.root = "/tmp/dataset"

    def __len__(self):
        return len(self.samples)


def test_select_balanced_indices_can_exceed_number_of_classes():
    samples = [(f"{class_idx}_{image_idx}.jpg", class_idx) for class_idx in range(1000) for image_idx in range(2)]
    ds = _Dataset(samples)

    selected = select_balanced_indices(ds, total_images=1500, seed=0)

    assert len(selected) == 1500
    assert len(set(selected)) == 1500


def test_select_balanced_indices_rejects_more_than_dataset_size():
    samples = [(f"{class_idx}_{image_idx}.jpg", class_idx) for class_idx in range(3) for image_idx in range(2)]
    ds = _Dataset(samples)

    try:
        select_balanced_indices(ds, total_images=7, seed=0)
    except ValueError as exc:
        assert "Need 7 images" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
