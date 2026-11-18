"""Local stimuli + human raw-data: presence checks and a dataset that only reads local files.

Layout (the one upstream ``modelvshuman`` extracts to, under an explicit root we choose):

    <root>/datasets/<task>/dnn/session-N/<img>.png      tasks with sessions (12 distortions, sketch, stylized)
    <root>/datasets/<task>/<category>/<img>.png         edge, silhouette
    <root>/raw-data/<task>/<task>_subject-XX_session_1.csv   human trials

The root comes from ``--data-root`` or ``$GEIRHOS_OOD_ROOT`` only; there is no default and
``$MODELVSHUMANDIR`` is ignored on purpose (it is what upstream's lazy downloader writes to).
Nothing here imports ``modelvshuman`` or opens a network connection: a missing folder is a
status, never a fetch.

Filename parsing mirrors upstream ``datasets/info_mappings.py``: with sessions,
``condition = name.split("_")[3]`` and ``category = name.split("_")[4]``; without,
condition ``"NaN"`` and category = parent folder.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from data_analysis.shape_bias.dataset import default_transform
from data_analysis.shape_bias.mapping import CATEGORIES
from experiments.geirhos_ood.tasks import TaskSpec

ENV_DATA_ROOT = "GEIRHOS_OOD_ROOT"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

READY = "ready"
READY_NO_HUMAN = "ready_no_human"      # images present, human raw-data missing -> accuracy only
MISSING_DATA = "missing_data"
BAD_LAYOUT = "bad_layout"
NO_ROOT = "no_data_root"


def resolve_data_root(cli_value: str | Path | None) -> Path | None:
    value = cli_value or os.environ.get(ENV_DATA_ROOT)
    return Path(value).expanduser() if value else None


def parse_image_path(path: Path, task: TaskSpec) -> dict:
    name = path.name
    if task.contains_sessions:
        parts = name.split("_")
        if len(parts) < 5:
            raise ValueError(f"{name}: expected <idx>_<code>_s<NN>_<condition>_<category>_..., got {len(parts)} fields")
        session, condition, category = path.parent.name, parts[3], parts[4]
    else:
        session, condition, category = "session-1", "NaN", path.parent.name
    if category not in CATEGORIES:
        raise ValueError(f"{path}: category {category!r} is not one of the 16 categories")
    return {"image_path": str(path), "imagename": name, "session": session,
            "condition": condition, "category": category}


def image_dir(root: Path, task: TaskSpec) -> Path:
    base = root / "datasets" / task.task_id
    return base / "dnn" if task.contains_sessions else base


def list_images(root: Path, task: TaskSpec) -> list[Path]:
    d = image_dir(root, task)
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob("*/*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def human_csvs(root: Path, task: TaskSpec) -> list[Path]:
    d = root / "raw-data" / task.task_id
    return sorted(d.glob(f"{task.task_id}_subject-*_session_*.csv")) if d.is_dir() else []


@dataclass
class TaskDataStatus:
    task_id: str
    status: str
    image_dir: str
    n_images: int
    n_images_expected: int | None
    n_human_files: int
    n_human_expected: int | None
    detail: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def check_task(root: Path | None, task: TaskSpec) -> TaskDataStatus:
    """Read-only presence check. Never creates, fetches or modifies anything."""
    if root is None:
        return TaskDataStatus(task.task_id, NO_ROOT, "", 0, task.n_images, 0, task.n_subjects,
                              f"no data root: pass --data-root or set ${ENV_DATA_ROOT}")
    d = image_dir(root, task)
    images = list_images(root, task)
    humans = human_csvs(root, task)
    base = dict(task_id=task.task_id, image_dir=str(d), n_images=len(images),
                n_images_expected=task.n_images, n_human_files=len(humans), n_human_expected=task.n_subjects)
    if not images:
        return TaskDataStatus(status=MISSING_DATA, detail=f"no images under {d}", **base)
    if task.contains_sessions:
        odd = [p.name for p in d.iterdir() if not p.name.startswith("session-")]
        if odd:
            return TaskDataStatus(status=BAD_LAYOUT, detail=f"non-session entries in {d}: {odd[:3]}", **base)
    try:
        for p in images:
            parse_image_path(p, task)
    except ValueError as err:
        return TaskDataStatus(status=BAD_LAYOUT, detail=str(err), **base)
    notes = []
    if task.n_images is not None and len(images) != task.n_images:
        notes.append(f"{len(images)} images, upstream human sessions have {task.n_images} trials")
    if not humans:
        notes.append("no human raw-data: accuracy only, no consistency metrics")
        return TaskDataStatus(status=READY_NO_HUMAN, detail="; ".join(notes), **base)
    return TaskDataStatus(status=READY, detail="; ".join(notes), **base)


def is_runnable(status: TaskDataStatus) -> bool:
    return status.status in (READY, READY_NO_HUMAN)


class OODTaskDataset(Dataset):
    """Local images of one task, with upstream's metadata per image."""

    def __init__(self, root: Path, task: TaskSpec, transform=None):
        images = list_images(root, task)
        if not images:
            raise FileNotFoundError(f"{task.task_id}: no images under {image_dir(root, task)}")
        self.samples = [parse_image_path(p, task) for p in images]
        self.transform = transform or default_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        with Image.open(sample["image_path"]) as image:
            return self.transform(image.convert("RGB")), idx
