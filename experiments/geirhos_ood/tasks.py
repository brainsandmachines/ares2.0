"""The 16 model-vs-human OOD tasks (everything except cue-conflict / shape bias).

Identifiers, conditions and excluded conditions are copied from bethgelab/model-vs-human
(master): ``modelvshuman/datasets/experiments.py`` (``data_conditions`` /
``plotting_conditions``), ``modelvshuman/plotting/plot.py`` (``EXCLUDE_CONDITIONS``, applied
because ``EXCLUDE = True``), ``modelvshuman/datasets/base.py`` (``contains_sessions``).

Condition strings are what pandas makes of the raw-data CSV column followed by
``astype(str)`` (upstream ``plotting_helper`` does the same), e.g. false-colour ``true`` ->
``"True"``, uniform-noise ``0.0`` -> ``"0.0"``.

Cue-conflict is deliberately absent: shape bias is already scored by
``data_analysis/shape_bias_selected.py`` and must not be rerun here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

EXCLUDED_TASKS = ("cue-conflict",)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str                     # upstream dataset name (release asset <task_id>.tar.gz)
    label: str                       # reader-friendly name for plots
    kind: str                        # "distortion" or "nonparametric"
    contains_sessions: bool          # images under dnn/session-N/ (else <category>/<img>)
    data_conditions: tuple[str, ...] = ()
    plotting_conditions: tuple[str, ...] = ()
    excluded_conditions: tuple[str, ...] = ()   # upstream EXCLUDE_CONDITIONS
    n_images: int | None = None      # trials per human subject file (upstream raw-data)
    n_subjects: int | None = None    # human subject files in upstream raw-data
    extra: dict = field(default_factory=dict, compare=False)

    @property
    def scored_conditions(self) -> tuple[str, ...] | None:
        """Conditions that enter the benchmark metrics; None = the whole dataset is one unit."""
        if not self.data_conditions:
            return None
        return tuple(c for c in self.data_conditions if c not in self.excluded_conditions)


def _eidolon(coherence: str) -> tuple[str, ...]:
    return tuple(f"{2 ** i}-{coherence}-10" for i in range(8))


TASKS: dict[str, TaskSpec] = {t.task_id: t for t in [
    TaskSpec("colour", "Colour / greyscale", "distortion", True,
             ("cr", "bw"), ("colour", "greyscale"), ("cr",), 1280, 4),
    TaskSpec("contrast", "Contrast", "distortion", True,
             ("c100", "c50", "c30", "c15", "c10", "c05", "c03", "c01"),
             ("100", "50", "30", "15", "10", "5", "3", "1"), ("c100", "c03", "c01"), 1280, 4),
    TaskSpec("high-pass", "High-pass filtering", "distortion", True,
             ("inf", "3.0", "1.5", "1.0", "0.7", "0.55", "0.45", "0.4"),
             ("inf", "3.0", "1.5", "1.0", ".7", ".55", ".45", ".4"),
             ("inf", "0.55", "0.45", "0.4"), 1280, 4),
    TaskSpec("low-pass", "Low-pass filtering", "distortion", True,
             ("0", "1", "3", "5", "7", "10", "15", "40"),
             ("0", "1", "3", "5", "7", "10", "15", "40"), ("0", "15", "40"), 1280, 4),
    TaskSpec("phase-scrambling", "Phase noise", "distortion", True,
             ("0", "30", "60", "90", "120", "150", "180"),
             ("0", "30", "60", "90", "120", "150", "180"), ("0", "150", "180"), 1120, 4),
    TaskSpec("power-equalisation", "Power equalisation", "distortion", True,
             ("0", "pow"), ("original", "equalised"), ("0",), 1120, 4),
    TaskSpec("false-colour", "Opponent colour", "distortion", True,
             ("True", "False"), ("true", "opponent"), ("True",), 1120, 4),
    TaskSpec("rotation", "Rotation", "distortion", True,
             ("0", "90", "180", "270"), ("0", "90", "180", "270"), ("0",), 1280, 4),
    TaskSpec("eidolonI", "Eidolon I", "distortion", True,
             _eidolon("10"), tuple(str(i) for i in range(8)),
             ("1-10-10", "64-10-10", "128-10-10"), 1280, 4),
    TaskSpec("eidolonII", "Eidolon II", "distortion", True,
             _eidolon("3"), tuple(str(i) for i in range(8)),
             ("1-3-10", "32-3-10", "64-3-10", "128-3-10"), 1280, 4),
    TaskSpec("eidolonIII", "Eidolon III", "distortion", True,
             _eidolon("0"), tuple(str(i) for i in range(8)),
             ("1-0-10", "16-0-10", "32-0-10", "64-0-10", "128-0-10"), 1280, 4),
    TaskSpec("uniform-noise", "Uniform noise", "distortion", True,
             ("0.0", "0.03", "0.05", "0.1", "0.2", "0.35", "0.6", "0.9"),
             ("0.0", ".03", ".05", ".1", ".2", ".35", ".6", ".9"), ("0.0", "0.6", "0.9"), 1280, 4),
    TaskSpec("sketch", "Sketch", "nonparametric", True, n_images=800, n_subjects=7),
    TaskSpec("stylized", "Stylized", "nonparametric", True, n_images=800, n_subjects=5),
    TaskSpec("edge", "Edge", "nonparametric", False, n_images=160, n_subjects=10),
    TaskSpec("silhouette", "Silhouette", "nonparametric", False, n_images=160, n_subjects=10),
]}

assert len(TASKS) == 16 and not set(EXCLUDED_TASKS) & set(TASKS)


def select_tasks(names: list[str] | None) -> list[TaskSpec]:
    """Tasks in registry order; refuses cue-conflict and unknown names."""
    if not names:
        return list(TASKS.values())
    bad = [n for n in names if n in EXCLUDED_TASKS]
    if bad:
        raise ValueError(f"{bad} excluded: shape bias / cue-conflict is already scored and must not be rerun")
    unknown = [n for n in names if n not in TASKS]
    if unknown:
        raise ValueError(f"Unknown task(s) {unknown}; known: {list(TASKS)}")
    return [TASKS[n] for n in TASKS if n in names]


def task_number(task_id: str) -> int:
    """1-based position in the registry (used for file names like 03_high-pass)."""
    return list(TASKS).index(task_id) + 1
