from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EpsilonSchedule:
    schedule_type: str
    source_epsilon: Optional[float]
    target_epsilon: float
    warmup_epochs: int = 0
    ramp_start_epoch: Optional[int] = None
    ramp_end_epoch: Optional[int] = None
    fixed_start_epoch: Optional[int] = None

    def value_for_epoch(self, epoch: int) -> float:
        epoch_one_based = int(epoch) + 1
        schedule_type = str(self.schedule_type).lower()

        if schedule_type == "fixed":
            return float(self.target_epsilon)

        if schedule_type == "linear_ramp":
            return _linear_value(
                epoch_one_based,
                start_epoch=int(self.ramp_start_epoch or 1),
                end_epoch=int(self.ramp_end_epoch or 1),
                source=float(_required(self.source_epsilon, "source_epsilon")),
                target=float(self.target_epsilon),
            )

        if schedule_type == "warmup_ramp_fixed":
            source = float(_required(self.source_epsilon, "source_epsilon"))
            if epoch_one_based <= int(self.warmup_epochs):
                return source
            if self.fixed_start_epoch is not None and epoch_one_based >= int(self.fixed_start_epoch):
                return float(self.target_epsilon)
            return _linear_value(
                epoch_one_based,
                start_epoch=int(_required(self.ramp_start_epoch, "ramp_start_epoch")),
                end_epoch=int(_required(self.ramp_end_epoch, "ramp_end_epoch")),
                source=source,
                target=float(self.target_epsilon),
            )

        raise ValueError(f"Unsupported epsilon_schedule.type: {self.schedule_type}")


def build_epsilon_schedule(cfg) -> Optional[EpsilonSchedule]:
    schedule = cfg.epsilon_schedule
    if not bool(schedule.enabled):
        return None

    target = schedule.target_epsilon
    if target is None:
        raise ValueError("epsilon_schedule.target_epsilon is required when epsilon_schedule.enabled=true")

    return EpsilonSchedule(
        schedule_type=schedule.get("type", "fixed"),
        source_epsilon=schedule.source_epsilon,
        target_epsilon=float(target),
        warmup_epochs=int(schedule.warmup_epochs or 0),
        ramp_start_epoch=schedule.ramp_start_epoch,
        ramp_end_epoch=schedule.ramp_end_epoch,
        fixed_start_epoch=schedule.fixed_start_epoch,
    )


def normalize_epsilon(epsilon: float, attack_norm: str) -> float:
    if attack_norm == "linf":
        return float(epsilon) / 255.0
    if attack_norm == "l1":
        return float(epsilon) * 255.0 / 2.0
    if attack_norm == "l2":
        return float(epsilon)
    raise ValueError(f"Unsupported attack norm: {attack_norm}")


def denormalize_epsilon(epsilon: float, attack_norm: str) -> float:
    if attack_norm == "linf":
        return float(epsilon) * 255.0
    if attack_norm == "l1":
        return float(epsilon) / (255.0 / 2.0)
    if attack_norm == "l2":
        return float(epsilon)
    raise ValueError(f"Unsupported attack norm: {attack_norm}")


def default_step_for_epsilon(epsilon: float, attack_norm: str, attack_it: int) -> Optional[float]:
    if attack_norm == "linf":
        return float(epsilon) / max(int(attack_it), 1)
    if attack_norm == "l2":
        return 2.0 * float(epsilon) / max(int(attack_it), 1)
    return None


def _linear_value(epoch: int, start_epoch: int, end_epoch: int, source: float, target: float) -> float:
    if end_epoch < start_epoch:
        raise ValueError("epsilon ramp_end_epoch must be >= ramp_start_epoch")
    if epoch <= start_epoch:
        return source
    if epoch >= end_epoch:
        return target
    span = end_epoch - start_epoch
    ratio = (epoch - start_epoch) / float(span)
    return source + ratio * (target - source)


def _required(value, name: str):
    if value is None:
        raise ValueError(f"epsilon_schedule.{name} is required for this schedule type")
    return value
