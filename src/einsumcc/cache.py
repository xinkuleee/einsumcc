"""Versioned, persistent storage for measured plan and schedule choices."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Dict, Mapping, Optional, Tuple

from .errors import CacheError
from .plans import PlanKind
from .schedule import DirectSchedule

CACHE_SCHEMA = 1


@dataclass(frozen=True)
class TuningRecord:
    workload_key: str
    target: str
    plan_kind: PlanKind
    median_us: float
    samples_us: Tuple[float, ...]
    schedule: Optional[DirectSchedule] = None
    created_at: str = ""

    @classmethod
    def create(
        cls,
        workload_key: str,
        target: str,
        plan_kind: PlanKind,
        median_us: float,
        samples_us: Tuple[float, ...],
        schedule: Optional[DirectSchedule] = None,
    ) -> "TuningRecord":
        normalized_samples = tuple(float(value) for value in samples_us)
        if not workload_key or not target:
            raise CacheError("tuning record identity cannot be empty")
        if not math.isfinite(float(median_us)) or float(median_us) < 0:
            raise CacheError("tuning median must be a finite non-negative number")
        if not normalized_samples or any(
            not math.isfinite(value) or value < 0 for value in normalized_samples
        ):
            raise CacheError("tuning samples must be finite non-negative numbers")
        return cls(
            workload_key,
            target,
            plan_kind,
            float(median_us),
            normalized_samples,
            schedule,
            datetime.now(timezone.utc).isoformat(),
        )

    def to_json(self) -> Mapping[str, object]:
        return {
            "workload_key": self.workload_key,
            "target": self.target,
            "plan_kind": self.plan_kind.value,
            "median_us": self.median_us,
            "samples_us": list(self.samples_us),
            "schedule": None if self.schedule is None else dict(self.schedule.as_dict()),
            "created_at": self.created_at,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "TuningRecord":
        try:
            schedule_value = value.get("schedule")
            schedule = (
                None
                if schedule_value is None
                else DirectSchedule.from_dict(schedule_value)  # type: ignore[arg-type]
            )
            record = cls.create(
                str(value["workload_key"]),
                str(value["target"]),
                PlanKind(str(value["plan_kind"])),
                float(value["median_us"]),
                tuple(float(item) for item in value["samples_us"]),  # type: ignore[union-attr]
                schedule,
            )
            return cls(
                record.workload_key,
                record.target,
                record.plan_kind,
                record.median_us,
                record.samples_us,
                record.schedule,
                str(value.get("created_at", "")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CacheError("malformed tuning record: {}".format(error)) from error


class TuningCache:
    """A tiny JSON database with atomic whole-file updates."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or Path(".einsumcc-cache") / "tuning-v1.json")

    @staticmethod
    def _entry_key(workload_key: str, target: str) -> str:
        return "{}:{}".format(target, workload_key)

    def _load(self) -> Dict[str, TuningRecord]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CacheError("cannot read tuning cache '{}': {}".format(self.path, error)) from error
        if not isinstance(payload, dict):
            raise CacheError("tuning cache root must be a JSON object")
        if payload.get("schema") != CACHE_SCHEMA or not isinstance(payload.get("records"), dict):
            raise CacheError("unsupported or malformed tuning-cache schema")
        records = {}
        for key, value in payload["records"].items():
            if not isinstance(value, dict):
                raise CacheError("tuning record '{}' must be a JSON object".format(key))
            record = TuningRecord.from_json(value)
            expected = self._entry_key(record.workload_key, record.target)
            if key != expected:
                raise CacheError(
                    "tuning record key '{}' does not match record identity '{}'".format(
                        key, expected
                    )
                )
            records[str(key)] = record
        return records

    def lookup(self, workload_key: str, target: str) -> Optional[TuningRecord]:
        return self._load().get(self._entry_key(workload_key, target))

    def store(self, record: TuningRecord) -> None:
        records = self._load()
        records[self._entry_key(record.workload_key, record.target)] = record
        payload = {
            "schema": CACHE_SCHEMA,
            "records": {key: value.to_json() for key, value in sorted(records.items())},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self.path.parent),
                prefix=self.path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                temporary = Path(handle.name)
            os.replace(str(temporary), str(self.path))
        except OSError as error:
            raise CacheError("cannot update tuning cache '{}': {}".format(self.path, error)) from error

    def records(self) -> Tuple[TuningRecord, ...]:
        return tuple(self._load().values())
