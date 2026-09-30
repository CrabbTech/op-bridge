"""Background jobs: device operations longer than a chat client's tool-call timeout.

Chat clients cut a tool call off after about a minute, so a take, a tape capture or a long vocoder
line cannot run inside one call. A job runs the same code on a daemon thread while the tool returns
at once with a job id; job_status reports progress and, when done, the very result a direct call
would have returned. A job can be cancelled between events; the player checks the flag and the
device context releases the notes it holds."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class Cancelled(Exception):
    """Raised inside a job's work when cancel_job was called."""


@dataclass
class Job:
    id: str
    kind: str
    name: str
    expected_seconds: float
    started: float = field(default_factory=time.time)
    stage: str = "starting"
    done: bool = False
    ok: bool | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    finished: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def status(self) -> str:
        if not self.done:
            return "cancelling" if self.cancel.is_set() else "running"
        if self.ok:
            return "done"
        return "cancelled" if self.cancel.is_set() else "failed"

    def snapshot(self) -> dict[str, Any]:
        elapsed = (self.finished or time.time()) - self.started
        progress = 1.0 if self.done else (min(0.98, elapsed / self.expected_seconds) if self.expected_seconds > 0 else 0.0)
        d: dict[str, Any] = {"job_id": self.id, "kind": self.kind, "name": self.name, "status": self.status, "stage": self.stage,
                             "elapsed_seconds": round(elapsed, 1), "expected_seconds": round(self.expected_seconds, 1), "progress": round(progress, 2),
                             "remaining_seconds": 0.0 if self.done else round(max(0.0, self.expected_seconds - elapsed), 1)}
        if self.done:
            d["result"] = self.result
            d["error"] = self.error
        return d


class Jobs:
    """Registry of jobs; keeps the last `keep` finished ones."""

    def __init__(self, keep: int = 50):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._keep = keep

    def start(self, job_id: str, kind: str, name: str, expected_seconds: float, work: Callable[[Job], dict[str, Any]]) -> Job:
        job = Job(job_id, kind, name, float(expected_seconds))

        def run() -> None:
            try:
                job.result = work(job)
                job.ok = True
            except Cancelled:
                job.ok = False
                job.error = "cancelled"
            except Exception as e:  # reported through job_status, never lost
                job.ok = False
                job.error = f"{type(e).__name__}: {e}"
            finally:
                job.done = True
                job.finished = time.time()
                job.stage = "finished"

        with self._lock:
            self._jobs[job_id] = job
            finished = [j for j in self._jobs.values() if j.done]
            for old in sorted(finished, key=lambda j: j.finished or 0)[: max(0, len(finished) - self._keep)]:
                self._jobs.pop(old.id, None)
        threading.Thread(target=run, name=f"op-bridge-job-{job_id}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.started)

    def running(self) -> list[Job]:
        return [j for j in self.all() if not j.done]

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job is None:
            raise KeyError(f"no job {job_id!r}")
        job.cancel.set()
        return job


def check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()
