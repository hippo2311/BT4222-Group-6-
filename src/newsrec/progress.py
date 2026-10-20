"""Approximate replay timing for display only; never used to compute features."""

import math
import time
from datetime import datetime, timedelta


def duration(seconds):
    if seconds < 60:
        return f"{max(0, math.ceil(seconds))}s"
    minutes = max(1, math.ceil(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"


class ReplayProgress:
    """Blend timestamp and exposure progress to allow for varying slate sizes.

    Equal weights are a workload heuristic, not a runtime guarantee. Smoke runs
    use timestamp progress alone because their exposure total is not known.
    """

    def __init__(self, total_batches, total_events=None, *, clock=time.perf_counter):
        self.total_batches = total_batches
        self.total_events = total_events
        self.clock = clock
        self.started = clock()

    def message(self, batches, events, *, now=None):
        elapsed = max(0, self.clock() - self.started)
        fraction = min(1, batches / max(1, self.total_batches))
        if self.total_events:
            fraction = (fraction + min(1, events / self.total_events)) / 2
        prefix = (
            f"{batches:,}/{self.total_batches:,} timestamp batches "
            f"({batches / max(1, self.total_batches):.1%}); "
            f"elapsed {duration(elapsed)}"
        )
        if batches >= self.total_batches:
            return prefix + "; replay complete; finalizing files and checks"
        if fraction <= 0 or elapsed < 30:
            return prefix + "; ETA warming up"
        remaining = elapsed * (1 - fraction) / fraction
        finish = (now or datetime.now().astimezone()) + timedelta(seconds=remaining)
        return (
            prefix
            + f"; replay ETA ~{duration(remaining)}; finish ~{finish:%Y-%m-%d %H:%M %Z}"
            + " (approx; final checks follow)"
        )
