"""How PAC's jobs share the workers asked between their processes."""

import pytest

from masw.runners.inversion import chain_jobs


@pytest.mark.parametrize(
    ("workers", "windows", "expected"),
    [(6, 12, 1), (6, 6, 1), (6, 3, 2), (6, 2, 3), (6, 1, 5), (2, 4, 1), (1, 1, 1)],
)
def test_an_inversion_takes_the_workers_asked_and_no_more(
    workers: int, windows: int, expected: int
) -> None:
    # The windows running at once and each window's chains in the workers they leave idle,
    # never more processes than the workers, nor than a window's 5 chains.
    running = min(workers, windows)

    assert chain_jobs(workers, running, chains=5) == expected
    assert running * expected <= workers
