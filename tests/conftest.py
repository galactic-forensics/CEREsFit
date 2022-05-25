"""Provide fixtures for test suite."""

from pathlib import Path

import pytest


@pytest.fixture(scope="package")
def ds_path(request) -> Path:
    """Provide the path to the data set folder."""
    curr = Path(request.fspath).parents[0]
    return Path(curr).joinpath("datasets").absolute()
