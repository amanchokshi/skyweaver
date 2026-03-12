from __future__ import annotations

import importlib
import sys
from importlib.metadata import PackageNotFoundError


def test_version_fallback_when_package_metadata_missing(monkeypatch) -> None:
    def mock_version(_: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr("importlib.metadata.version", mock_version)

    sys.modules.pop("skyweaver", None)
    import skyweaver

    importlib.reload(skyweaver)

    assert skyweaver.__version__ == "0.0.0"
