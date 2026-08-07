"""Validate that direct dependencies are pinned and represented in exact locks."""

from __future__ import annotations

import re
from pathlib import Path

EXACT_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)")
REQUIRED_DIRECT_VERSIONS = {
    "streamlit": "1.61.1",
    "httpx": "0.28.1",
    "python-dotenv": "1.2.2",
    "pillow": "12.3.0",
    "pytest": "9.1.1",
}


def _requirements(path: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "-r ", "--hash=")):
            continue
        match = EXACT_REQUIREMENT.match(line)
        if match is None:
            raise SystemExit(f"{path} contains a non-exact requirement: {line}")
        result[match.group(1).lower()] = match.group(2)
    return result


def main() -> None:
    runtime_direct = _requirements("requirements.in")
    development_direct = _requirements("requirements-dev.in")
    runtime_lock = _requirements("requirements.lock")
    development_lock = _requirements("requirements-dev.lock")

    for name, version in runtime_direct.items():
        if runtime_lock.get(name) != version:
            raise SystemExit(f"Runtime lock mismatch for {name}")
    for name, version in development_direct.items():
        if name in runtime_direct:
            continue
        if development_lock.get(name) != version:
            raise SystemExit(f"Development lock mismatch for {name}")
    for name, expected in REQUIRED_DIRECT_VERSIONS.items():
        version = runtime_lock.get(name) or development_lock.get(name)
        if version != expected:
            raise SystemExit(f"{name} does not match the required reviewed direct version")
    if len(runtime_lock) <= len(runtime_direct):
        raise SystemExit("Runtime candidate does not include transitive dependency pins")
    print("Dependency candidate files use internally consistent exact pins.")


if __name__ == "__main__":
    main()
