# QfinVidCore/core/version.py
from __future__ import annotations

import argparse
import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# SemVer: MAJOR.MINOR.PATCH with optional prerelease/build metadata.
_SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


@dataclass(frozen=True)
class VersionInfo:
    version: str
    release_date: str
    commit_id: str


def _find_repo_root(start: Optional[Path] = None) -> Path:
    cursor = (start or Path(__file__)).resolve()
    for p in [cursor, *cursor.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("pyproject.toml not found from current module path")


def _pyproject_path(path: Optional[Path] = None) -> Path:
    if path is not None:
        return Path(path).resolve()
    return _find_repo_root() / "pyproject.toml"


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")


def _current_git_commit(repo_root: Optional[Path] = None) -> str:
    root = repo_root or _find_repo_root()
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        value = cp.stdout.strip()
        return value if value else "unknown"
    except Exception:
        return "unknown"


def _today() -> str:
    return date.today().strftime("%Y-%m-%d")


def _find_section_range(lines: list[str], section: str) -> tuple[int, int] | None:
    target = f"[{section}]"
    start = -1
    for i, line in enumerate(lines):
        if line.strip() == target:
            start = i
            break
    if start < 0:
        return None

    end = len(lines)
    for i in range(start + 1, len(lines)):
        s = lines[i].strip()
        if s.startswith("[") and s.endswith("]"):
            end = i
            break
    return start, end


def _get_value(lines: list[str], section: str, key: str) -> Optional[str]:
    sec = _find_section_range(lines, section)
    if sec is None:
        return None
    start, end = sec
    pat = re.compile(rf"^\s*{re.escape(key)}\s*=\s*\"(.*)\"\s*$")
    for i in range(start + 1, end):
        m = pat.match(lines[i])
        if m:
            return m.group(1)
    return None


def _upsert_value(lines: list[str], section: str, key: str, value: str) -> list[str]:
    sec = _find_section_range(lines, section)
    value_line = f'{key} = "{value}"'

    if sec is None:
        out = list(lines)
        if out and out[-1].strip() != "":
            out.append("")
        out.append(f"[{section}]")
        out.append(value_line)
        return out

    start, end = sec
    out = list(lines)
    pat = re.compile(rf"^\s*{re.escape(key)}\s*=")
    for i in range(start + 1, end):
        if pat.match(out[i]):
            out[i] = value_line
            return out

    insert_at = end
    out.insert(insert_at, value_line)
    return out


def _require_project_version(lines: list[str]) -> str:
    value = _get_value(lines, "project", "version")
    if not value:
        raise ValueError("[project].version is missing in pyproject.toml")
    return value


def update_pyproject_metadata(
    *,
    pyproject_file: Optional[Path] = None,
    commit_id: Optional[str] = None,
    release_date: Optional[str] = None,
) -> None:
    """Update commit_id and release_date in pyproject.toml metadata section."""
    pyproject = _pyproject_path(pyproject_file)
    lines = _read_lines(pyproject)

    # Ensure project.version exists and remains the source of truth.
    _require_project_version(lines)

    repo_root = pyproject.parent
    cid = commit_id or _current_git_commit(repo_root)
    rdate = release_date or _today()

    lines = _upsert_value(lines, "tool.qfinvidcore.metadata", "commit_id", cid)
    lines = _upsert_value(lines, "tool.qfinvidcore.metadata", "release_date", rdate)

    _write_lines(pyproject, lines)
    logger.info("Updated pyproject metadata: commit_id=%s, release_date=%s", cid, rdate)


def get_version(*, pyproject_file: Optional[Path] = None) -> str:
    pyproject = _pyproject_path(pyproject_file)
    lines = _read_lines(pyproject)
    return _require_project_version(lines)


def get_release_date(*, pyproject_file: Optional[Path] = None) -> str:
    pyproject = _pyproject_path(pyproject_file)
    lines = _read_lines(pyproject)
    value = _get_value(lines, "tool.qfinvidcore.metadata", "release_date")
    return value or _today()


def get_commit_id(*, pyproject_file: Optional[Path] = None) -> str:
    pyproject = _pyproject_path(pyproject_file)
    lines = _read_lines(pyproject)
    value = _get_value(lines, "tool.qfinvidcore.metadata", "commit_id")
    if value:
        return value
    return _current_git_commit(pyproject.parent)


def set_version(version: str, *, pyproject_file: Optional[Path] = None) -> None:
    """Set [project].version after SemVer validation and refresh metadata."""
    if not _SEMVER_RE.match(version):
        raise ValueError(
            f"Invalid version '{version}'. Expected SemVer, e.g. 1.2.3 or 1.2.3-rc.1"
        )

    pyproject = _pyproject_path(pyproject_file)
    lines = _read_lines(pyproject)

    if _find_section_range(lines, "project") is None:
        raise ValueError("[project] section is missing in pyproject.toml")

    lines = _upsert_value(lines, "project", "version", version)
    _write_lines(pyproject, lines)

    logger.info("Set project version to %s", version)

    # Keep metadata in sync on every version change.
    update_pyproject_metadata(pyproject_file=pyproject)


def get_version_info(*, pyproject_file: Optional[Path] = None) -> VersionInfo:
    return VersionInfo(
        version=get_version(pyproject_file=pyproject_file),
        release_date=get_release_date(pyproject_file=pyproject_file),
        commit_id=get_commit_id(pyproject_file=pyproject_file),
    )


# Backward-compatible exported variable used by QfinVidCore.__init__
try:
    sdk_version = get_version()
except Exception:
    sdk_version = "0.0.0"


__all__ = [
    "VersionInfo",
    "get_version",
    "get_release_date",
    "get_commit_id",
    "get_version_info",
    "set_version",
    "update_pyproject_metadata",
    "sdk_version",
]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Version metadata manager for pyproject.toml")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("update", help="Update commit_id and release_date from current git/date")

    show = sub.add_parser("show", help="Show current version metadata")
    show.add_argument("--json", action="store_true", help="Print as JSON-like dict string")

    setv = sub.add_parser("set-version", help="Set semantic version and refresh metadata")
    setv.add_argument("version", help="Semantic version, e.g. 1.2.3 or 1.2.3-rc.1")

    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args()

    cmd = args.command or "update"
    if cmd == "update":
        update_pyproject_metadata()
        info = get_version_info()
        print(f"version={info.version} release_date={info.release_date} commit_id={info.commit_id}")
        return 0

    if cmd == "show":
        info = get_version_info()
        if args.json:
            print(
                "{"
                f"'version': '{info.version}', "
                f"'release_date': '{info.release_date}', "
                f"'commit_id': '{info.commit_id}'"
                "}"
            )
        else:
            print(f"version={info.version}")
            print(f"release_date={info.release_date}")
            print(f"commit_id={info.commit_id}")
        return 0

    if cmd == "set-version":
        set_version(args.version)
        info = get_version_info()
        print(f"version={info.version} release_date={info.release_date} commit_id={info.commit_id}")
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
