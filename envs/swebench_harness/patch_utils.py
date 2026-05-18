import re
from collections.abc import Mapping

from .constants import NON_TEST_EXTS


def _changed_files(patch: str) -> list[tuple[str | None, str | None]]:
    files: list[tuple[str | None, str | None]] = []
    old_path: str | None = None
    new_path: str | None = None

    for line in patch.splitlines():
        if line.startswith("diff --git "):
            if old_path is not None or new_path is not None:
                files.append((old_path, new_path))
            old_path = None
            new_path = None
        elif line.startswith("--- "):
            old_path = _parse_patch_path(line[4:])
        elif line.startswith("+++ "):
            new_path = _parse_patch_path(line[4:])

    if old_path is not None or new_path is not None:
        files.append((old_path, new_path))
    return files


def _parse_patch_path(raw: str) -> str | None:
    raw = raw.strip().split("\t", 1)[0]
    if raw == "/dev/null":
        return None
    if raw.startswith("a/") or raw.startswith("b/"):
        return raw[2:]
    return raw


def get_modified_files(patch: str) -> list[str]:
    return _unique(old for old, new in _changed_files(patch) if old and new)


def get_new_files(patch: str) -> list[str]:
    return _unique(new for old, new in _changed_files(patch) if old is None and new)


def get_test_directives(instance: Mapping[str, object]) -> list[str]:
    directives: list[str] = []
    for match in re.finditer(r"diff --git a/.* b/(.*)", str(instance["test_patch"])):
        path = match.group(1)
        if not any(path.endswith(ext) for ext in NON_TEST_EXTS):
            directives.append(path)

    if instance["repo"] == "django/django":
        directives = [_django_test_directive(path) for path in directives]

    return directives


def _django_test_directive(path: str) -> str:
    path = path.removesuffix(".py")
    path = path.removeprefix("tests/")
    return path.replace("/", ".")


def _unique(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
