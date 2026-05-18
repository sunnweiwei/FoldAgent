from enum import Enum

from .python_specs import (
    MAP_REPO_TO_INSTALL_PY,
    MAP_REPO_VERSION_TO_SPECS_PY,
    USE_X86_PY,
)


KEY_INSTANCE_ID = "instance_id"
KEY_MODEL = "model_name_or_path"
KEY_PREDICTION = "model_patch"

FAIL_TO_PASS = "FAIL_TO_PASS"
PASS_TO_PASS = "PASS_TO_PASS"
FAIL_TO_FAIL = "FAIL_TO_FAIL"
PASS_TO_FAIL = "PASS_TO_FAIL"

START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"

NON_TEST_EXTS = [
    ".json",
    ".png",
    "csv",
    ".txt",
    ".md",
    ".jpg",
    ".jpeg",
    ".pkl",
    ".yml",
    ".yaml",
    ".toml",
]


class TestStatus(Enum):
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    XFAIL = "XFAIL"


class ResolvedStatus(Enum):
    NO = "RESOLVED_NO"
    PARTIAL = "RESOLVED_PARTIAL"
    FULL = "RESOLVED_FULL"


MAP_REPO_VERSION_TO_SPECS = MAP_REPO_VERSION_TO_SPECS_PY
MAP_REPO_TO_INSTALL = MAP_REPO_TO_INSTALL_PY
USE_X86 = USE_X86_PY
