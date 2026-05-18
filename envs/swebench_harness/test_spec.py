import json
import shlex
from collections.abc import Mapping
from dataclasses import dataclass

from .constants import (
    END_TEST_OUTPUT,
    FAIL_TO_PASS,
    MAP_REPO_VERSION_TO_SPECS,
    PASS_TO_PASS,
    START_TEST_OUTPUT,
)
from .patch_utils import get_modified_files, get_new_files, get_test_directives


@dataclass(frozen=True)
class PythonEvalSpec:
    instance_id: str
    repo: str
    version: str
    base_commit: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    test_command: str
    eval_script: str


def make_python_eval_spec(instance: Mapping[str, object]) -> PythonEvalSpec:
    repo = str(instance["repo"])
    version = str(instance["version"])
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    directives = get_test_directives(instance)
    test_command = " ".join([str(specs["test_cmd"]), *directives]).strip()
    fail_to_pass = _parse_test_list(instance, FAIL_TO_PASS)
    pass_to_pass = _parse_test_list(instance, PASS_TO_PASS)

    return PythonEvalSpec(
        instance_id=str(instance["instance_id"]),
        repo=repo,
        version=version,
        base_commit=str(instance["base_commit"]),
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        test_command=test_command,
        eval_script=_make_eval_script(instance, test_command, specs),
    )


def _make_eval_script(
    instance: Mapping[str, object],
    test_command: str,
    specs: Mapping[str, object],
) -> str:
    base_commit = str(instance["base_commit"])
    test_patch_path = "/tmp/test.patch"
    modified_files = " ".join(shlex.quote(path) for path in get_modified_files(str(instance["test_patch"])))
    new_files = " ".join(shlex.quote(path) for path in get_new_files(str(instance["test_patch"])))
    reset_modified = f"git checkout {shlex.quote(base_commit)} {modified_files}" if modified_files else "true"
    reset_new = f"rm -f {new_files}" if new_files else "true"

    install = specs.get("install")
    install_command = str(install) if install else "true"

    return "\n".join(
        [
            "#!/bin/bash",
            "set -uxo pipefail",
            "export PYTHONIOENCODING=utf-8",
            "export LANG=C.UTF-8",
            "export LC_ALL=C.UTF-8",
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            "cd /testbed",
            "git config --global --add safe.directory /testbed",
            "git status",
            "git show --stat",
            f"git -c core.fileMode=false diff {shlex.quote(base_commit)}",
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            install_command,
            reset_modified,
            reset_new,
            f"git apply -v {test_patch_path}",
            f"echo {shlex.quote(START_TEST_OUTPUT)}",
            test_command,
            f"echo {shlex.quote(END_TEST_OUTPUT)}",
            reset_modified,
            reset_new,
            "",
        ]
    )


def _parse_test_list(instance: Mapping[str, object], key: str) -> list[str]:
    value = instance[key]
    if isinstance(value, str):
        return list(json.loads(value))
    return list(value)
