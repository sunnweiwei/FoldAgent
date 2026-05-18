from .constants import END_TEST_OUTPUT, START_TEST_OUTPUT, ResolvedStatus, TestStatus
from .log_parsers import MAP_REPO_TO_PARSER_PY
from .test_spec import PythonEvalSpec


PASSING_STATUSES = {TestStatus.PASSED.value, TestStatus.XFAIL.value}
FAILING_STATUSES = {TestStatus.FAILED.value, TestStatus.ERROR.value}


def grade_eval_log(spec: PythonEvalSpec, log: str) -> dict:
    status_map = _parse_log(spec, log)
    fail_to_pass = _grade_cases(spec.fail_to_pass, status_map, expect_pass=True)
    pass_to_pass = _grade_cases(spec.pass_to_pass, status_map, expect_pass=True)

    resolution_status = _resolution_status(fail_to_pass, pass_to_pass)
    return {
        "instance_id": spec.instance_id,
        "repo": spec.repo,
        "version": spec.version,
        "resolved": resolution_status == ResolvedStatus.FULL.value,
        "resolution_status": resolution_status,
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "parsed_tests": len(status_map),
        "test_command": spec.test_command,
    }


def _parse_log(spec: PythonEvalSpec, log: str) -> dict[str, str]:
    parser = MAP_REPO_TO_PARSER_PY[spec.repo]
    test_output = _between_markers(log)
    status_map = parser(test_output, spec)
    if not status_map:
        status_map = parser(log, spec)
    return status_map


def _between_markers(log: str) -> str:
    start = log.find(START_TEST_OUTPUT)
    end = log.find(END_TEST_OUTPUT)
    if start == -1 or end == -1 or end <= start:
        return log
    return log[start + len(START_TEST_OUTPUT) : end]


def _grade_cases(cases: list[str], status_map: dict[str, str], expect_pass: bool) -> dict:
    success: list[str] = []
    failure: list[str] = []
    for case in cases:
        passed = status_map.get(case) in PASSING_STATUSES
        failed = case not in status_map or status_map.get(case) in FAILING_STATUSES
        ok = passed if expect_pass else failed
        if ok:
            success.append(case)
        else:
            failure.append(case)
    return {
        "success": success,
        "failure": failure,
        "success_count": len(success),
        "failure_count": len(failure),
        "total": len(cases),
        "all_success": not failure,
    }


def _ratio(section: dict) -> float:
    total = section["success_count"] + section["failure_count"]
    if total == 0:
        return 1.0
    return section["success_count"] / total


def _resolution_status(fail_to_pass: dict, pass_to_pass: dict) -> str:
    f2p = _ratio(fail_to_pass)
    p2p = _ratio(pass_to_pass)
    if f2p == 1 and p2p == 1:
        return ResolvedStatus.FULL.value
    if 0 < f2p < 1 and p2p == 1:
        return ResolvedStatus.PARTIAL.value
    return ResolvedStatus.NO.value
