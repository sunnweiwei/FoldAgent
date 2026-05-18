"""Lightweight controller for a Modal Sandbox using native exec/filesystem APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Observation:
    ok: bool
    returncode: int
    stdout: str = ""
    stderr: str = ""
    error: str = ""


class ModalSandbox:
    def __init__(
        self,
        *,
        image: Any | None = None,
        app_name: str = "foldagent-swe",
        timeout: int = 30 * 60,
        cpu: int = 2,
    ):
        self.app_name = app_name
        self.timeout = timeout
        self.cpu = cpu
        self._app = None
        self._sandbox = None
        self._image = image

    def start(self) -> "ModalSandbox":
        import modal

        self._app = modal.App.lookup(self.app_name, create_if_missing=True)
        image = self._image or modal.Image.debian_slim().apt_install("git", "patch")
        self._sandbox = modal.Sandbox.create(
            "bash",
            "-lc",
            f"sleep {self.timeout}",
            app=self._app,
            image=image,
            timeout=self.timeout + 60,
            cpu=self.cpu,
        )
        return self

    def close(self) -> None:
        if self._sandbox is not None:
            try:
                self._sandbox.terminate()
            except Exception:
                pass
            self._sandbox = None
        self._app = None

    def __enter__(self) -> "ModalSandbox":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _require(self):
        if self._sandbox is None:
            raise RuntimeError("sandbox has not been started")
        return self._sandbox

    def run(
        self,
        cmd: str,
        *,
        cwd: str = "/",
        timeout_sec: int = 300,
        env: dict[str, str] | None = None,
    ) -> Observation:
        sandbox = self._require()
        try:
            process = sandbox.exec(
                "bash",
                "-lc",
                cmd,
                timeout=timeout_sec,
                workdir=cwd or None,
                env=env or {},
            )
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            returncode = process.wait()
        except Exception as exc:
            return Observation(ok=False, returncode=-1, error=str(exc))
        return Observation(
            ok=returncode == 0,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def read_file(self, path: str) -> Observation:
        sandbox = self._require()
        try:
            content = sandbox.filesystem.read_text(path)
        except Exception as exc:
            return Observation(ok=False, returncode=-1, error=str(exc))
        return Observation(ok=True, returncode=0, stdout=content)

    def write_file(self, path: str, content: str) -> Observation:
        sandbox = self._require()
        try:
            sandbox.filesystem.write_text(content, path)
        except Exception as exc:
            return Observation(ok=False, returncode=-1, error=str(exc))
        return Observation(ok=True, returncode=0)

    def apply_patch(self, patch: str, *, cwd: str = "/testbed", timeout_sec: int = 120) -> Observation:
        if not patch:
            return Observation(ok=True, returncode=0)
        patch_path = "/tmp/foldagent-patch.diff"
        written = self.write_file(patch_path, patch)
        if not written.ok:
            return written
        first = self.run(f"git apply -v {patch_path}", cwd=cwd, timeout_sec=timeout_sec)
        if first.ok:
            return first
        second = self.run(
            f"patch --batch --fuzz=5 -p1 -i {patch_path}",
            cwd=cwd,
            timeout_sec=timeout_sec,
        )
        return Observation(
            ok=second.ok,
            returncode=second.returncode,
            stdout=(first.stdout + "\n" + second.stdout).strip(),
            stderr=(first.stderr + "\n" + second.stderr).strip(),
            error=second.error or first.error,
        )

    def diff(self, *, cwd: str = "/testbed", timeout_sec: int = 60) -> Observation:
        return self.run("git -c core.fileMode=false diff", cwd=cwd, timeout_sec=timeout_sec)