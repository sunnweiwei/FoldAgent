"""SWE-Bench environment backed by a Modal Sandbox running the official eval image.

Ability string format:
    SWEModalEnv@<json-encoded-instance>

The JSON payload must contain a full SWE-Bench instance row (instance_id, repo,
version, base_commit, problem_statement, patch, test_patch, FAIL_TO_PASS,
PASS_TO_PASS). If only ``instance_id`` is present, the row is loaded from a
HuggingFace dataset via ``--dataset``/``--split`` env hints.
"""

from __future__ import annotations

import asyncio
import collections
import copy
import json
import os
import re
import time
import uuid
from typing import Any

import numpy as np

from envs.modal_sandbox import ModalSandbox, Observation
from envs.swebench_harness.grading import grade_eval_log
from envs.swebench_harness.test_spec import make_python_eval_spec


WORKDIR = "/testbed"

NO_FNCALL_PROMPT = (
    "Please continue working on the task on whatever approach you think is suitable.\n"
    "If you think you have solved the task, please first send your answer to user through "
    "message and then finish the interaction.\n"
    "If you want to give up, use the \"finish\" tool to finish the interaction."
)

AFTER_THINK_PROMPT = "Your thought has been recorded. Please continue your work."


def _swebench_image_name(instance_id: str, namespace: str = "swebench",
                         arch: str = "x86_64", tag: str = "latest") -> str:
    image_instance = instance_id.lower().replace("__", "_1776_")
    prefix = f"{namespace}/" if namespace else ""
    return f"{prefix}sweb.eval.{arch}.{image_instance}:{tag}"


def _truncate(text: str, max_lines: int = 500, max_length: int = 6000,
              keep_tail: int = 20) -> str:
    if not text:
        return text
    lines = text.splitlines()
    if len(lines) > max_lines:
        head = lines[: max_lines - keep_tail - 1]
        tail = lines[-keep_tail:]
        omitted = len(lines) - len(head) - len(tail)
        lines = head + [f"... {omitted} lines omitted ..."] + tail
    out_lines: list[str] = []
    for ln in lines:
        if len(ln) > max_length:
            ln = ln[:max_length] + "... (line truncated)"
        out_lines.append(ln)
    return "\n".join(out_lines)


def _parse_fn_call(text: str) -> dict | None:
    """Parse the last <function=name>...<parameter=k>v</parameter>...</function> block."""
    if not text:
        return None
    matches = list(re.finditer(r"<function=([^>]+)>(.*?)</function>", text, re.DOTALL))
    if not matches:
        # Tolerate a missing closing tag on the final call
        m = re.search(r"<function=([^>]+)>(.*)$", text, re.DOTALL)
        if not m:
            return None
        name = m.group(1)
        body = m.group(2)
    else:
        name = matches[-1].group(1)
        body = matches[-1].group(2)

    params: dict[str, str] = {}
    for k, v in re.findall(r"<parameter=([^>]+)>(.*?)</parameter>", body, re.DOTALL):
        params[k] = v.strip("\n")
    return {"function": name.strip(), "arguments": params}


class SWEModalEnv:
    env_str_prefix = "SWEModalEnv"

    def __init__(self, config, tokenizer, ability: str):
        self.config = config
        self.tokenizer = tokenizer
        self.ability = ability
        self.instance_info = self._load_instance(ability)

        self.sandbox: ModalSandbox | None = None
        self.spec = None
        self.eval_report: dict[str, Any] = {}

        self.think_history: list[str] = []
        self.file_originals: dict[str, str] = {}
        self.is_finish = False
        self.env_fail = False
        self.answer: str | None = None
        self.stats = collections.Counter()
        self.stats["finish"] = 0

    # ---------------------------------------------------------------- init

    def _load_instance(self, ability: str) -> dict:
        if "@" in ability:
            payload = ability.split("@", 1)[1]
        else:
            payload = ability
        info = json.loads(payload)

        # Allow lazy dataset lookup when only instance_id is provided.
        if "problem_statement" not in info or "test_patch" not in info:
            dataset_name = info.get("dataset_name") or os.getenv(
                "SWE_DATASET", "princeton-nlp/SWE-bench_Verified"
            )
            split = info.get("split") or os.getenv("SWE_SPLIT", "test")
            from datasets import load_dataset

            iid = info["instance_id"]
            for row in load_dataset(dataset_name, split=split, streaming=True):
                if row["instance_id"] == iid:
                    info = {**dict(row), **info}
                    break
            else:
                raise ValueError(f"Instance {iid} not found in {dataset_name}:{split}")
        return info

    async def init_env(self, item):
        start = time.time()
        try:
            import modal

            image_name = _swebench_image_name(
                self.instance_info["instance_id"],
                namespace=os.getenv("SWE_IMAGE_NAMESPACE", "swebench"),
                arch=os.getenv("SWE_IMAGE_ARCH", "x86_64"),
                tag=os.getenv("SWE_IMAGE_TAG", "latest"),
            )
            image = modal.Image.from_registry(image_name, add_python="3.11")
            timeout_sec = int(getattr(getattr(self.config, "plugin", {}),
                                      "session_timeout", 1800)) if self.config else 1800
            cpu = int(os.getenv("SWE_SANDBOX_CPU", "4"))

            self.sandbox = await asyncio.to_thread(
                lambda: ModalSandbox(
                    image=image,
                    app_name=os.getenv("SWE_APP_NAME", "foldagent-swe"),
                    timeout=timeout_sec + 600,
                    cpu=cpu,
                ).start()
            )
            # The prebuilt SWE-Bench image is already at base_commit with compiled
            # extensions / editable installs in place — no git or env setup needed.
            self.spec = make_python_eval_spec(self.instance_info)
        except Exception as e:
            import traceback
            print(f"[SWEModalEnv] init failed for {self.instance_info.get('instance_id')}: {e}")
            traceback.print_exc()
            self.env_fail = True
        self.stats["env_init_time"] = int(time.time() - start)
        status = "ok" if (not self.env_fail and self.spec) else "FAILED"
        print(f"[SWEModalEnv] init {status} {time.time() - start:.1f}s "
              f"instance={self.instance_info.get('instance_id')}")

    # -------------------------------------------------------------- actions

    async def run_action(self, response):
        self.stats["action"] += 1
        if self.env_fail or self.sandbox is None:
            return {"action": "finish", "arguments": {}}

        fn = _parse_fn_call(response)
        turn = self.stats["action"]
        if not fn:
            self.stats["no_fncall"] += 1
            print(f"[turn {turn:>3}] no function call")
            return {"observation": NO_FNCALL_PROMPT}

        name = fn["function"]
        args = fn["arguments"]
        preview_src = (args.get("command") or args.get("path") or args.get("message")
                       or args.get("old_str") or args.get("content") or "")
        preview = preview_src.replace("\n", " ⏎ ")[:120]
        sub = f"[{args['command']}]" if name == "str_replace_editor" and args.get("command") else ""
        print(f"[turn {turn:>3}] {name}{sub} :: {preview}")

        try:
            if name == "finish":
                self.is_finish = True
                self.stats["finish"] = 1
                self.answer = args.get("message", "")
                return {"action": "finish", "arguments": args}

            if name == "branch":  # forwarded to fold_agent
                return {"action": name, "arguments": args}

            if name == "think":
                self.think_history.append(args.get("content", ""))
                return {"observation": AFTER_THINK_PROMPT}

            if name == "execute_bash":
                cmd = args.get("command", "")
                obs = await asyncio.to_thread(
                    self.sandbox.run, cmd, cwd=WORKDIR, timeout_sec=300
                )
                return {"observation": _truncate(self._format_obs(obs))}

            if name == "str_replace_editor":
                return {"observation": _truncate(await self._str_replace_editor(args))}

            return {"observation": f"Unknown function: {name}"}
        except Exception as e:
            return {"observation": f"Action error: {e}"}

    @staticmethod
    def _format_obs(obs: Observation) -> str:
        parts = []
        if obs.stdout:
            parts.append(obs.stdout)
        if obs.stderr:
            parts.append(obs.stderr)
        if obs.error:
            parts.append(f"[error] {obs.error}")
        text = "\n".join(p for p in parts if p)
        if obs.returncode not in (0, None):
            text = (text + f"\n[exit code {obs.returncode}]").strip()
        return text or ""

    async def _str_replace_editor(self, args: dict) -> str:
        command = (args.get("command") or "").strip().lower()
        path = args.get("path") or ""
        if not path:
            return "Error: missing path"

        if command == "view":
            view_range = args.get("view_range")
            if isinstance(view_range, str):
                try:
                    view_range = json.loads(view_range)
                except Exception:
                    view_range = None
            return await self._view(path, view_range)

        if command == "create":
            return await self._create(path, args.get("file_text", ""))

        if command == "str_replace":
            return await self._str_replace(path, args.get("old_str", ""),
                                           args.get("new_str", ""))

        if command == "insert":
            try:
                line = int(args.get("insert_line", 0))
            except (TypeError, ValueError):
                return "Error: insert_line must be an integer"
            return await self._insert(path, line, args.get("new_str", ""))

        if command == "undo_edit":
            return await self._undo(path)

        return f"Error: unknown str_replace_editor command {command!r}"

    async def _read(self, path: str) -> str:
        obs = await asyncio.to_thread(self.sandbox.read_file, path)
        if not obs.ok:
            raise FileNotFoundError(obs.error or path)
        return obs.stdout

    async def _write(self, path: str, content: str) -> None:
        # Snapshot original on first edit so undo_edit works.
        if path not in self.file_originals:
            try:
                self.file_originals[path] = await self._read(path)
            except FileNotFoundError:
                self.file_originals[path] = ""
        await asyncio.to_thread(self.sandbox.run,
                                f"mkdir -p {os.path.dirname(path) or '/'}",
                                cwd=WORKDIR, timeout_sec=30)
        obs = await asyncio.to_thread(self.sandbox.write_file, path, content)
        if not obs.ok:
            raise IOError(obs.error or f"write failed: {path}")

    async def _view(self, path: str, view_range) -> str:
        ls = await asyncio.to_thread(
            self.sandbox.run, f"test -d {path} && echo DIR || echo FILE",
            cwd=WORKDIR, timeout_sec=15,
        )
        if "DIR" in ls.stdout:
            listing = await asyncio.to_thread(
                self.sandbox.run, f"ls -1 {path}", cwd=WORKDIR, timeout_sec=15,
            )
            return f"Directory listing for {path}:\n{listing.stdout}"

        try:
            content = await self._read(path)
        except FileNotFoundError:
            return f"Error: file {path} not found"

        lines = content.splitlines()
        offset = 0
        if view_range and isinstance(view_range, (list, tuple)) and len(view_range) >= 2:
            start = max(0, int(view_range[0]) - 1)
            end_v = int(view_range[1])
            end = len(lines) if end_v == -1 else min(len(lines), end_v)
            lines = lines[start:end]
            offset = start
        return "\n".join(f"{offset + i + 1:5d} | {ln}" for i, ln in enumerate(lines))

    async def _create(self, path: str, text: str) -> str:
        exists = await asyncio.to_thread(
            self.sandbox.run, f"test -e {path} && echo Y || echo N",
            cwd=WORKDIR, timeout_sec=15,
        )
        if "Y" in exists.stdout:
            return f"Error: file {path} already exists"
        await self._write(path, text)
        return f"File {path} created"

    async def _str_replace(self, path: str, old: str, new: str) -> str:
        try:
            content = await self._read(path)
        except FileNotFoundError:
            return f"Error: file {path} not found"
        old_e = old.expandtabs()
        new_e = (new or "").expandtabs()
        content_e = content.expandtabs()

        count = content_e.count(old_e)
        if count == 0:
            return f"Error: old_str not found in {path}"
        if count > 1:
            return f"Error: old_str matches {count} locations; make it unique"
        idx = content_e.index(old_e)
        line_no = content_e.count("\n", 0, idx) + 1
        new_content = content_e[:idx] + new_e + content_e[idx + len(old_e):]
        await self._write(path, new_content)
        return f"Replacement successful at line {line_no}."

    async def _insert(self, path: str, line: int, text: str) -> str:
        try:
            content = await self._read(path)
        except FileNotFoundError:
            return f"Error: file {path} not found"
        lines = content.splitlines()
        if line < 0 or line > len(lines):
            return f"Error: invalid line {line}; file has {len(lines)} lines"
        new_lines = lines[:line] + text.splitlines() + lines[line:]
        await self._write(path, "\n".join(new_lines))
        return f"Inserted {len(text.splitlines())} line(s) after line {line}."

    async def _undo(self, path: str) -> str:
        if path not in self.file_originals:
            return f"Error: file {path} has not been edited"
        await self._write(path, self.file_originals[path])
        del self.file_originals[path]
        return f"File {path} restored"

    # ---------------------------------------------------------------- reward

    async def get_reward(self, item, messages, context):
        if self.env_fail or self.sandbox is None or self.spec is None:
            print(f"[SWEModalEnv] get_reward skipped: env_fail={self.env_fail} "
                  f"sandbox={'ok' if self.sandbox else 'none'} spec={'ok' if self.spec else 'none'}")
            self.eval_report = {"error": "env_fail"}
            self._close_sandbox()
            return "", 0, self.eval_report

        report: dict[str, Any] = {}
        try:
            # Write test_patch and run the official eval script in the same sandbox.
            await asyncio.to_thread(
                self.sandbox.write_file, "/tmp/test.patch", str(self.instance_info["test_patch"])
            )
            await asyncio.to_thread(
                self.sandbox.write_file, "/tmp/run_eval.sh", self.spec.eval_script
            )
            timeout_sec = int(
                getattr(getattr(self.config, "trainer", {}), "agent_eval_timeout", 1500)
                if self.config else 1500
            )
            print(f"[SWEModalEnv] running eval_script for {self.instance_info['instance_id']}...")
            t0 = time.time()
            obs = await asyncio.to_thread(
                self.sandbox.run,
                "bash /tmp/run_eval.sh",
                cwd=WORKDIR,
                timeout_sec=timeout_sec,
            )
            log = (obs.stdout or "") + "\n" + (obs.stderr or "")
            report = grade_eval_log(self.spec, log)
            report["returncode"] = obs.returncode
            score = int(bool(report.get("resolved")))
            print(f"[SWEModalEnv] eval done in {time.time()-t0:.1f}s "
                  f"resolved={report.get('resolved')} returncode={obs.returncode} "
                  f"resolution_status={report.get('resolution_status')}")
            # keep last 4000 chars of log on the report for debugging
            report["log_tail"] = log[-4000:]
        except Exception as e:
            print(f"[SWEModalEnv] eval failed: {e}")
            report = {"error": str(e)}
            score = 0
        finally:
            self.eval_report = report
            # Surface report through env.stats so AgentLoopOutput.extra_fields['env_stats'] carries it.
            self.stats["eval_report"] = report
            self._close_sandbox()

        return "", score, report

    def _close_sandbox(self) -> None:
        if self.sandbox is not None:
            try:
                self.sandbox.close()
            except Exception:
                pass
            self.sandbox = None

    # ------------------------------------------------------------- bookkeeping

    async def update_dataproto(self, out, item, messages, score, reward_dict,
                               tag="main", metrics=None):
        final_score = score[1] if isinstance(score, (list, tuple)) else score
        out.meta_info["xperf_metrics"] = metrics
        out.meta_info["generation_kwargs"] = item.meta_info.get("generation_kwargs", {})
        out.non_tensor_batch = copy.deepcopy(item.non_tensor_batch)
        out.non_tensor_batch["num_of_turns"] = np.array([len(messages)], dtype=object)
        out.non_tensor_batch["turn_clipped"] = np.array([False], dtype=object)
        out.non_tensor_batch["tag"] = np.array([tag], dtype=object)
        out.non_tensor_batch["is_summary"] = np.array(
            [int("summary" in str(tag))], dtype=object
        )
        out.non_tensor_batch["traj_cnt"] = np.array([1], dtype=object)
        extra = {
            "score": score,
            "call_fail": self.env_fail,
            "action_fail": 0,
            "answer_reached": True,
            "stats": dict(self.stats),
            "eval_report": self.eval_report,
        }
        out.non_tensor_batch["extra_data"] = np.array([extra], dtype=object)
        return out

    def __del__(self):
        self._close_sandbox()