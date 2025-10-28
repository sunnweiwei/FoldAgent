import asyncio
import collections
import copy
import json
import os
import re
import time
import uuid
from itertools import groupby
from typing import *
import numpy as np
import requests
import torch


ENV_PREFIX_MAP = {}


def register_env(cls):
    assert hasattr(cls, "env_str_prefix"), f"cls {cls} has no env_str_prefix"
    assert cls.env_str_prefix not in ENV_PREFIX_MAP, f"cls {cls} has duplicate env_str_prefix"
    ENV_PREFIX_MAP[cls.env_str_prefix] = cls
    return cls


def get_agent_env_from_str(env_str: str):
    if env_str is None or "@" not in env_str:
        return None

    prefix = env_str.split("@")[0]
    if prefix in ENV_PREFIX_MAP:
        env = ENV_PREFIX_MAP[prefix].from_env_str(env_str)
        return env
    else:
        raise NotImplementedError(f"Unknown env_str: {env_str}")



class GymEnv:
    # Gym stype env wrapper
    def __init__(self, config, tokenizer, ability):
        self.config = config
        self.tokenizer = tokenizer
        self.ability = ability
        self.gym = get_agent_env_from_str(ability)
        self.instance_info = self.gym.instance_info
        self.stats = collections.Counter()
        self.stats['finish'] = 0
        self.env_fail = False

    async def init_env(self, item):
        start_env = time.time()
        cc = 0
        while True:
            cc += 1
            if self.gym.ping():
                break
            if cc == 60:  # Only log if taking too long
                print(f"WARNING: Server initialization taking longer than expected ({cc}s)")
            if cc == 300:  # Only log if taking too long
                print(f"WARNING: Server initialization taking longer than expected ({cc}s)")
            if cc >= 60 * 10:
                print(f"ERROR: Server initialization timeout after ({cc}s)")
                break
            await asyncio.sleep(1)
        self.stats['env_init_time'] = int(time.time() - start_env)
        print('ENV START COST', time.time() - start_env)

    async def get_data(self, item, context):
        if 'prompt' in item.non_tensor_batch['extra_info'][0]:
            prompt = item.non_tensor_batch['extra_info'][0]['prompt']
            conversations = [
                {'role': 'system', 'content': prompt[0]['content']},
                {'role': 'user', 'content': prompt[1]['content']},
            ]

        else:
            instance_info = self.instance_info
            user_prompt = instance_info['problem_statement']
            system_prompt = "\nYou are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.\n<IMPORTANT>\n* If user provides a path, you should NOT assume it's relative to the current working directory. Instead, you should explore the file system to find the file before working on it.\n* When configuring git credentials, use \"openhands\" as the user.name and \"openhands@all-hands.dev\" as the user.email by default, unless explicitly instructed otherwise.\n* The assistant MUST NOT include comments in the code unless they are necessary to describe non-obvious behavior.\n</IMPORTANT>\n\nYou have access to the following functions:\n\n---- BEGIN FUNCTION #1: execute_bash ----\nDescription: Execute a bash command in the terminal.\n* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.\n* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together.\n\nParameters:\n  (1) command (string, required): The bash command to execute. Can be empty string to view additional logs when previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt the currently running process. Note: You can only execute one bash command at a time. If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together.\n---- END FUNCTION #1 ----\n\n---- BEGIN FUNCTION #2: finish ----\nDescription: Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.\nNo parameters are required for this function.\n---- END FUNCTION #2 ----\n\n---- BEGIN FUNCTION #3: str_replace_editor ----\nDescription: Custom editing tool for viewing, creating and editing files in plain-text format\n* State is persistent across command calls and discussions with the user\n* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep\n* The `create` command cannot be used if the specified `path` already exists as a file\n* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`\n* The `undo_edit` command will revert the last edit made to the file at `path`\n\nNotes for using the `str_replace` command:\n* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!\n* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique\n* The `new_str` parameter should contain the edited lines that should replace the `old_str`\n\nParameters:\n  (1) command (string, required): The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.\nAllowed values: [`view`, `create`, `str_replace`, `insert`, `undo_edit`]\n  (2) path (string, required): Absolute path to file or directory, e.g. `/workspace/file.py` or `/workspace`.\n  (3) file_text (string, optional): Required parameter of `create` command, with the content of the file to be created.\n  (4) old_str (string, optional): Required parameter of `str_replace` command containing the string in `path` to replace.\n  (5) new_str (string, optional): Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.\n  (6) insert_line (integer, optional): Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.\n  (7) view_range (array, optional): Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.\n---- END FUNCTION #3 ----\n\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<function=example_function_name>\n<parameter=example_parameter_1>value_1</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format, start with <function= and end with </function>\n- Required parameters MUST be specified\n- Only call one function at a time\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>\n\n"
            conversations = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ]
        meta_info = copy.copy(item.meta_info)
        meta_info['uid'] = item.non_tensor_batch['uid'][0]
        meta_info['reward_model'] = item.non_tensor_batch['reward_model'][0]

        if "max_turn" in item.meta_info:
            max_turn = item.meta_info["max_turn"]
        else:
            if context.is_train:
                max_turn = self.config.plugin.max_turn
            else:
                if "val_max_turn" in self.config.plugin:
                    max_turn = self.config.plugin.val_max_turn
                else:
                    max_turn = self.config.plugin.max_turn
        return conversations, {'max_turn': max_turn, 'instance_info': self.instance_info, 'meta_info': meta_info}

    async def run_action(self, response):
        self.stats['action'] += 1
        success, observation = await asyncio.to_thread(self.gym.step, response)
        if observation == "Task finished":
            return {'action': 'finish'}

        return {'observation': observation}

    async def get_reward(self, item, messages, context):
        if self.env_fail:  # If env fail, direct return 0 reward
            return "", 0, {}

        reward = await asyncio.to_thread(lambda: self.gym.reward)
        self.gym.release()
        return "", reward, {}

    async def update_dataproto(self, out, item, messages, score, reward_dict, tag='main', metrics=None):
        final_score = score[1]
        out.batch['swalm_agent_score'] = torch.Tensor([final_score]).to(torch.float32)
        out.meta_info["xperf_metrics"] = metrics
        out.meta_info["generation_kwargs"] = item.meta_info['generation_kwargs']
        out.non_tensor_batch = copy.deepcopy(item.non_tensor_batch)
        out.non_tensor_batch["num_of_turns"] = np.array([len(messages)], dtype=object)
        out.non_tensor_batch["turn_clipped"] = np.array([False], dtype=object)
        out.non_tensor_batch["tag"] = np.array([tag, ], dtype=object)
        out.non_tensor_batch["is_summary"] = np.array([int("summary" in tag), ], dtype=object)
        out.non_tensor_batch["traj_cnt"] = np.array([1, ], dtype=object)
        stats = dict(self.stats)
        extra_data = {"score": score, "call_fail": self.env_fail, "action_fail": 0, "answer_reached": True,
                      "stats": stats}
        out.non_tensor_batch['extra_data'] = np.array([extra_data, ], dtype=object)
        return out



NO_FNCALL_PROMPT = """
Please continue working on the task on whatever approach you think is suitable.
If you think you have solved the task, please first send your answer to user through message and then finish the interaction.
IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.
If you want to give up, use the "finish" tool to finish the interaction.
"""

AFTER_THINK_PROMPT = "You thought has been recorded. Please continue your work."


class SimpleHttpClient:
    def __init__(self, base_url, timeout=30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

        # Set up session headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'CodeServer/3.0'
        })

    def get(self, endpoint):
        """Make GET request"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()

        try:
            return response.json()
        except json.JSONDecodeError:
            return {'content': response.text} if response.text else {}

    def post(self, endpoint, data=None):
        """Make POST request"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.post(url, json=data, timeout=self.timeout)
        response.raise_for_status()

        try:
            return response.json()
        except json.JSONDecodeError:
            return {'content': response.text} if response.text else {}


def truncate_text(
        text: str,
        max_lines: int | None = None,
        max_length: int | None = None,
        merge_repeat: bool = False,
        merge_num: int = 128,
        keep_tail_lines: int = 5,
) -> str:
    lines = text.splitlines()

    # 1) Merge repeated lines if requested
    if merge_repeat:
        merged: list[str] = []
        for line, group in groupby(lines):
            grp = list(group)
            cnt = len(grp)
            if cnt > merge_num:
                merged += [line] * 2
                merged.append(f"[This line repeated {cnt - 4} more times]")
                merged += [line] * 2
            else:
                merged += grp
        lines = merged

    # 2) Line-count truncation (keep last keep_tail_lines)
    if max_lines is not None and len(lines) > max_lines:
        total = len(lines)
        if max_lines <= keep_tail_lines + 1:
            lines = lines[:max_lines]
        else:
            head_count = max_lines - keep_tail_lines - 1
            head = lines[:head_count]
            tail = lines[-keep_tail_lines:]
            omitted = total - head_count - keep_tail_lines
            lines = head + [f"… {omitted} lines omitted …"] + tail

    # 3) Per-line character-length truncation
    if max_length is not None:
        truncated_lines: list[str] = []
        for line in lines:
            if len(line) > max_length:
                truncated_lines.append(line[:max_length] + "… (truncated)")
            else:
                truncated_lines.append(line)
        lines = truncated_lines
    return "\n".join(lines)


def convert_non_fncall_messages_to_fncall_messages(messages, tools):
    content = (messages[0]['content'] or '') + (
        '' if (messages[0]['content'] or '').endswith('</function>') else '</function>')
    match = re.search(r'<function=([^>]+)>\n?(.*?)</function>', content, re.DOTALL)
    if not match: return messages[0]

    name, body = match.groups()
    tool = next((t['function'] for t in tools if t.get('function', {}).get('name') == name), None)
    if not tool: return messages[0]

    props = tool.get('parameters', {}).get('properties', {})
    params = {}
    for m in re.finditer(r'<parameter=([^>]+)>(.*?)</parameter>', body, re.DOTALL):
        k, v = m.group(1), m.group(2).strip()
        if k in props:
            try:
                params[k] = int(v) if props[k].get('type') == 'integer' else json.loads(v) if props[k].get(
                    'type') == 'array' else v
            except:
                params[k] = v

    if not set(tool.get('parameters', {}).get('required', [])).issubset(params.keys()): return messages[0]

    return [{'role': 'assistant', 'content': content.split('<function=')[0].strip(),
            'tool_calls': [{'index': 1, 'id': 'toolu_01', 'type': 'function',
                            'function': {'name': name, 'arguments': json.dumps(params)}}]}]



@register_env
class FileLocEnv:
    """
    File localization env, read-only actions
    """
    env_str_prefix = "FileLocEnv"

    def __init__(self, env_str, service_url, **kwargs):
        self.session_id = str(uuid.uuid4())
        self.env_str = env_str
        self.instance_info = json.loads(self.env_str)
        self.service_url = service_url
        self.kwargs = kwargs

        # Get instance ID for base_dir mapping
        self.instance_id = self.instance_info.get('instance_id', 'default')
        base_dir_base = os.getenv('BASE_DIR_PATH', './gym_data')
        self.base_dir = f"{base_dir_base}/{self.instance_id}"
        self.answer = None

        # Simple state management
        self._finish_called = False
        self.think_history = []
        self.client = SimpleHttpClient(service_url)

    def ping(self):
        """Check if the service is responding"""
        try:
            response = self.client.get('ping')

            if isinstance(response, dict):
                content = response.get('content', response.get('result', ''))
            else:
                content = str(response)

            return content == 'pong' or 'pong' in content.lower()
        except Exception as e:
            self._try_next_port()
            print(f"Ping failed: {e}")
            return False

    def _try_next_port(self):
        """Try ports from 8000 to 8099 until one works"""
        for port in range(8000, 8100):
            try:
                # Build new URL with this port
                if '://' in self.service_url:
                    from urllib.parse import urlparse
                    parsed = urlparse(self.service_url)
                    base = parsed.hostname
                    new_url = f"http://[{base}]:{port}"
                else:
                    new_url = f"http://localhost:{port}"

                # Test this port
                old_timeout = self.client.timeout
                self.client.timeout = 3  # Short timeout for testing
                self.client.base_url = new_url

                if self.ping():
                    print(f"Switched to port {port}")
                    self.service_url = new_url
                    self.client.timeout = old_timeout
                    return True

            except:
                pass
            finally:
                self.client.timeout = old_timeout

        return False

    def _call_service(self, provider: str, action_id: str, data: dict) -> str:
        """Call the new_main.py service with base_dir, retry 3 times with 120s timeout each"""
        # Try ping first, switch port if needed
        if not self.ping():
            if not self._try_next_port():
                return "Service error: No available service found on ports 8000-8099"

        max_retries = 3
        timeout = 120

        # Store original timeout
        original_timeout = self.client.timeout

        for attempt in range(max_retries):
            try:
                # Update client timeout for this call
                self.client.timeout = timeout

                endpoint = f"api/v1/actions/{provider}"
                payload = {
                    "action_id": action_id,
                    "data": data,
                    "base_dir": self.base_dir
                }

                response = self.client.post(endpoint, payload)

                # Restore original timeout
                self.client.timeout = original_timeout

                # Handle different response formats
                if isinstance(response, dict):
                    return response.get('result', response.get('content', str(response)))
                else:
                    return str(response)

            except Exception as e:
                # Restore original timeout
                self.client.timeout = original_timeout

                error_msg = str(e)
                print(f"Service call attempt {attempt + 1}/{max_retries} failed: {error_msg}")

                # If this is the last attempt, return error message
                if attempt == max_retries - 1:
                    # Return user-friendly error messages
                    if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                        return "Service error: Cannot connect to the code execution service. Please check if the service is running."
                    elif "404" in error_msg:
                        return f"Service error: Action '{action_id}' not found in provider '{provider}'."
                    elif "400" in error_msg:
                        return f"Service error: Invalid request for action '{action_id}'. Please check your parameters."
                    else:
                        return f"Service error: {error_msg}"

                # Wait before retrying (except for last attempt)
                if attempt < max_retries - 1:
                    time.sleep(1)

        # This should never be reached, but just in case
        return "Service error: All retry attempts failed"

    def step(self, action, *args, **kwargs):
        """Execute a step - simplified version without complex error handling"""
        try:
            # Parse action format
            if isinstance(action, str):
                if action.startswith(('execute_bash', 'finish', 'str_replace_editor', 'think')):
                    if not action.startswith('<function='):
                        action = '<function=' + action
                if action.rstrip().endswith('</parameter>') and not action.rstrip().endswith('</function>'):
                    action = action.rstrip() + '\n</function>'

            # Convert to function call format
            if isinstance(action, str):
                fncall = convert_non_fncall_messages_to_fncall_messages(
                    [{'role': 'assistant', 'content': action}], self.tools
                )[0]
            else:
                fncall = {'tool_calls': [{'function': action}]}

            if 'tool_calls' not in fncall:
                return True, NO_FNCALL_PROMPT

            # Extract function call details
            fncall = fncall['tool_calls'][0]['function']
            if isinstance(fncall['arguments'], str):
                arguments = json.loads(fncall['arguments'])
            else:
                arguments = fncall['arguments']
            name = fncall['name']

            # Handle different action types
            if name == 'finish':
                if 'answer' in arguments:
                    self.answer = arguments.get('answer', '')
                self._finish_called = True
                return True, "Task finished"
            elif name == 'think':
                self.think_history.append(arguments.get('content', ''))
                return True, AFTER_THINK_PROMPT
            else:
                # Call the service
                observation = self._call_service('code_act', name, arguments)
                observation = truncate_text(observation, max_lines=500, max_length=6_000, merge_repeat=True,
                                            merge_num=32, max_tokens=10_000)
                return True, observation

        except Exception as e:
            return False, f"Step failed: {str(e)}"

    @classmethod
    def from_env_str(cls, env_str: str, **kwargs):
        """Create environment from environment string"""
        if "@" in env_str:
            env_str = env_str.split("@", 1)[1]
        # Extract service URL from kwargs or use default
        service_url = os.getenv('LOC_IP_ADDRESS', 'http://localhost:8000')
        return cls(env_str=env_str, service_url=service_url, **kwargs)

    @property
    def finished(self) -> bool:
        """Check if task is finished"""
        return 1 if self._finish_called else 0

    def reward_f1(self, predicted: str, patch: str) -> float:
        """Return F1 score between predicted file list and files edited in a diff."""

        def clean(path: str) -> str:
            path = path.strip().lstrip('/')  # toss any number of leading "/"
            path = re.sub(r'^(?:\.?/)?(?:testbed/|workspace/)', '', path)
            return os.path.normpath(path)

        pred = {clean(p) for p in predicted.splitlines() if p.strip()}
        gt = {os.path.normpath(m) for m in re.findall(r'^diff --git a/([^ ]+)', patch, re.M)}

        if not pred and not gt: return 1.0
        if not pred or not gt:  return 0.0
        tp = len(pred & gt)

        return 0.0 if tp == 0 else 2 * tp / (len(pred) + len(gt))

    @property
    def reward(self):
        if self.answer is not None:
            patch = self.instance_info.get('patch')
            return self.reward_f1(self.answer, patch)
        """Get reward - always returns 0 (placeholder)"""
        return 0

    def release(self):
        """Release environment resources - no-op for simple environment"""
        pass

    def __del__(self):
        """Destructor - no-op for simple environment"""
        pass


@register_env
class FuncLocEnv(FileLocEnv):
    """
    Function localization env, read-only actions
    """
    env_str_prefix = "FuncLocEnv"

    def reward_f1(self, predicted: str, label_functions: str) -> float:
        """Return F1 score between predicted file list and files edited in a diff."""

        def clean(path: str) -> str:
            path = path.strip().lstrip('/')  # toss any number of leading "/"
            path = re.sub(r'^(?:\.?/)?(?:testbed/|workspace/)', '', path)
            return os.path.normpath(path)

        pred = {clean(p) for p in predicted.splitlines() if p.strip()}
        gt = {os.path.normpath(m) for m in label_functions}

        if not pred and not gt: return 1.0
        if not pred or not gt:  return 0.0
        tp = len(pred & gt)

        return 0.0 if tp == 0 else 2 * tp / (len(pred) + len(gt))

    @property
    def reward(self):
        if self.answer is not None:
            label_functions = self.instance_info.get('edited_functions')
            return self.reward_f1(self.answer, label_functions)
        """Get reward - always returns 0 (placeholder)"""
        return 0


from typing import TypedDict


class ChangeSimilarity(TypedDict):
    path: str
    pred_change: str
    oracle_change: str
    similarity: float


@register_env
class RepairEnv:
    """
    Support file editing (simulated edition) and read-only bash actions. No docker.
    """
    env_str_prefix = "RepairEnv"

    def __init__(self, env_str, service_url, **kwargs):
        self.session_id = str(uuid.uuid4())
        self.env_str = env_str
        self.instance_info = json.loads(self.env_str)
        self.service_url = service_url
        self.kwargs = kwargs
        self.instance_id = self.instance_info.get('instance_id', 'default')
        base_dir_base = os.getenv('BASE_DIR_PATH', './gym_data')
        self.base_dir = f"{base_dir_base}/{self.instance_id}"
        self.answer = None

        self._finish_called = False
        self.think_history = []

        self.file_cache = {}  # {path: {'original': str, 'current': str}}

        self.client = SimpleHttpClient(service_url)


    def _get_file_content(self, path: str) -> str:
        """Get file content, either from cache or by reading from server"""
        if path in self.file_cache:
            return self.file_cache[path]['current']

        cat_result = self._call_service('code_act', 'execute_bash', {'command': f'cat "{path}"'})
        if '[Exit code:' in cat_result:
            cat_result = "\n".join(cat_result.splitlines()[:-2])

        if ('No such file or directory' in cat_result or
                'cat:' in cat_result or
                'cannot access' in cat_result.lower()):
            raise FileNotFoundError(f"File {path} not found")

        self.file_cache[path] = {
            'original': cat_result,
            'current': cat_result
        }

        return cat_result

    def _str_replace_local(self, path: str, old_str: str, new_str: str) -> str:
        """Perform str_replace operation locally on cached content"""
        import re

        path = path.strip().lstrip('/')
        path = re.sub(r'^(?:\.?/)?(?:testbed/|workspace/)', '', path)
        path = os.path.normpath(path)

        try:
            current_content = self._get_file_content(path)
        except FileNotFoundError:
            return f"Error: File {path} not found"

        # Expand tabs
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs() if new_str is not None else ''
        current_content = current_content.expandtabs()

        # Find all occurrences using regex
        pattern = re.escape(old_str)
        matches = list(re.finditer(pattern, current_content))

        if not matches:
            return f"Error: No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."

        if len(matches) > 1:
            line_numbers = []
            for match in matches:
                line_num = current_content.count('\n', 0, match.start()) + 1
                line_numbers.append(line_num)
            line_numbers = sorted(set(line_numbers))
            return f"Error: Multiple occurrences of old_str `{old_str}` found in lines {line_numbers}. Please ensure it is unique."

        # Single occurrence - perform replacement
        match = matches[0]
        replacement_line = current_content.count('\n', 0, match.start()) + 1

        new_content = (
                current_content[:match.start()] +
                new_str +
                current_content[match.end():]
        )

        # Update cache
        self.file_cache[path]['current'] = new_content

        # Generate snippet around the change for output
        lines = new_content.split('\n')
        context_window = 10
        start_line = max(0, replacement_line - context_window - 1)
        end_line = min(len(lines), replacement_line + context_window + new_str.count('\n'))

        snippet_lines = lines[start_line:end_line]
        snippet = '\n'.join(f"{start_line + i + 1:4d} | {line}" for i, line in enumerate(snippet_lines))

        return f"The file {path} has been edited. Here is the result of running `cat -n` on a snippet of {path}:\n{snippet}\nReview the changes and make sure they are as expected. Edit the file again if necessary."

    def _view_file_local(self, path: str, start_line: int = None, end_line: int = None) -> str:
        """View file content from cache or server, or list directory contents"""
        import re
        import os

        path = path.strip()
        if path != '/testbed':
            path = path.lstrip('/')
            path = re.sub(r'^(?:\.?/)?(?:testbed/|workspace/)', '', path)
        path = os.path.normpath(path)

        # First, try to get file content (this will fail if it's a directory)
        try:
            content = self._get_file_content(path)
            lines = content.split('\n')

            # Handle line range
            if start_line is not None:
                start_line = max(1, start_line)
                start_idx = start_line - 1
            else:
                start_idx = 0
                start_line = 1

            if end_line is not None:
                end_idx = min(len(lines), end_line)
            else:
                end_idx = len(lines)

            # Show numbered lines
            snippet_lines = lines[start_idx:end_idx]
            snippet = '\n'.join(f"{start_line + i:4d} | {line}" for i, line in enumerate(snippet_lines))

            total_lines = len(lines)
            if start_idx > 0 or end_idx < total_lines:
                snippet = f"Here is the result of running `cat -n` on {path} (lines {start_line}-{end_idx}):\n{snippet}"
            else:
                snippet = f"Here is the result of running `cat -n` on {path}:\n{snippet}"

            return snippet

        except FileNotFoundError:
            # If file not found, try to list as directory
            return self._list_directory_local(path)

    def _list_directory_local(self, path: str) -> str:
        """List directory contents, including files from local cache"""

        # Try to get directory listing from server
        try:
            ls_result = self._call_service('code_act', 'execute_bash', {'command': f'ls -la "{path}"'})

            # Check if it's actually a directory listing
            if '[Exit code:' in ls_result:
                ls_result = "\n".join(ls_result.splitlines()[:-2])

            if ('No such file or directory' in ls_result or
                    'ls:' in ls_result or
                    'cannot access' in ls_result.lower()):

                # Check if we have any cached files in this directory
                cached_files = self._get_cached_files_in_dir(path)
                if cached_files:
                    # Directory doesn't exist on server but we have cached files
                    result = f""
                    for filename in sorted(cached_files):
                        result += f"{filename}\n"
                    return result
                else:
                    return f"Error: Directory {path} not found"

            # Parse the ls output to get existing files
            existing_files = set()
            lines = ls_result.strip().split('\n')

            # Skip the first line if it's a total line
            start_idx = 0
            if lines and lines[0].startswith('total '):
                start_idx = 1

            for line in lines[start_idx:]:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 9:  # Standard ls -la format
                        filename = ' '.join(parts[8:])  # Handle filenames with spaces
                        if filename not in ['.', '..']:
                            existing_files.add(filename)

            # Add cached files that are in this directory
            cached_files = self._get_cached_files_in_dir(path)

            # Combine and display
            all_files = existing_files.union(cached_files)

            if not all_files:
                return f"Directory {path} is empty"

            result = f""
            for filename in sorted(all_files):
                if filename in cached_files and filename not in existing_files:
                    result += f"{filename}\n"
                elif filename in cached_files:
                    result += f"{filename}\n"
                else:
                    result += f"{filename}\n"

            return result

        except Exception as e:
            # If bash command failed, check for cached files only
            cached_files = self._get_cached_files_in_dir(path)
            if cached_files:
                result = f""
                for filename in sorted(cached_files):
                    result += f"{filename}\n"
                return result
            else:
                return f"Error: Directory {path} not found"

    def _get_cached_files_in_dir(self, dir_path: str) -> set:
        """Get all cached files that are in the specified directory"""
        import os

        # Normalize directory path
        dir_path = os.path.normpath(dir_path)
        if dir_path == '.':
            dir_path = ''

        cached_files = set()
        for file_path in self.file_cache.keys():
            file_path = os.path.normpath(file_path)
            file_dir = os.path.dirname(file_path)

            # Check if this file is directly in the specified directory
            if file_dir == dir_path:
                filename = os.path.basename(file_path)
                cached_files.add(filename)

        return cached_files

    def _execute_bash_local(self, command: str) -> str:
        """Execute bash command with cached file awareness"""
        import shlex, re

        if not command.strip():
            return "Error: Empty command"

        try:
            cmd_parts = shlex.split(command)
        except ValueError:
            cmd_parts = command.split()

        if not cmd_parts:
            return "Error: Empty command"

        cmd_name = cmd_parts[0].split('/')[-1]  # Handle /bin/ls -> ls

        # Handle commands that should see cached files
        if cmd_name == 'ls':
            return self._handle_ls_command(cmd_parts)
        elif cmd_name == 'cat':
            return self._handle_cat_command(cmd_parts)
        elif cmd_name in ['head', 'tail', 'wc']:
            return self._handle_file_command(cmd_name, cmd_parts)
        elif cmd_name == 'find':
            return self._handle_find_command(command)
        elif cmd_name in ['grep', 'rg']:
            return self._handle_grep_command(command, cmd_parts)
        else:
            return self._call_service('code_act', 'execute_bash', {'command': command})

    def _clean_path(self, path: str) -> str:
        """Clean and normalize file path"""
        path = path.strip()
        if path != '/testbed':
            path = path.lstrip('/')
            path = re.sub(r'^(?:\.?/)?(?:testbed/|workspace/)', '', path)
        return path

    def _handle_ls_command(self, cmd_parts: list) -> str:
        path = '.'
        for part in cmd_parts[1:]:
            if not part.startswith('-'):
                path = part
                break
        return self._list_directory_local(self._clean_path(path))

    def _handle_cat_command(self, cmd_parts: list) -> str:
        if len(cmd_parts) < 2:
            return "cat: missing file operand"

        file_path = self._clean_path(cmd_parts[1])
        if file_path in self.file_cache:
            return self.file_cache[file_path]['current']

        return self._call_service('code_act', 'execute_bash', {'command': ' '.join(cmd_parts)})

    def _handle_file_command(self, cmd_name: str, cmd_parts: list) -> str:
        if len(cmd_parts) < 2:
            return self._call_service('code_act', 'execute_bash', {'command': ' '.join(cmd_parts)})

        file_path = self._clean_path(cmd_parts[-1])
        if file_path not in self.file_cache:
            return self._call_service('code_act', 'execute_bash', {'command': ' '.join(cmd_parts)})

        content = self.file_cache[file_path]['current']
        lines = content.splitlines()

        if cmd_name == 'head':
            num_lines = self._parse_line_count(cmd_parts, 10)
            return '\n'.join(lines[:num_lines])
        elif cmd_name == 'tail':
            num_lines = self._parse_line_count(cmd_parts, 10)
            return '\n'.join(lines[-num_lines:]) if lines else ''
        elif cmd_name == 'wc':
            return f"{len(lines)} {len(content.split())} {len(content)} {file_path}"

        return self._call_service('code_act', 'execute_bash', {'command': ' '.join(cmd_parts)})

    def _parse_line_count(self, cmd_parts: list, default: int) -> int:
        """Parse -n option for head/tail commands"""
        for i, part in enumerate(cmd_parts):
            if part == '-n' and i + 1 < len(cmd_parts):
                try:
                    return int(cmd_parts[i + 1])
                except ValueError:
                    pass
            elif part.startswith('-') and part[1:].isdigit():
                try:
                    return int(part[1:])
                except ValueError:
                    pass
        return default

    def _handle_find_command(self, command: str) -> str:
        """Handle find command to include cached files"""
        server_result = self._call_service('code_act', 'execute_bash', {'command': command})

        # Parse server result
        if '[Exit code:' in server_result:
            lines = server_result.splitlines()
            server_files, exit_info = lines[:-2], lines[-2:]
        else:
            server_files, exit_info = server_result.splitlines(), []

        # Add cached files (simple heuristic)
        all_files = set(server_files) | set(self.file_cache.keys())
        result = '\n'.join(sorted(all_files))

        return result + ('\n' + '\n'.join(exit_info) if exit_info else '')

    def _handle_grep_command(self, command: str, cmd_parts: list) -> str:
        """Handle grep command to search in cached files as well as server files"""
        import re

        # Extract pattern and options from grep command
        pattern = None
        files_to_search = []
        case_insensitive = False
        line_numbers = False

        i = 1  # Skip grep command itself
        while i < len(cmd_parts):
            arg = cmd_parts[i]
            if arg == '-i':
                case_insensitive = True
            elif arg == '-n':
                line_numbers = True
            elif arg.startswith('-'):
                # Skip other options
                pass
            elif pattern is None:
                pattern = arg
            else:
                # This is a file to search
                files_to_search.append(self._clean_path(arg))
            i += 1

        if not pattern:
            # No pattern found, run original command
            return self._call_service('code_act', 'execute_bash', {'command': command})

        all_results = []

        if files_to_search:
            # Specific files mentioned - handle each file
            server_files_to_search = []

            for file_path in files_to_search:
                if file_path in self.file_cache:
                    # File is cached - search local version only (it's the current state)
                    content = self.file_cache[file_path]['current']
                    lines = content.splitlines()

                    # Compile regex pattern
                    flags = re.IGNORECASE if case_insensitive else 0
                    try:
                        regex = re.compile(pattern, flags)
                    except re.error:
                        pattern_escaped = re.escape(pattern)
                        regex = re.compile(pattern_escaped, flags)

                    # Search for pattern in cached file
                    for line_num, line in enumerate(lines, 1):
                        if regex.search(line):
                            if line_numbers:
                                all_results.append(f"{file_path}:{line_num}:{line}")
                            else:
                                all_results.append(f"{file_path}:{line}")
                else:
                    # File not cached - add to server search list
                    server_files_to_search.append(file_path)

            # Search server files (only those not in cache)
            if server_files_to_search:
                grep_cmd = cmd_parts[0]
                options = [part for part in cmd_parts[1:] if part.startswith('-')]
                quoted_files = [f'"{f}"' for f in server_files_to_search]
                server_command = f"{grep_cmd} {' '.join(options)} {pattern} {' '.join(quoted_files)}"
                server_result = self._call_service('code_act', 'execute_bash', {'command': server_command})

                # Parse server result and add valid matches
                if '[Exit code:' in server_result:
                    lines = server_result.splitlines()
                    server_output = lines[:-2]
                else:
                    server_output = server_result.splitlines()

                for line in server_output:
                    if line and not line.startswith('grep:') and ':' in line:
                        all_results.append(line)

        else:
            # No specific files mentioned - search all files
            # First run original command on server
            server_result = self._call_service('code_act', 'execute_bash', {'command': command})

            # Parse server result
            if '[Exit code:' in server_result:
                lines = server_result.splitlines()
                server_output = lines[:-2]
            else:
                server_output = server_result.splitlines()

            # Collect files found on server (we'll exclude these from final results if they're cached)
            server_files_found = set()
            server_results_by_file = {}

            for line in server_output:
                if line and not line.startswith('grep:') and ':' in line:
                    filename = line.split(':', 1)[0]
                    server_files_found.add(filename)
                    if filename not in server_results_by_file:
                        server_results_by_file[filename] = []
                    server_results_by_file[filename].append(line)

            # Add server results for files NOT in cache
            for filename, results in server_results_by_file.items():
                if filename not in self.file_cache:
                    all_results.extend(results)

            # Search cached files - these override server versions
            for file_path in self.file_cache.keys():
                content = self.file_cache[file_path]['current']
                lines = content.splitlines()

                # Compile regex pattern
                flags = re.IGNORECASE if case_insensitive else 0
                try:
                    regex = re.compile(pattern, flags)
                except re.error:
                    pattern_escaped = re.escape(pattern)
                    regex = re.compile(pattern_escaped, flags)

                # Search for pattern in cached file
                for line_num, line in enumerate(lines, 1):
                    if regex.search(line):
                        if line_numbers:
                            all_results.append(f"{file_path}:{line_num}:{line}")
                        else:
                            all_results.append(f"{file_path}:{line}")

        return '\n'.join(all_results) if all_results else ''

    def generate_git_diff(self) -> str:
        """Generate git diff for all modified files"""
        if not self.file_cache:
            return "No files have been modified."

        import difflib

        all_diffs = []

        for path, content in self.file_cache.items():
            original = content['original']
            current = content['current']

            if original == current:
                continue  # No changes

            original_lines = original.splitlines()
            current_lines = current.splitlines()

            diff = difflib.unified_diff(
                original_lines,
                current_lines,
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm='',
                n=3
            )

            diff_lines = list(diff)
            if diff_lines:
                diff_text = '\n'.join(diff_lines)
                all_diffs.append(diff_text)

        if not all_diffs:
            return "No differences found."

        return '\n'.join(all_diffs)

    def ping(self):
        """Check if the service is responding"""
        try:
            response = self.client.get('ping')

            if isinstance(response, dict):
                content = response.get('content', response.get('result', ''))
            else:
                content = str(response)

            return content == 'pong' or 'pong' in content.lower()
        except Exception as e:
            print(f"Ping failed: {e}")
            return False

    def _call_service(self, provider: str, action_id: str, data: dict) -> str:
        """Call the new_main.py service with base_dir, retry 3 times with 120s timeout each"""
        max_retries = 3
        timeout = 120

        original_timeout = self.client.timeout

        for attempt in range(max_retries):
            try:
                self.client.timeout = timeout

                endpoint = f"api/v1/actions/{provider}"
                payload = {
                    "action_id": action_id,
                    "data": data,
                    "base_dir": self.base_dir
                }

                response = self.client.post(endpoint, payload)
                self.client.timeout = original_timeout
                if isinstance(response, dict):
                    return response.get('result', response.get('content', str(response)))
                else:
                    return str(response)

            except Exception as e:
                self.client.timeout = original_timeout

                error_msg = str(e)
                print(f"Service call attempt {attempt + 1}/{max_retries} failed: {error_msg}")

                if attempt == max_retries - 1:
                    if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                        return "Service error: Cannot connect to the code execution service. Please check if the service is running."
                    elif "404" in error_msg:
                        return f"Service error: Action '{action_id}' not found in provider '{provider}'."
                    elif "400" in error_msg:
                        return f"Service error: Invalid request for action '{action_id}'. Please check your parameters."
                    else:
                        return f"Service error: {error_msg}"

                if attempt < max_retries - 1:
                    time.sleep(1)

        return "Service error: All retry attempts failed"

    def step(self, action, *args, **kwargs):
        try:
            if isinstance(action, str):
                if action.startswith(('execute_bash', 'finish', 'str_replace_editor', 'think')):
                    if not action.startswith('<function='):
                        action = '<function=' + action
                if action.rstrip().endswith('</parameter>') and not action.rstrip().endswith('</function>'):
                    action = action.rstrip() + '\n</function>'

            if isinstance(action, str):
                fncall = convert_non_fncall_messages_to_fncall_messages(
                    [{'role': 'assistant', 'content': action}], self.tools
                )[0]
            else:
                fncall = {'tool_calls': [{'function': action}]}

            if 'tool_calls' not in fncall:
                return True, NO_FNCALL_PROMPT

            fncall = fncall['tool_calls'][0]['function']
            if isinstance(fncall['arguments'], str):
                arguments = json.loads(fncall['arguments'])
            else:
                arguments = fncall['arguments']
            name = fncall['name']

            if name == 'finish':
                self._finish_called = True
                return True, "Task finished"
            elif name == 'think':
                self.think_history.append(arguments.get('content', ''))
                return True, AFTER_THINK_PROMPT
            elif name == 'str_replace_editor':
                command = arguments.get('command', '')

                if command == 'str_replace':
                    path = arguments.get('path', '')
                    old_str = arguments.get('old_str', '')
                    new_str = arguments.get('new_str', '')

                    observation = self._str_replace_local(path, old_str, new_str)

                elif command == 'view':
                    path = arguments.get('path', '')
                    start_line = arguments.get('view_range', [None, None])[0]
                    end_line = arguments.get('view_range', [None, None])[1]

                    observation = self._view_file_local(path, start_line, end_line)

                elif command == 'create':
                    path = arguments.get('path', '')
                    file_text = arguments.get('file_text', '')

                    import re
                    path = path.strip().lstrip('/')
                    path = re.sub(r'^(?:\.?/)?(?:testbed/|workspace/)', '', path)
                    path = os.path.normpath(path)

                    self.file_cache[path] = {
                        'original': '',
                        'current': file_text
                    }

                    lines = file_text.split('\n')
                    snippet = '\n'.join(f"{i + 1:4d} | {line}" for i, line in enumerate(lines))
                    observation = f"File created successfully at: {path}\n{snippet}"

                else:
                    observation = self._call_service('code_act', name, arguments)

                observation = truncate_text(observation, max_lines=500, max_length=6_000, merge_repeat=True,
                                            merge_num=32, max_tokens=10_000)
                return True, observation
            elif name == 'execute_bash':
                # Handle bash commands with cached file awareness
                observation = self._execute_bash_local(arguments.get('command', ''))
                observation = truncate_text(observation, max_lines=500, max_length=6_000, merge_repeat=True,
                                            merge_num=32, max_tokens=10_000)
                return True, observation
            else:
                observation = self._call_service('code_act', name, arguments)
                observation = truncate_text(observation, max_lines=500, max_length=6_000, merge_repeat=True,
                                            merge_num=32, max_tokens=10_000)
                return True, observation

        except Exception as e:
            return False, f"Step failed: {str(e)}"

    @classmethod
    def from_env_str(cls, env_str: str, **kwargs):
        """Create environment from environment string"""
        if "@" in env_str:
            env_str = env_str.split("@", 1)[1]
        # Extract service URL from kwargs or use default
        service_url = os.getenv('LOC_IP_ADDRESS', 'http://localhost:8000')
        return cls(env_str=env_str, service_url=service_url, **kwargs)

    @property
    def finished(self) -> bool:
        """Check if task is finished"""
        return 1 if self._finish_called else 0

    def get_filelevel_diff(self, patch_text: str) -> dict[str, str]:
        from unidiff import PatchedFile, PatchSet
        from unidiff.errors import UnidiffParseError
        try:
            patch = PatchSet(patch_text)
        except UnidiffParseError:
            return {}
        except Exception as e:
            print(f"Unexpected unidiff parsing error: {str(e)}")
            return {}

        result = dict[str, str]()
        for patchfile in patch:
            patchfile: PatchedFile = patchfile
            if patchfile.is_binary_file:
                continue
            if patchfile.is_rename:
                source_file = patchfile.source_file
                target_file = patchfile.target_file
                if source_file.startswith("a/"):
                    source_file = source_file[2:]
                if target_file.startswith("b/"):
                    target_file = target_file[2:]
                header = f"rename from {source_file} to {target_file}"
                path = source_file
            else:
                header = ""
                path = patchfile.path
            body = "\n".join(str(hunk).strip() for hunk in patchfile)
            content = header + "\n" + body
            content = content.strip()
            result[path] = content
        return result

    def extract_changed_lines(self, diff_text: str) -> str:
        return diff_text
        if not diff_text:
            return ""
        changed_lines = []
        for line in diff_text.split('\n'):
            if line.startswith(('+', '-')) and not line.startswith(('+++', '---')):
                changed_lines.append(line)
        return '\n'.join(changed_lines)

    def compute_change_similarities(self, pred_patch: dict[str, str], oracle_patch: dict[str, str]):
        import difflib

        all_file_paths = set(oracle_patch.keys()).union(set(pred_patch.keys()))
        similarities = list[ChangeSimilarity]()
        # similarity_func = lambda a, b: fuzz.ratio(a, b) / 100.0

        for path in all_file_paths:
            pred_change = pred_patch.get(path, "")
            oracle_change = oracle_patch.get(path, "")
            if oracle_change == "" or pred_change == "":
                change_similarity = 0.0
            else:
                pred_changed_lines = self.extract_changed_lines(pred_change)
                oracle_changed_lines = self.extract_changed_lines(oracle_change)
                change_similarity = difflib.SequenceMatcher(
                    None,
                    pred_changed_lines,
                    oracle_changed_lines,
                    autojunk=False,
                ).ratio()

            similarities.append(
                ChangeSimilarity(
                    path=path,
                    pred_change=pred_change,
                    oracle_change=oracle_change,
                    similarity=change_similarity,
                )
            )
        return similarities

    def calculate_reward_unidiff(self, oracle_patches: list[str], pred_patches: list[str]) -> tuple[float, dict]:
        pred_patch_dict = dict[str, str]()
        oracle_patch_dict = dict[str, str]()

        for patch_text in oracle_patches:
            oracle_patch_dict.update(self.get_filelevel_diff(patch_text))

        for patch_text in pred_patches:
            pred_patch_dict.update(self.get_filelevel_diff(patch_text))

        is_code = lambda p: p.endswith(('.py', '.pyx', '.pxd'))
        oracle_patch_dict = {k: v for k, v in oracle_patch_dict.items() if is_code(k)}
        pred_patch_dict = {k: v for k, v in pred_patch_dict.items() if is_code(k)}

        similarities = self.compute_change_similarities(pred_patch_dict, oracle_patch_dict)
        if len(similarities) == 0:
            return 1.0, dict(similarities=[])
        reward = sum(map(lambda x: x["similarity"], similarities)) / len(similarities)
        return reward, dict(similarities=similarities)

    def calculate_swe_reward(self) -> tuple[float, dict]:
        oracle_patch = self.instance_info.get('patch', '')
        if not oracle_patch:
            return 0.0, {'error': 'No ground truth patch available'}
        predicted_diff = self.generate_git_diff()
        if not predicted_diff or predicted_diff == "No files have been modified.":
            if not oracle_patch.strip():
                return 1.0, {'message': 'Both predicted and oracle patches are empty'}
            else:
                return 0.0, {'error': 'No predicted changes but oracle patch exists'}
        oracle_patches = [oracle_patch] if oracle_patch.strip() else []
        pred_patches = [predicted_diff] if predicted_diff.strip() else []
        reward, metadata = self.calculate_reward_unidiff(oracle_patches, pred_patches)
        return reward, metadata

    @property
    def reward(self):
        if not self.finished:
            return 0
        if self.file_cache:
            reward, metadata = self.calculate_swe_reward()
            return reward
        return 0.0

    def release(self):
        pass

    def __del__(self):
        pass
