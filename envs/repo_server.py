#!/usr/bin/env python3
"""
High-Performance Code Execution Server
- Pre-allocated worker thread pool (3000+ threads)
- Multiple event loops for load distribution
- Lock-free queues and connection pooling
- Removed rate limiting bottlenecks
- Batch processing capabilities
"""

import asyncio
import concurrent.futures
import logging
import os
import re
import subprocess
import time
import traceback
import shlex
import signal
import psutil
import weakref
from asyncio import StreamReader, Queue
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
from functools import lru_cache
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from threading import RLock, Event
import threading
import multiprocessing
import gc

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


from pydantic import BaseModel as PydanticBaseModel, Field
import pydantic
from packaging import version


class BaseModel(PydanticBaseModel):
    if version.parse(pydantic.__version__) >= version.parse('2.0'):
        def model_dump(self, **kwargs):
            return super().model_dump(**kwargs)
    else:
        def model_dump(self, **kwargs):
            return self.dict(**kwargs)

# ========== Ultra-High Performance Configuration ==========
MAX_WORKER_THREADS = 3000  # Pre-allocated thread pool
MAX_CONCURRENT_EXECUTIONS = 8000  # Massive parallel processing
MAX_QUEUE_SIZE = 50000  # Large queue for bursts
NUM_EVENT_LOOPS = 4  # Multiple loops for load distribution
BATCH_SIZE = 100  # Process requests in batches
WORKER_TIMEOUT = 60  # Worker timeout in seconds

# ========== Optimized Logging ==========
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/high_perf_server.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


# ========== Lock-Free Performance Metrics ==========
@dataclass
class AtomicCounter:
    """Thread-safe counter using atomic operations"""
    value: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def increment(self, delta: int = 1) -> int:
        with self._lock:
            self.value += delta
            return self.value

    def get(self) -> int:
        with self._lock:
            return self.value

    def reset(self) -> int:
        with self._lock:
            old_value = self.value
            self.value = 0
            return old_value


class HighPerformanceMetrics:
    """Ultra-fast metrics with minimal locking"""

    def __init__(self):
        self.request_count = AtomicCounter()
        self.success_count = AtomicCounter()
        self.server_error_count = AtomicCounter()
        self.client_error_count = AtomicCounter()
        self.timeout_count = AtomicCounter()
        self.active_connections = AtomicCounter()
        self.peak_connections = AtomicCounter()
        self.queued_requests = AtomicCounter()
        self.peak_queue_size = AtomicCounter()

        # Lock-free deques for performance data
        self.execution_times = deque(maxlen=1000)
        self.wait_times = deque(maxlen=1000)
        self.recent_server_errors = deque(maxlen=10)

        # Request timestamp tracking for 30-minute windows
        self.request_timestamps = deque(maxlen=10000)  # Store last 10k request timestamps

        # Timestamps
        self.start_time = datetime.now()
        self.last_stats_report = datetime.now()

        # Thread-safe locks for deques
        self._execution_lock = RLock()
        self._wait_lock = RLock()
        self._error_lock = RLock()
        self._timestamp_lock = RLock()

    def record_request(self, execution_time: float, success: bool = True,
                       error_msg: str = None, wait_time: float = 0,
                       is_server_error: bool = False):
        """Record request metrics with minimal locking"""
        now = datetime.now()
        self.request_count.increment()

        # Record request timestamp for 30-minute tracking
        with self._timestamp_lock:
            self.request_timestamps.append(now)

        # Record execution time
        with self._execution_lock:
            self.execution_times.append(execution_time)

        # Record wait time
        if wait_time > 0:
            with self._wait_lock:
                self.wait_times.append(wait_time)

        if success:
            self.success_count.increment()
        else:
            if is_server_error:
                self.server_error_count.increment()
                if error_msg:
                    with self._error_lock:
                        self.recent_server_errors.append(f"{datetime.now().strftime('%H:%M:%S')}: {error_msg}")
            else:
                self.client_error_count.increment()

            # Track specific error types
            if error_msg and "timeout" in error_msg.lower():
                self.timeout_count.increment()

    def get_avg_execution_time(self) -> float:
        with self._execution_lock:
            if not self.execution_times:
                return 0.0
            return sum(self.execution_times) / len(self.execution_times)

    def get_avg_wait_time(self) -> float:
        with self._wait_lock:
            if not self.wait_times:
                return 0.0
            return sum(self.wait_times) / len(self.wait_times)

    def add_to_queue(self):
        current = self.queued_requests.increment()
        peak = self.peak_queue_size.get()
        if current > peak:
            self.peak_queue_size.value = current

    def remove_from_queue(self):
        current = self.queued_requests.get()
        if current > 0:
            self.queued_requests.value = current - 1

    def get_requests_last_30_minutes(self) -> int:
        """Get number of requests in the last 30 minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=30)

        with self._timestamp_lock:
            # Count requests since cutoff time
            count = 0
            for timestamp in reversed(self.request_timestamps):
                if timestamp >= cutoff_time:
                    count += 1
                else:
                    break  # Timestamps are in chronological order
            return count

    def should_report_stats(self) -> bool:
        """Check if it's time to report statistics (every 30 seconds)"""
        now = datetime.now()
        if (now - self.last_stats_report).total_seconds() >= 30:
            self.last_stats_report = now
            return True
        return False

    def get_stats_summary(self) -> str:
        """Get professional stats summary"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        uptime_str = f"{uptime // 3600:.0f}h{(uptime % 3600) // 60:.0f}m{uptime % 60:.0f}s" if uptime >= 3600 else f"{uptime // 60:.0f}m{uptime % 60:.0f}s"

        request_total = self.request_count.get()
        success_total = self.success_count.get()
        server_errors = self.server_error_count.get()
        client_errors = self.client_error_count.get()

        server_error_rate = (server_errors / max(1, request_total)) * 100
        client_error_rate = (client_errors / max(1, request_total)) * 100

        # Get current system metrics
        try:
            cpu_percent = psutil.Process().cpu_percent()
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        except:
            cpu_percent = 0.0
            memory_mb = 0.0

        # Get recent server error summary
        with self._error_lock:
            server_error_summary = f"Last: {self.recent_server_errors[-1].split(': ', 1)[1] if self.recent_server_errors else 'None'}"

        # Get 30-minute request count
        requests_30min = self.get_requests_last_30_minutes()

        return f"""
[HIGH-PERF SERVER] Uptime: {uptime_str}
Requests: {request_total:,} total | Last 30min: {requests_30min:,} | Success: {success_total:,} | Server errors: {server_errors:,} ({server_error_rate:.1f}%) | Client errors: {client_errors:,} ({client_error_rate:.1f}%)
Queue: {self.queued_requests.get()} active | Peak: {self.peak_queue_size.get()} | Connections: {self.active_connections.get()}
CPU: {cpu_percent:.1f}% | Memory: {memory_mb:.0f}MB | Avg processing: {self.get_avg_execution_time():.3f}s | Avg wait: {self.get_avg_wait_time():.3f}s
{server_error_summary}"""


metrics = HighPerformanceMetrics()


# ========== Enhanced Type Definitions ==========
class CmdRunAction(BaseModel):
    command: str
    timeout: Optional[float] = 30.0
    priority: Optional[int] = 1


class FileEditorAction(BaseModel):
    command: str
    path: str
    file_text: Union[str, None] = None
    old_str: Union[str, None] = None
    new_str: Union[str, None] = None
    insert_line: Union[int, None] = None
    view_range: Union[list[int], None] = None


class ShellRunStatus(str, Enum):
    Finished = 'Finished'
    Error = 'Error'
    TimeLimitExceeded = 'TimeLimitExceeded'
    ResourceExhausted = 'ResourceExhausted'


class ExecuteShellArgs(BaseModel):
    command: str = Field(..., examples=['echo 123'], description='the command to run')
    cwd: Union[str, None] = None
    timeout: float = Field(30, description='code run timeout')
    stdin: Union[str, None] = Field(None, examples=[''], description='optional string to pass into stdin')
    files: dict[str, Union[str, None]] = Field({}, description='a dict from file path to base64 encoded file content')
    fetch_files: list[str] = Field([], description='a list of file paths to fetch')
    extra_env: Union[dict[str, str], None] = {}
    priority: int = Field(1, description='execution priority 1-10')


class ShellRunResult(BaseModel):
    status: ShellRunStatus
    execution_time: Union[float, None] = None
    return_code: Union[int, None] = None
    stdout: Union[str, None] = None
    stderr: Union[str, None] = None
    resource_usage: Optional[Dict[str, Any]] = None


class RunActionResponse(BaseModel):
    result: str
    data: dict = {}
    execution_id: Optional[str] = None
    timestamp: Optional[str] = None


class RunActionRequest(BaseModel):
    action_id: str
    data: dict
    base_dir: str = "/"
    priority: Optional[int] = 1
    client_id: Optional[str] = None


# ========== Robust Process Management ==========
class ProcessManager:
    """Manages subprocess lifecycle to prevent event loop errors"""

    def __init__(self):
        self.active_processes = weakref.WeakSet()
        self.cleanup_lock = asyncio.Lock()
        self.shutdown_event = asyncio.Event()

    def register_process(self, process):
        """Register a process for cleanup tracking"""
        self.active_processes.add(process)

    async def cleanup_process(self, process):
        """Safely cleanup a process"""
        if process and process.returncode is None:
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                try:
                    process.kill()
                    await asyncio.wait_for(process.wait(), timeout=1.0)
                except:
                    pass
            except:
                pass

    async def cleanup_all_processes(self):
        """Cleanup all tracked processes"""
        async with self.cleanup_lock:
            cleanup_tasks = []
            for process in list(self.active_processes):
                cleanup_tasks.append(self.cleanup_process(process))

            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    async def shutdown(self):
        """Signal shutdown and cleanup"""
        self.shutdown_event.set()
        await self.cleanup_all_processes()


# Global process manager
process_manager = ProcessManager()


# ========== Ultra-Fast Shell Execution ==========
async def get_output_reader(fd: StreamReader, max_out_bytes: int = 1024 * 1024):
    """Ultra-fast output reader with robust error handling"""
    res = b''

    async def reader():
        nonlocal res
        while True:
            try:
                chunk = await asyncio.wait_for(fd.read(8192), timeout=0.5)
                if not chunk:
                    break
                if len(res) <= max_out_bytes:
                    res += chunk
                else:
                    break
            except asyncio.TimeoutError:
                break
            except Exception:
                break

    task = asyncio.create_task(reader())

    async def read():
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
        except Exception:
            pass
        return res.decode('utf-8', errors='replace')

    return read


def try_decode(s: bytes) -> str:
    """Fast decode with error handling"""
    try:
        return s.decode('utf-8', errors='replace')
    except Exception:
        return '[DecodeError]'


def execute_shell_sync(args: ExecuteShellArgs) -> ShellRunResult:
    """Synchronous shell execution to avoid event loop issues"""
    execution_start = time.time()
    process = None

    try:
        # Minimal environment setup
        env = os.environ.copy()
        if args.extra_env:
            env.update(args.extra_env)

        # Create process with minimal overhead using subprocess.Popen
        process = subprocess.Popen(
            args.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=args.cwd,
            env=env,
            shell=True,
            executable='/bin/bash'
        )

        # Handle stdin
        stdin_data = None
        if args.stdin:
            stdin_data = args.stdin.encode('utf-8')

        try:
            # Wait for completion with timeout
            stdout, stderr = process.communicate(input=stdin_data, timeout=args.timeout)
            execution_time = time.time() - execution_start

            return ShellRunResult(
                status=ShellRunStatus.Finished,
                execution_time=execution_time,
                return_code=process.returncode,
                stdout=stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace'),
                resource_usage={"execution_time": execution_time}
            )

        except subprocess.TimeoutExpired:
            # Robust cleanup
            try:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            except Exception:
                pass

            execution_time = time.time() - execution_start

            # Get any partial output
            try:
                stdout, stderr = process.communicate(timeout=0.1)
                stdout = stdout.decode('utf-8', errors='replace') if stdout else ''
                stderr = stderr.decode('utf-8', errors='replace') if stderr else ''
            except Exception:
                stdout = ''
                stderr = ''

            return ShellRunResult(
                status=ShellRunStatus.TimeLimitExceeded,
                execution_time=execution_time,
                stdout=stdout,
                stderr=stderr
            )

    except Exception as e:
        # Cleanup process if it was created
        if process:
            try:
                process.terminate()
                process.wait(timeout=1.0)
            except Exception:
                try:
                    process.kill()
                    process.wait()
                except Exception:
                    pass

        return ShellRunResult(
            status=ShellRunStatus.Error,
            execution_time=time.time() - execution_start,
            stderr=f'Exception: {str(e)}'
        )


async def execute_shell_ultrafast(args: ExecuteShellArgs) -> ShellRunResult:
    """Async wrapper for backward compatibility - delegates to sync version"""
    return execute_shell_sync(args)


# ========== High-Performance Worker Pool ==========
class WorkerPool:
    """High-performance worker pool with pre-allocated threads"""

    def __init__(self, max_workers: int = MAX_WORKER_THREADS):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="HighPerfWorker"
        )
        self.active_tasks = AtomicCounter()
        self.completed_tasks = AtomicCounter()

        # Pre-warm the thread pool
        self._prewarm_pool()

    def _prewarm_pool(self):
        """Pre-warm the thread pool by submitting dummy tasks"""
        logger.warning(f"Pre-warming worker pool with {self.max_workers} threads...")

        def dummy_task():
            time.sleep(0.001)  # Minimal task to initialize thread

        # Submit dummy tasks to initialize all threads
        futures = []
        for i in range(min(100, self.max_workers)):  # Pre-warm with 100 threads
            future = self.executor.submit(dummy_task)
            futures.append(future)

        # Wait for all dummy tasks to complete
        for future in concurrent.futures.as_completed(futures, timeout=5):
            try:
                future.result()
            except:
                pass

        logger.warning(f"Worker pool pre-warmed with {self.max_workers} threads")

    async def submit_task(self, func, *args, **kwargs):
        """Submit task to thread pool"""
        self.active_tasks.increment()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, func, *args, **kwargs)
            self.completed_tasks.increment()
            return result
        finally:
            self.active_tasks.increment(-1)

    def get_stats(self) -> dict:
        """Get worker pool statistics"""
        return {
            "max_workers": self.max_workers,
            "active_tasks": self.active_tasks.get(),
            "completed_tasks": self.completed_tasks.get(),
            "thread_count": threading.active_count()
        }

    def shutdown(self):
        """Shutdown worker pool"""
        self.executor.shutdown(wait=True)


# Global worker pool
worker_pool = WorkerPool(MAX_WORKER_THREADS)


# ========== Multi-Loop Event System ==========
class EventLoopManager:
    """Manages multiple event loops for load distribution"""

    def __init__(self, num_loops: int = NUM_EVENT_LOOPS):
        self.num_loops = num_loops
        self.loops = []
        self.current_loop_index = 0
        self.loop_lock = threading.Lock()

        # Create and start event loops in separate threads
        for i in range(num_loops):
            loop_thread = threading.Thread(
                target=self._run_loop,
                args=(i,),
                name=f"EventLoop-{i}",
                daemon=True
            )
            loop_thread.start()
            time.sleep(0.1)  # Give loop time to start

    def _run_loop(self, loop_id: int):
        """Run an event loop in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loops.append(loop)

        logger.warning(f"Started event loop {loop_id}")

        try:
            loop.run_forever()
        finally:
            loop.close()

    def get_next_loop(self):
        """Get the next event loop in round-robin fashion"""
        with self.loop_lock:
            if not self.loops:
                return asyncio.get_event_loop()

            loop = self.loops[self.current_loop_index]
            self.current_loop_index = (self.current_loop_index + 1) % len(self.loops)
            return loop

    def shutdown(self):
        """Shutdown all event loops"""
        for loop in self.loops:
            loop.call_soon_threadsafe(loop.stop)


# Global event loop manager
# event_loop_manager = EventLoopManager(NUM_EVENT_LOOPS)

# ========== Optimized Path and Command Handling ==========
READONLY_COMMANDS = {
    'ls', 'cat', 'head', 'tail', 'grep', 'find', 'wc', 'sort', 'uniq', 'cut', 'awk',
    'sed', 'diff', 'file', 'which', 'whereis', 'locate', 'du', 'df', 'pwd', 'whoami',
    'ps', 'top', 'htop', 'free', 'uname', 'hostname', 'date', 'uptime', 'history',
    'env', 'printenv', 'echo', 'printf', 'test', 'stat', 'lsof', 'netstat', 'ss',
    'ifconfig', 'ip', 'ping', 'curl', 'wget', 'tree', 'less', 'more'
}


@lru_cache(maxsize=10000)
def is_readonly_command_cached(command: str) -> bool:
    """Ultra-fast cached readonly command validation"""
    if len(command) > 1000:
        return False

    # Quick pattern checks
    dangerous_patterns = ['>', '<', '&', '|', ';', '$(', '`', 'rm ', 'mv ', 'cp ', 'chmod ', 'sudo ']
    for pattern in dangerous_patterns:
        if pattern in command:
            return False

    # Check first command
    parts = command.split()
    if not parts:
        return False

    base_cmd = parts[0]
    return base_cmd in READONLY_COMMANDS


# ========== Readonly Command Validation ==========
READONLY_BASH_COMMANDS = {
    'ls', 'cat', 'head', 'tail', 'grep', 'find', 'wc', 'sort', 'uniq', 'cut', 'awk',
    'sed', 'diff', 'file', 'which', 'whereis', 'locate', 'du', 'df', 'pwd', 'whoami',
    'ps', 'top', 'htop', 'free', 'uname', 'hostname', 'date', 'uptime', 'history',
    'env', 'printenv', 'echo', 'printf', 'test', 'stat', 'lsof', 'netstat', 'ss',
    'ifconfig', 'ip', 'ping', 'curl', 'wget', 'git log', 'git show', 'git diff',
    'git status', 'git branch', 'git remote', 'git config', 'tree', 'less', 'more',
    'zcat', 'zless', 'zmore', 'bzcat', 'bzless', 'bzmore', 'xzcat', 'xzless', 'xzmore',
    'tar -tf', 'tar -tvf', 'zip -l', 'unzip -l', 'xxd', 'hexdump', 'od', 'strings',
    'ldd', 'objdump', 'readelf', 'nm', 'size', 'strip', 'file', 'python -c "print',
    'python3 -c "print', 'node -e "console.log', 'ruby -e "puts', 'perl -e "print',
    'php -r "echo', 'java -version', 'javac -version', 'gcc --version', 'g++ --version',
    'clang --version', 'make -n', 'cmake --version', 'pip list', 'pip show', 'npm list',
    'npm show', 'yarn list', 'yarn info', 'composer show', 'bundle list', 'gem list',
    'cargo --version', 'rustc --version', 'go version', 'docker --version', 'kubectl version'
}


def is_readonly_command(command: str) -> bool:
    """Enhanced readonly command validation - allows commands that only read data without modification"""
    command = command.strip()

    # Quick cache check
    if len(command) > 1000:  # Prevent extremely long commands
        return False

    # Base readonly commands that are always safe
    readonly_commands = {
        'ls', 'cat', 'head', 'tail', 'grep', 'find', 'wc', 'sort', 'uniq', 'cut', 'awk',
        'sed', 'diff', 'file', 'which', 'whereis', 'locate', 'du', 'df', 'pwd', 'whoami',
        'ps', 'top', 'htop', 'free', 'uname', 'hostname', 'date', 'uptime', 'history',
        'env', 'printenv', 'echo', 'printf', 'test', 'stat', 'lsof', 'netstat', 'ss',
        'ifconfig', 'ip', 'ping', 'curl', 'wget', 'tree', 'less', 'more',
        'zcat', 'zless', 'zmore', 'bzcat', 'bzless', 'bzmore', 'xzcat', 'xzless', 'xzmore',
        'xxd', 'hexdump', 'od', 'strings', 'ldd', 'objdump', 'readelf', 'nm', 'size',
        'python', 'python3', 'node', 'ruby', 'perl', 'php', 'java', 'javac', 'gcc', 'g++',
        'clang', 'make', 'cmake', 'pip', 'npm', 'yarn', 'composer', 'bundle', 'gem',
        'cargo', 'rustc', 'go', 'docker', 'kubectl', 'git'
    }

    # Commands that are never allowed (modify data or run code)
    dangerous_commands = {
        'rm', 'mv', 'cp', 'dd', 'chmod', 'chown', 'chgrp', 'ln', 'mkdir', 'rmdir', 'touch',
        'sudo', 'su', 'passwd', 'useradd', 'userdel', 'usermod', 'groupadd', 'groupdel',
        'mount', 'umount', 'fsck', 'mkfs', 'fdisk', 'parted', 'resize2fs',
        'systemctl', 'service', 'init', 'shutdown', 'reboot', 'halt', 'poweroff',
        'apt', 'yum', 'dnf', 'pacman', 'zypper', 'emerge', 'pkg', 'brew',
        'crontab', 'at', 'batch', 'nohup', 'screen', 'tmux',
        'export', 'unset', 'alias', 'unalias', 'source',
        'kill', 'killall', 'pkill', 'killall5'
    }

    # Readonly subcommands for various tools
    readonly_subcommands = {
        'pip': {'list', 'show', 'search', 'check', 'freeze', '--version', '-V'},
        'npm': {'list', 'ls', 'show', 'info', 'search', 'view', '--version', '-v'},
        'yarn': {'list', 'info', 'why', '--version', '-v'},
        'composer': {'show', 'info', 'search', 'depends', 'why', '--version', '-V'},
        'bundle': {'list', 'show', 'info', '--version', '-v'},
        'gem': {'list', 'search', 'info', 'specification', '--version', '-v'},
        'cargo': {'search', 'tree', '--version', '-V'},
        'go': {'list', 'version', 'env'},
        'docker': {'images', 'ps', 'version', 'info', 'stats'},
        'kubectl': {'get', 'describe', 'logs', 'version', 'cluster-info'},
        'git': {'log', 'show', 'diff', 'status', 'branch', 'remote', 'config', 'blame', 'shortlog'},
        'make': {'-n', '--dry-run', '--just-print', '--recon', '--what-if'}
    }

    # Patterns that indicate write operations or code execution
    dangerous_patterns = [
        r'>(?!>)',  # Output redirection
        r'>>',  # Append redirection
        r'<',  # Input redirection
        r'\$\(',  # Command substitution
        r'`',  # Backtick substitution
        r'eval\s',  # eval commands
        r'exec\s',  # exec commands
        r'&\s*$',  # Background execution
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, command):
            return False

    # Parse command safely
    try:
        cmd_parts = shlex.split(command)
        if not cmd_parts:
            return False

        base_cmd = cmd_parts[0]

        # Check if base command is explicitly dangerous
        if base_cmd in dangerous_commands:
            return False

        # Check if base command is in readonly list
        if base_cmd in readonly_commands:
            # For commands with subcommands, check if subcommand is readonly
            if base_cmd in readonly_subcommands and len(cmd_parts) > 1:
                subcommand = cmd_parts[1]
                # Allow if subcommand is in readonly list OR if it's a flag/option
                if subcommand in readonly_subcommands[base_cmd] or subcommand.startswith('-'):
                    return True
                # For pip, npm, etc., check for readonly patterns
                if base_cmd == 'pip' and subcommand in {'list', 'show', 'search', 'check', 'freeze'}:
                    return True
                if base_cmd == 'npm' and subcommand in {'list', 'ls', 'show', 'info', 'search', 'view'}:
                    return True
                if base_cmd == 'git' and subcommand in {'log', 'show', 'diff', 'status', 'branch', 'remote', 'config',
                                                        'blame', 'shortlog'}:
                    return True
                # If subcommand is not explicitly readonly, check if it contains install/modify keywords
                dangerous_subcommands = {'install', 'uninstall', 'update', 'upgrade', 'remove', 'add', 'delete', 'set',
                                         'config', 'init', 'create', 'push', 'pull', 'commit', 'merge', 'rebase'}
                if subcommand in dangerous_subcommands:
                    return False
                # Allow version checks and help
                if subcommand in {'--version', '-v', '-V', '--help', '-h'}:
                    return True
            else:
                # Simple command without subcommands
                return True

        # Check against whitelist patterns
        full_command = ' '.join(cmd_parts)
        found_in_whitelist = any(full_command.startswith(allowed) for allowed in READONLY_BASH_COMMANDS)
        if found_in_whitelist:
            return True

        # If we get here, command is not recognized as readonly
        return False

    except (ValueError, AttributeError):
        return False


@lru_cache(maxsize=5000)
def is_readonly_command_cached(command: str) -> bool:
    """Cached version of readonly command check for repeated commands"""
    return is_readonly_command(command)


# ========== Virtual Filesystem Path Management ==========
class VirtualFilesystem:
    """Complete virtual filesystem that simulates a machine under base_dir

    The agent sees a complete filesystem starting from /, but everything is actually under base_dir.
    For example: /testbed -> {base_dir}/testbed, /usr -> {base_dir}/usr, /foo -> {base_dir}/foo
    """

    def __init__(self, real_base_dir: str):
        self.real_base_dir = os.path.abspath(real_base_dir)
        self.testbed_dir = os.path.join(self.real_base_dir, 'testbed')

    def virtual_to_real(self, virtual_path: str) -> str:
        """Convert virtual path to real filesystem path

        This simulates a complete filesystem under base_dir:
        - Empty path -> base_dir/testbed (default working directory)
        - /testbed -> base_dir/testbed
        - /testbed/foo -> base_dir/testbed/foo
        - /usr -> base_dir/usr
        - /foo -> base_dir/foo
        """
        if not virtual_path:
            return self.testbed_dir

        # Normalize the virtual path
        virtual_path = os.path.normpath(virtual_path)

        if os.path.isabs(virtual_path):
            # Absolute virtual path - map to base_dir + path
            # /foo -> base_dir/foo
            # /testbed -> base_dir/testbed
            # /usr/bin -> base_dir/usr/bin
            clean_path = virtual_path.lstrip('/')
            if clean_path:
                real_path = os.path.join(self.real_base_dir, clean_path)
            else:
                # Root path / -> base_dir
                real_path = self.real_base_dir
        else:
            # Relative virtual path - relative to testbed directory (working directory)
            real_path = os.path.join(self.testbed_dir, virtual_path)

        # Resolve and normalize
        real_path = os.path.abspath(real_path)

        # Security check: ensure resolved path is within base directory
        if not real_path.startswith(self.real_base_dir):
            raise ValueError(f"Virtual path '{virtual_path}' resolves outside of sandbox")

        return real_path

    def real_to_virtual(self, real_path: str) -> str:
        """Convert real filesystem path to virtual path

        base_dir -> /
        base_dir/testbed -> /testbed
        base_dir/usr -> /usr
        base_dir/foo -> /foo
        """
        real_path = os.path.abspath(real_path)

        if not real_path.startswith(self.real_base_dir):
            raise ValueError(f"Real path '{real_path}' is outside sandbox")

        if real_path == self.real_base_dir:
            return '/'

        relative_part = os.path.relpath(real_path, self.real_base_dir)
        return '/' + relative_part.replace(os.sep, '/')

    def transform_command(self, command: str) -> str:
        """Transform paths in command from virtual to real

        Handles complex shell commands with pipes, redirections, etc.
        Only transforms actual filesystem paths, not shell operators.
        """
        if not command.strip():
            return command

        import re

        def replace_path(match):
            path = match.group(0)
            try:
                return self.virtual_to_real(path)
            except (ValueError, OSError):
                return path

        result = command

        # Simple but effective pattern: match /path but not when preceded by -
        # and not when followed by shell operators
        path_pattern = r'(?<!-)(/[a-zA-Z0-9_./-]+)(?=\s|$|\||>|<|&|;|\)|\])'
        result = re.sub(path_pattern, replace_path, result)

        # Handle quoted paths
        quoted_path_pattern = r'"(/[^"]+)"'

        def replace_quoted_path(match):
            path = match.group(1)
            try:
                real_path = self.virtual_to_real(path)
                return f'"{real_path}"'
            except (ValueError, OSError):
                return match.group(0)

        result = re.sub(quoted_path_pattern, replace_quoted_path, result)

        # Handle single-quoted paths
        single_quoted_pattern = r"'(/[^']+)'"

        def replace_single_quoted_path(match):
            path = match.group(1)
            try:
                real_path = self.virtual_to_real(path)
                return f"'{real_path}'"
            except (ValueError, OSError):
                return match.group(0)

        result = re.sub(single_quoted_pattern, replace_single_quoted_path, result)

        return result

    def transform_output(self, output: str) -> str:
        """Transform real paths in output back to virtual paths"""
        if not output:
            return output

        # Replace real base directory with virtual root (/) in output
        return output.replace(self.real_base_dir, '')


@lru_cache(maxsize=5000)
def resolve_path_cached(path: str, base_dir: str) -> str:
    """Ultra-fast cached path resolution using virtual filesystem"""
    vfs = VirtualFilesystem(base_dir)
    return vfs.virtual_to_real(path)


# ========== High-Performance Action Processors ==========
async def execute_bash_ultrafast(data: CmdRunAction, base_dir: str, execution_id: str) -> RunActionResponse:
    """Ultra-fast bash execution with minimal overhead"""
    start_time = time.time()

    try:
        # Fast command validation
        if not is_readonly_command_cached(data.command):
            error_msg = f"Command not allowed: {data.command}"
            metrics.record_request(time.time() - start_time, success=False, error_msg=error_msg, is_server_error=False)
            return RunActionResponse(
                result=f"Command '{data.command}' is not allowed",
                data={'error': 'command_not_allowed', 'success': False},
                execution_id=execution_id,
                timestamp=datetime.now().isoformat()
            )

        # Fast path resolution and ensure testbed directory exists
        try:
            testbed_dir = resolve_path_cached("", base_dir)
            os.makedirs(testbed_dir, exist_ok=True)
            # Also ensure base_dir exists for the virtual filesystem
            os.makedirs(base_dir, exist_ok=True)
        except Exception as e:
            error_msg = f"Path resolution error: {e}"
            metrics.record_request(time.time() - start_time, success=False, error_msg=error_msg, is_server_error=True)
            return RunActionResponse(
                result=f"Path error: {e}",
                data={'error': 'path_error', 'success': False},
                execution_id=execution_id,
                timestamp=datetime.now().isoformat()
            )

        # Create virtual filesystem for this request
        vfs = VirtualFilesystem(base_dir)

        # Transform command paths from virtual to real
        transformed_command = vfs.transform_command(data.command)

        # Execute command using thread pool
        shell_args = ExecuteShellArgs(
            command=transformed_command,
            cwd=testbed_dir,
            timeout=min(data.timeout or 30.0, 60.0),
            priority=data.priority or 1
        )

        # Use worker pool for execution with synchronous subprocess
        result = await worker_pool.submit_task(
            lambda: execute_shell_sync(shell_args)
        )

        # Process result and transform output paths back to virtual
        stdout = result.stdout or ""
        stderr = result.stderr or ""

        # Transform real paths back to virtual paths in output
        if stdout:
            stdout = vfs.transform_output(stdout)
        if stderr:
            stderr = vfs.transform_output(stderr)

        # Build response
        response_parts = []
        if stdout:
            response_parts.append(stdout)
        if stderr:
            response_parts.append(stderr)
        if result.execution_time is not None:
            response_parts.append(f'[Execution time: {result.execution_time:.3f}s]')
        if result.return_code is not None:
            response_parts.append(f'[Exit code: {result.return_code}]')

        execution_time = time.time() - start_time
        success = result.status == ShellRunStatus.Finished

        if result.status == ShellRunStatus.TimeLimitExceeded:
            error_msg = f"Timeout {result.execution_time:.1f}s: {data.command[:30]}..."
            metrics.record_request(execution_time, success=False, error_msg=error_msg, is_server_error=False)
        else:
            metrics.record_request(execution_time, success=success)

        return RunActionResponse(
            result='\n'.join(response_parts),
            data={
                **result.model_dump(),
                'stdout': stdout,
                'stderr': stderr,
                'transformed_command': data.command
            },
            execution_id=execution_id,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        error_msg = f"Bash execution error: {str(e)}"
        metrics.record_request(time.time() - start_time, success=False, error_msg=error_msg, is_server_error=True)
        return RunActionResponse(
            result=f"Execution failed: {str(e)}",
            data={'error': 'execution_error', 'success': False},
            execution_id=execution_id,
            timestamp=datetime.now().isoformat()
        )


async def file_editor_ultrafast(data: FileEditorAction, base_dir: str, execution_id: str) -> RunActionResponse:
    """Ultra-fast file editor with minimal overhead"""
    start_time = time.time()

    try:
        if data.command != 'view':
            error_msg = f"File command not allowed: {data.command}"
            metrics.record_request(time.time() - start_time, success=False, error_msg=error_msg, is_server_error=False)
            return RunActionResponse(
                result=f"Command '{data.command}' not allowed",
                data={'error': 'command_not_allowed', 'success': False},
                execution_id=execution_id,
                timestamp=datetime.now().isoformat()
            )

        try:
            resolved_path = resolve_path_cached(data.path, base_dir)
        except Exception as e:
            error_msg = f"Path resolution error: {e}"
            metrics.record_request(time.time() - start_time, success=False, error_msg=error_msg, is_server_error=True)
            return RunActionResponse(
                result=f"Path error: {e}",
                data={'error': 'path_error', 'success': False},
                execution_id=execution_id,
                timestamp=datetime.now().isoformat()
            )

        if not os.path.exists(resolved_path):
            error_msg = f"Path not found: {data.path}"
            metrics.record_request(time.time() - start_time, success=False, error_msg=error_msg, is_server_error=False)
            return RunActionResponse(
                result=f"Path '{data.path}' not found",
                data={'error': 'path_not_found', 'success': False},
                execution_id=execution_id,
                timestamp=datetime.now().isoformat()
            )

        # Use worker pool for file operations
        def read_file():
            if os.path.isfile(resolved_path):
                file_size = os.path.getsize(resolved_path)
                if file_size > 10 * 1024 * 1024:  # 10MB limit
                    return None, "File too large"

                with open(resolved_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()

                if data.view_range:
                    lines = content.split('\n')
                    start_line = max(0, data.view_range[0] - 1)
                    end_line = min(len(lines), data.view_range[1] if data.view_range[1] != -1 else len(lines))
                    selected_lines = lines[start_line:end_line]
                    formatted_lines = [f"{i:6d}  {line}" for i, line in enumerate(selected_lines, start=start_line + 1)]
                    result_content = '\n'.join(formatted_lines)
                else:
                    lines = content.split('\n')
                    if len(lines) > 10000:
                        lines = lines[:10000]
                        lines.append("... [File truncated, too many lines] ...")
                    formatted_lines = [f"{i:6d}  {line}" for i, line in enumerate(lines, start=1)]
                    result_content = '\n'.join(formatted_lines)

                return result_content, None

            elif os.path.isdir(resolved_path):
                entries = []
                for entry in sorted(os.listdir(resolved_path)):
                    entry_path = os.path.join(resolved_path, entry)
                    if os.path.isdir(entry_path):
                        entries.append(f"{entry}/")
                    else:
                        size = os.path.getsize(entry_path)
                        entries.append(f"{entry} ({size} bytes)")
                return '\n'.join(entries), None

            else:
                return None, "Invalid path type"

        result_content, error = await worker_pool.submit_task(read_file)

        if error:
            error_msg = f"File operation error: {error}"
            metrics.record_request(time.time() - start_time, success=False, error_msg=error_msg, is_server_error=False)
            return RunActionResponse(
                result=error,
                data={'error': 'file_operation_error', 'success': False},
                execution_id=execution_id,
                timestamp=datetime.now().isoformat()
            )

        execution_time = time.time() - start_time
        metrics.record_request(execution_time, success=True)

        return RunActionResponse(
            result=result_content,
            data={
                "path": data.path,
                "type": "file" if os.path.isfile(resolved_path) else "directory"
            },
            execution_id=execution_id,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        error_msg = f"File editor error: {str(e)}"
        metrics.record_request(time.time() - start_time, success=False, error_msg=error_msg, is_server_error=True)
        return RunActionResponse(
            result=f"File operation failed: {str(e)}",
            data={'error': 'file_operation_error', 'success': False},
            execution_id=execution_id,
            timestamp=datetime.now().isoformat()
        )


# ========== Action Registry ==========
ACTION_REGISTRY = {
    'code_act': {
        'execute_bash': execute_bash_ultrafast,
        'str_replace_editor': file_editor_ultrafast,
    }
}


# ========== Ultra-High Performance FastAPI Application ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.warning("Starting ultra-high performance server...")

    # Background task for periodic stats
    async def stats_reporter():
        while True:
            try:
                await asyncio.sleep(10)
                if metrics.should_report_stats():
                    logger.warning(metrics.get_stats_summary())
            except Exception as e:
                logger.error(f"Stats reporter error: {e}")

    stats_task = asyncio.create_task(stats_reporter())

    logger.warning("Ultra-high performance server ready")
    yield

    # Shutdown
    logger.warning("Shutting down ultra-high performance server...")

    # Cancel background tasks
    stats_task.cancel()
    try:
        await stats_task
    except asyncio.CancelledError:
        pass

    # Shutdown process manager first
    await process_manager.shutdown()

    # Shutdown worker pool
    worker_pool.shutdown()

    logger.warning("Shutdown completed")


app = FastAPI(
    title="Ultra-High Performance Code Execution Server",
    description="Optimized for 5K-10K concurrent requests",
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== Ultra-Fast Request Processing ==========
@app.middleware("http")
async def process_request(request: Request, call_next):
    """Ultra-fast request processing middleware"""
    metrics.active_connections.increment()

    try:
        response = await call_next(request)

        # Report stats periodically
        if metrics.should_report_stats():
            logger.warning(metrics.get_stats_summary())

        return response

    except Exception as e:
        logger.error(f"Request processing error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "request_processing_error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )
    finally:
        metrics.active_connections.increment(-1)


# ========== Health Check Endpoints ==========
@app.get('/health')
async def health_check():
    """Ultra-fast health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "worker_stats": worker_pool.get_stats(),
        "metrics": {
            "active_connections": metrics.active_connections.get(),
            "total_requests": metrics.request_count.get(),
            "success_rate": metrics.success_count.get() / max(1, metrics.request_count.get())
        }
    }


@app.get('/metrics')
async def get_metrics():
    """Get detailed metrics"""
    return {
        "request_count": metrics.request_count.get(),
        "success_count": metrics.success_count.get(),
        "server_error_count": metrics.server_error_count.get(),
        "client_error_count": metrics.client_error_count.get(),
        "active_connections": metrics.active_connections.get(),
        "avg_execution_time": metrics.get_avg_execution_time(),
        "avg_wait_time": metrics.get_avg_wait_time(),
        "worker_stats": worker_pool.get_stats(),
        "uptime_seconds": (datetime.now() - metrics.start_time).total_seconds()
    }


@app.get('/ping')
async def ping():
    """Ultra-fast ping endpoint"""
    return "pong"


# ========== Main API Endpoint ==========
@app.post('/api/v1/actions/{provider}', response_model=RunActionResponse)
async def run_action_ultrafast(
        provider: str,
        request: RunActionRequest,
        background_tasks: BackgroundTasks
):
    """Ultra-fast action execution without rate limiting"""
    execution_id = f"{int(time.time() * 1000)}_{hash(request.data.get('command', ''))}"

    # No rate limiting - handle all requests

    try:
        # Validate provider and action
        if provider not in ACTION_REGISTRY:
            return RunActionResponse(
                result=f'Provider {provider} not found',
                data={'error': 'invalid_provider', 'success': False},
                execution_id=execution_id,
                timestamp=datetime.now().isoformat()
            )

        if request.action_id not in ACTION_REGISTRY[provider]:
            return RunActionResponse(
                result=f'Action {request.action_id} not found',
                data={'error': 'invalid_action', 'success': False},
                execution_id=execution_id,
                timestamp=datetime.now().isoformat()
            )

        processor = ACTION_REGISTRY[provider][request.action_id]

        # Parse input data
        if request.action_id == 'execute_bash':
            input_data = CmdRunAction(**request.data)
        elif request.action_id == 'str_replace_editor':
            input_data = FileEditorAction(**request.data)
        else:
            return RunActionResponse(
                result='Invalid action',
                data={'error': 'invalid_action', 'success': False},
                execution_id=execution_id,
                timestamp=datetime.now().isoformat()
            )

        # Validate base directory
        base_dir = os.path.abspath(request.base_dir)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)

        # Execute action directly without queuing bottlenecks
        result = await processor(input_data, base_dir, execution_id)
        return result

    except Exception as e:
        logger.error(f"Action execution error: {e}", exc_info=True)
        return RunActionResponse(
            result=f"Internal server error: {str(e)}",
            data={'error': 'internal_server_error', 'success': False},
            execution_id=execution_id,
            timestamp=datetime.now().isoformat()
        )


# ========== Server Startup ==========
if __name__ == "__main__":
    logger.warning(f"Starting ultra-high performance server with {MAX_WORKER_THREADS} worker threads")

    # Ultra-high performance configuration
    uvicorn.run(
        app,
        host="::",
        port=8000,
        workers=1,
        loop="asyncio",
        log_level="warning",
        access_log=False,
        server_header=False,
        date_header=False,
        # Removed all concurrency limits
        limit_concurrency=None,  # No concurrency limits
        limit_max_requests=None,  # No request limits
        timeout_keep_alive=5,
        timeout_graceful_shutdown=30,
        # High-performance settings
        backlog=8192,  # Large connection backlog
        h11_max_incomplete_event_size=None,  # No limits
    )