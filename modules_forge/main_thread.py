# This file is the main thread that handles all Gradio calls for major T2I / I2I processing
# Other Gradio calls (e.g. those from Extensions) are not influenced
# By using one single thread to process all major calls, model moving is significantly faster

import threading
import sys
import traceback
from collections import deque
from typing import Callable, Optional

lock = threading.Lock()
condition = threading.Condition(lock)

last_id: int = 0
waiting_queue: deque["Task"] = deque()
finished_tasks: dict[int, "Task"] = {}
last_exception: Optional[str] = None


def _oom_retry_enabled() -> bool:
    try:
        from modules.shared import opts

        return bool(getattr(opts, "forge_oom_retry_enabled", True))
    except Exception:
        return True


def _recover_from_oom():
    """Release every known model manager before the single retry."""
    from backend import memory_management

    try:
        from modules import sd_models

        sd_models.unload_model_weights()
    except Exception:
        memory_management.logger.debug("Forge model cleanup after OOM failed", exc_info=True)

    for module_name in ("modules.ui_ltx2_video", "modules.ui_ideogram"):
        try:
            module = sys.modules.get(module_name)
            if module is not None:
                module.unload()
        except Exception:
            memory_management.logger.debug("Native pipeline cleanup after OOM failed for %s", module_name, exc_info=True)

    try:
        memory_management.unload_all_models()
    except Exception:
        memory_management.logger.debug("Memory manager cleanup after OOM failed", exc_info=True)
    try:
        memory_management.soft_empty_cache(force=True)
    except Exception:
        memory_management.logger.debug("CUDA cache cleanup after OOM failed", exc_info=True)


class Task:
    def __init__(self, task_id, func, args, kwargs):
        self.task_id: int = task_id
        self.func: Callable = func
        self.args = args
        self.kwargs = kwargs
        self.result = None

    def work(self):
        global last_exception
        try:
            self.result = self.func(*self.args, **self.kwargs)
            last_exception = None
        except Exception as e:
            from backend.memory_management import is_oom, logger

            if not is_oom(e):
                if isinstance(e, ModuleNotFoundError) and "recognize" in str(e):
                    logger.error("Failed to recognize diffusion model... (check README for supported models)")
                else:
                    traceback.print_exc()
                last_exception = f"{type(e).__name__}: {e}"
                return

            if not _oom_retry_enabled():
                logger.error("Encountered Out of Memory; automatic retry is disabled")
                _recover_from_oom()
                last_exception = "OOM"
                return

            logger.warning("Encountered Out of Memory; clearing model state and retrying once")
            try:
                from modules import shared

                shared.state.textinfo = "Recovering from out of memory; retrying..."
            except Exception:
                pass
            _recover_from_oom()

            try:
                self.result = self.func(*self.args, **self.kwargs)
                last_exception = None
            except Exception as retry_error:
                if is_oom(retry_error):
                    logger.error("Out of Memory persisted after automatic retry")
                    _recover_from_oom()
                    last_exception = "OOM"
                else:
                    traceback.print_exc()
                    last_exception = f"{type(retry_error).__name__}: {retry_error}"


def loop():
    global waiting_queue, finished_tasks

    while True:
        with condition:
            while not waiting_queue:
                condition.wait(timeout=0.1)

            task = waiting_queue.popleft()

        task.work()

        with condition:
            finished_tasks[task.task_id] = task
            condition.notify_all()


def async_run(func, *args, **kwargs):
    global last_id

    with condition:
        last_id += 1
        task = Task(task_id=last_id, func=func, args=args, kwargs=kwargs)
        waiting_queue.append(task)
        condition.notify()

        return task.task_id


def run_and_wait_result(func, *args, **kwargs):
    task_id = async_run(func, *args, **kwargs)

    with condition:
        while task_id not in finished_tasks:
            condition.wait()

        task = finished_tasks.pop(task_id)
        return task.result
