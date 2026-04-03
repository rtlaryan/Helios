from __future__ import annotations

import multiprocessing as mp
from multiprocessing.process import BaseProcess
from pathlib import Path
from queue import Empty
from typing import Any

import torch


def _writerWorker(queue, timeoutSeconds: float = 0.2) -> None:
    while True:
        try:
            task = queue.get(timeout=timeoutSeconds)
        except Empty:
            continue

        if task is None:
            queue.task_done()
            break

        pathString, payload = task
        path = Path(pathString)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        queue.task_done()


class AsyncWriterPool:
    def __init__(self, workerCount: int, queueSize: int, enabled: bool = True) -> None:
        self.enabled = enabled and workerCount > 0
        self.workerCount = workerCount
        self.queueSize = queueSize
        self.processes: list[BaseProcess] = []
        self.queue = None

        if not self.enabled:
            return

        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        self.context = mp.get_context(method)
        self.queue = self.context.JoinableQueue(maxsize=queueSize)

        for _ in range(workerCount):
            process = self.context.Process(target=_writerWorker, args=(self.queue,))
            process.start()
            self.processes.append(process)

    def submit(self, path: str | Path, payload: Any) -> None:
        if not self.enabled:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path)
            return

        assert self.queue is not None
        self.queue.put((str(path), payload), block=True)

    def flush(self) -> None:
        if not self.enabled:
            return

        assert self.queue is not None
        self.queue.join()

    def close(self) -> None:
        if not self.enabled:
            return

        assert self.queue is not None
        for _ in self.processes:
            self.queue.put(None, block=True)
        self.queue.join()

        for process in self.processes:
            process.join()

        self.processes.clear()
        self.queue.close()
        self.queue = None
