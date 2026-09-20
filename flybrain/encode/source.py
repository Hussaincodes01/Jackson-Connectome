"""Frame acquisition. Webcam, screen and synthetic all emit the same thing,
so the input source is a runtime switch rather than an architectural choice."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float
    stale: bool = False


class FrameSource(ABC):
    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self._last: Frame | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @abstractmethod
    def read(self) -> Frame: ...

    def close(self) -> None:
        return None

    def _finish(self, image: np.ndarray) -> Frame:
        image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
        self._last = Frame(image=image, timestamp=time.perf_counter(), stale=False)
        return self._last

    def _hold_last(self) -> Frame:
        """Source failed. Hold the previous frame and flag it; the simulation
        keeps its own clock regardless."""
        if self._last is None:
            return Frame(np.zeros((self.height, self.width), np.float32), time.perf_counter(), True)
        return Frame(self._last.image, time.perf_counter(), True)


class SyntheticSource(FrameSource):
    def __init__(self, generator: Callable[[int], np.ndarray], height: int, width: int) -> None:
        super().__init__(height, width)
        self.generator = generator
        self.index = 0

    def read(self) -> Frame:
        image = self.generator(self.index)
        self.index += 1
        return self._finish(image)


class WebcamSource(FrameSource):
    def __init__(self, index: int = 0, height: int = 120, width: int = 160, fps: int = 30) -> None:
        super().__init__(height, width)
        import cv2

        self._cv2 = cv2
        self.capture = cv2.VideoCapture(index)
        self.capture.set(cv2.CAP_PROP_FPS, fps)

    def read(self) -> Frame:
        ok, raw = self.capture.read()
        if not ok:
            return self._hold_last()
        gray = self._cv2.cvtColor(raw, self._cv2.COLOR_BGR2GRAY)
        small = self._cv2.resize(gray, (self.width, self.height))
        return self._finish(small.astype(np.float32) / 255.0)

    def close(self) -> None:
        self.capture.release()


class ScreenSource(FrameSource):
    def __init__(self, monitor: int = 1, height: int = 120, width: int = 160) -> None:
        super().__init__(height, width)
        import cv2
        import mss

        self._cv2 = cv2
        self._sct = mss.mss()
        self._monitor = self._sct.monitors[monitor]

    def read(self) -> Frame:
        try:
            shot = np.asarray(self._sct.grab(self._monitor))
        except Exception:
            return self._hold_last()
        gray = self._cv2.cvtColor(shot[:, :, :3], self._cv2.COLOR_BGR2GRAY)
        small = self._cv2.resize(gray, (self.width, self.height))
        return self._finish(small.astype(np.float32) / 255.0)

    def close(self) -> None:
        self._sct.close()
