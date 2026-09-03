"""Worker-safe HDF5 handle manager for multi-process PyTorch DataLoaders."""

from __future__ import annotations

import os
from typing import ClassVar

import h5py


class WorkerHDF5Manager:
    """Manages process-local HDF5 file descriptors for high-throughput streaming.

    In PyTorch multi-process DataLoaders (`num_workers > 0`), sharing HDF5 file
    handles across process boundaries causes race conditions and memory corruption.
    `WorkerHDF5Manager` ensures each worker process lazily opens and caches its own
    process-local HDF5 handles tied to `os.getpid()`.

    Args:
        rdcc_nbytes: Size of chunk cache in bytes (default: 4MB = 4 * 1024 * 1024).
        rdcc_nslots: Number of chunk slots in cache prime number (default: 521).
    """

    _instance: ClassVar[WorkerHDF5Manager | None] = None

    def __init__(
        self,
        rdcc_nbytes: int = 4 * 1024 * 1024,
        rdcc_nslots: int = 521,
    ) -> None:
        self.rdcc_nbytes = rdcc_nbytes
        self.rdcc_nslots = rdcc_nslots
        self._pid: int = os.getpid()
        self._handles: dict[str, h5py.File] = {}

    @classmethod
    def get_instance(
        cls,
        rdcc_nbytes: int = 4 * 1024 * 1024,
        rdcc_nslots: int = 521,
    ) -> WorkerHDF5Manager:
        """Retrieve or initialize the process-local singleton manager instance.

        Args:
            rdcc_nbytes: Chunk cache size in bytes.
            rdcc_nslots: Number of chunk slots in cache.

        Returns:
            The singleton WorkerHDF5Manager instance for current process.
        """
        if cls._instance is None:
            cls._instance = cls(rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance and close all open handles."""
        if cls._instance is not None:
            cls._instance.close_all()
            cls._instance = None

    def _check_pid(self) -> None:
        """Detect worker process fork and reset handle cache if PID changed."""
        current_pid = os.getpid()
        if current_pid != self._pid:
            # Fork detected: discard stale parent handles without closing them
            # (closing parent handles in child could invalidate the parent descriptors)
            self._handles = {}
            self._pid = current_pid

    def get_handle(self, file_path: str, mode: str = "r") -> h5py.File:
        """Get or open an HDF5 file handle for the specified path in current worker.

        Args:
            file_path: Absolute or relative path to the HDF5 file.
            mode: File open mode (default: 'r').

        Returns:
            Open h5py.File handle with configured chunk cache and SWMR settings.
        """
        self._check_pid()
        handle = self._handles.get(file_path)
        if handle is not None:
            try:
                if handle.id.valid:
                    return handle
            except Exception:
                pass
            del self._handles[file_path]

        try:
            handle = h5py.File(
                file_path,
                mode=mode,
                libver="latest",
                swmr=(mode == "r"),
                rdcc_nbytes=self.rdcc_nbytes,
                rdcc_nslots=self.rdcc_nslots,
            )
        except Exception:
            # Fallback if SWMR or libver='latest' fails on specific format
            handle = h5py.File(
                file_path,
                mode=mode,
                rdcc_nbytes=self.rdcc_nbytes,
                rdcc_nslots=self.rdcc_nslots,
            )

        self._handles[file_path] = handle
        return handle

    def close_handle(self, file_path: str) -> None:
        """Close and remove a specific HDF5 handle.

        Args:
            file_path: Path to the HDF5 file.
        """
        self._check_pid()
        handle = self._handles.pop(file_path, None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    def close_all(self) -> None:
        """Close all open HDF5 file handles in the current process."""
        self._check_pid()
        for handle in list(self._handles.values()):
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()

    def __enter__(self) -> WorkerHDF5Manager:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close_all()

    def __del__(self) -> None:
        try:
            self.close_all()
        except Exception:
            pass
