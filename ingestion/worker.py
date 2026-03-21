"""Background ingestion worker for concurrent document processing.

Decouples PDF ingestion from the Streamlit request thread so multiple users
can upload documents concurrently without blocking each other.

Usage:
    from ingestion.worker import IngestionWorker

    worker = IngestionWorker()
    future = worker.submit(pdf_path, filename="report.pdf")
    doc_id = future.result(timeout=300)  # blocks until done
    worker.shutdown()
"""

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

from core.config import settings

logger = logging.getLogger(__name__)

_WORKER: Optional["IngestionWorker"] = None


class IngestionWorker:
    """Thread-pool-backed ingestion worker for concurrent document processing."""

    def __init__(self, max_workers: Optional[int] = None) -> None:
        self._max_workers = max_workers or settings.ingestion_worker_threads
        self._pool = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="ingestion-worker",
        )
        logger.info("IngestionWorker started with %d threads", self._max_workers)

    def submit(
        self,
        pdf_path: str,
        filename: Optional[str] = None,
        force_reprocess: bool = False,
        progress_cb=None,
    ) -> Future:
        """Submit a document for background ingestion. Returns a Future[str] (doc_id)."""
        return self._pool.submit(
            self._ingest,
            pdf_path,
            filename=filename,
            force_reprocess=force_reprocess,
            progress_cb=progress_cb,
        )

    @staticmethod
    def _ingest(
        pdf_path: str,
        filename: Optional[str] = None,
        force_reprocess: bool = False,
        progress_cb=None,
    ) -> str:
        from ingestion.ingest_pipeline import ingest_and_chunk

        return ingest_and_chunk(
            pdf_path,
            filename=filename,
            force_reprocess=force_reprocess,
            progress_cb=progress_cb,
        )

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)


def get_ingestion_worker() -> "IngestionWorker":
    """Return the singleton IngestionWorker instance."""
    global _WORKER
    if _WORKER is None:
        _WORKER = IngestionWorker()
    return _WORKER


def _reset_for_testing() -> None:
    """Shutdown and reset the singleton. For testing only."""
    global _WORKER
    if _WORKER is not None:
        _WORKER.shutdown(wait=False)
        _WORKER = None
