from typing import Any, Dict, Callable, List

class EventEmitter:
    """
    A simple event bus to decouple the pipeline execution from notification logic.
    This allows external services (like a FastAPI web server) to listen to 
    pipeline progress without modifying the core processing logic.
    """
    def __init__(self):
        # List of callback functions to notify on events
        self._subscribers: List[Callable[[str, Dict[str, Any]], None]] = []

    def subscribe(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Register a new subscriber."""
        self._subscribers.append(callback)

    def _notify(self, event_name: str, data: Dict[str, Any]) -> None:
        """Internal method to trigger all subscribers."""
        for callback in self._subscribers:
            try:
                callback(event_name, data)
            except Exception:
                # We catch exceptions here so that a failing subscriber 
                # (e.g., a broken network connection) does not crash the pipeline.
                pass

    def on_start(self, file_id: str) -> None:
        """Called when a new file processing task begins."""
        self._notify("on_start", {"file_id": file_id})

    def on_stage_change(self, file_id: str, stage_name: str, status: str) -> None:
        """Called when a pipeline stage (OCR, NER, or Norm) changes state."""
        self._notify("on_stage_change", {
            "file_id": file_id,
            "stage": stage_name,
            "status": status
        })

    def on_success(self, file_id: str, result_path: str) -> None:
        """Called when a file is successfully processed and saved."""
        self._notify("on_success", {
            "file_id": file_id, 
            "result_path": result_path
        })

    def on_error(self, file_id: str, error_message: str) -> None:
        """Called when an unrecoverable error occurs during processing."""
        self._notify("on_error", {
            "filele_id": file_id, 
            "error": error_message
        })

# Singleton instance to be used across the package
event_emitter = EventEmitter()
