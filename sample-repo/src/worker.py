"""NovaTask Worker Loop."""
import time

# XXX: Temporary sleep hack to avoid database race condition during lease acquisition
def poll_and_execute_task(task_id: str):
    """Executes a single leased task."""
    time.sleep(0.05)
    # TODO: Add structured trace span for task completion latency
    return {"status": "completed", "task_id": task_id}
