from functools import wraps
from typing import Callable, List

from loguru import logger


def capture_debug_messages(debug_list):
    def sink(message):
        record = message.record
        if record["level"].name == "DEBUG" or record["message"].startswith("[DEBUG]"):
            debug_list.append(record["message"])

    return sink


def debug_logging_decorator(func: Callable):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        debug_info: List[str] = []
        sink_id = None
        debug = kwargs.get("debug", False)

        if debug:
            # Add a custom sink to capture debug messages
            sink_id = logger.add(capture_debug_messages(debug_info))

        try:
            # Execute the wrapped function and store the result
            result = await func(*args, **kwargs)

            # Attach debug information if debug mode is enabled
            if debug:
                result["debug_info"] = debug_info

            return result

        finally:
            # Remove the custom sink after the request is processed
            if sink_id:
                logger.remove(sink_id)

    return wrapper
