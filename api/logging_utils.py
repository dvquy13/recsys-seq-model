import json
import uuid

from fastapi import Request
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, _StreamingResponse
from starlette.responses import Response


# Add middleware to assign request ID
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rec_id = str(uuid.uuid4())
        request.state.rec_id = rec_id

        # Contextualize logger with the request ID
        with logger.contextualize(rec_id=rec_id):
            response = await call_next(request)

        # Add rec_id to the output response
        if response.headers.get("Content-Type", "") == "application/json":
            if isinstance(response, _StreamingResponse):
                response_body = b""
                async for chunk in response.body_iterator:
                    response_body += chunk
                response_json = json.loads(response_body.decode("utf-8"))
                if "metadata" in response_json:
                    response_json["metadata"]["rec_id"] = rec_id
                else:
                    response_json["metadata"] = {"rec_id": rec_id}

                modified_response = json.dumps(response_json).encode("utf-8")
                response.headers["Content-Length"] = str(len(modified_response))
                return Response(
                    content=json.dumps(response_json).encode("utf-8"),
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )

        return response
