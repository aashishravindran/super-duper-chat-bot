"""
AWS Lambda entry point.
Mangum adapts the FastAPI ASGI app to the Lambda/API Gateway event format.

For true token-level streaming, upgrade to Lambda Response Streaming
(InvokeMode: RESPONSE_STREAM) — the FastAPI SSE endpoints already support it.
"""
from mangum import Mangum
from src.api import app

handler = Mangum(app, lifespan="off")
