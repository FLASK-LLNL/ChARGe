import argparse

parser = argparse.ArgumentParser(description="Run a ChARGe MCP Server")
parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
parser.add_argument(
    "--host", type=str, default="http://127.0.0.1", help="Host to run the server on"
)
args = parser.parse_args()

