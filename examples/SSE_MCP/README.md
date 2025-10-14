## SSE Server with ChARGe

ChARGe can be used with persistent SSE MCP servers through the client interface as well. This example shows how to set up an SSE MCP server with ChARGe.

### Setup

You must first run the server on a different process. You can do this by running the following command in a terminal:

```bash
python server.py
```

This will start an SSE MCP server locally. Note the address and port where the server is running (by default, it will be `http://127.0.0.1:8000`).

### Client Usage
You can then use the ChARGe client to connect to this server and perform operations. 
Run the following script to see how to use the client with the SSE MCP server:

```bash
python main.py --backend <backend> --model <model> --server-url <server_url>/sse
```

**Note:** The `--server-url` should point to the address where your SSE MCP server is running, appended with `/sse`.
