## Multi-server Support in Charge

This directory contains an example of a multi-server application using the ChARGe framework and the AutoGen library. The application allows users to interact with a language model that can utilize multiple molecular tools hosted on different servers with different transports (SSE and STDIO).

## SSE and STDIO Servers
To run the multi-server application, you need to start SSE server on a different process:
```bash
python3 sse_server_1.py --port 8000 --host http://127.0.0.1
```
and 
```bash
python3 sse_server_2.py --port 8001 --host http://127.0.0.1
```

The STDIO servers will be started automatically by the client.

## Running the Multi-Server Client
Once the SSE server is running, you can start the multi-server client by executing:
```bash
python3 main.py --server-urls <sse_server_1_url> <sse_server_2_url> --backend <backend> --model <model_name>
```

The `--server-urls` argument can take multiple server URLs (space-separated) that the client will connect to. Each URL must end with `/sse`.


