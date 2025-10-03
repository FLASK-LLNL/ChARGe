## ChARGe Chat
This directory contains an example of a multi-turn chat application using the ChARGe framework and the AutoGen library. The application allows users to interact with a language model that can utilize various molecular tools to assist in drug discovery tasks.

## SSE Server
To run the multi-turn chat application, you need to start the SSE server that hosts the molecular tools. You can do this by executing the following command in your terminal:
```bash
python3 mol_tool_server.py
```

## Running the Chat Client
Once the SSE server is running, you can start the chat client by executing:
```bash
python3 main.py --server_url <server_url>
```

