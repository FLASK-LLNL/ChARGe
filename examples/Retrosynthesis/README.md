# Single Step Retrosynthesis

This experiment focuses on retrosynthesis, where the goal is to determine a synthetic route to a target molecule from simpler starting materials. 

## Running the Experiment
To run the experiment, first setup the server:

```bash
python3 ../../charge/servers/retrosynthesis_reaction_server.py
```

Then execute the `main.py` script. You can configure the model and server paths as needed.

```bash
python main.py --client <client_type> --backend <backend_type> --model <model_name_or_path> --server-path <server_path> --user-prompt "<user_prompt>"
```

And example user prompt would be: 
```
"Generate a new reaction SMARTS and reactants for the target molecule: CC(=O)OC1=CC=CC=C1C(=O)O"
```
**Note**: Make sure to enclose the user prompt in quotes.

The `--client` argument can be either `autogen` or `gemini`, depending on which client you want to use. Default is `autogen`.

The `--backend` is used for the `autogen` client and can be `ollama`, `livai`, `openai`, `gemini`, or `livchat`. The default is `openai`. This decides which model provider to use for the client. 

Some providers such as `ollama` and `openai` provide multiple models. You can specify the model using the `--model` argument. The model name should be compatible with the provider.

For example, to use `autogen` with `ollama` backend and `gpt-oss:latest` model, you would run:

```bash
python main.py --client autogen --backend ollama --model gpt-oss:latest --server-path reaction_server.py --user-prompt "Generate a new reaction SMARTS and reactants for the target molecule: CC(=O)OC1=CC=CC=C1C(=O)O" 
``` 


## Notes
- Ensure you have the required dependencies installed, including ChARGe and RDKit.
- The `reaction_server.py` file can be customized to additional functionalities as needed.
