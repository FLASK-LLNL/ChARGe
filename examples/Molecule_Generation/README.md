# Lead Molecule Generation

This directory contains an experiment setup for lead molecule generation using ChARGe.

There are two ways we can define the tools used for the experiment. Either directly in the `LMOExperiment` class or in a separate file like the `molecular_generation_server.py` file. 

The `LMOExperiment` also handles the boilerplate prompts that will be used in the experiment.

`molecular_generation_server.py` is an example MCP server that exposes RDKit functionalities as tools.

## Running the Experiment
To run the experiment, first setup the server:

```bash
python3 ../../charge/servers/molecular_generation_server.py
```

Then execute the `main.py` script. You can configure the model and server paths as needed.

```bash
python main.py --lead-molecule <lead_molecule> --client <client_type> --backend <backend_type> >--model <model_name_or_path> --server-path ../../charge/servers/molecular_generation_server.py
```

The `--client` argument can be either `autogen` or `gemini`, depending on which client you want to use. Default is `autogen`.

The `--backend` is used for the `autogen` client and can be `ollama`, `livai`, `openai`, `gemini`, or `livchat`. The default is `openai`. This decides which model provider to use for the client. 

Some providers such as `ollama` and `openai` provide multiple models. You can specify the model using the `--model` argument. The model name should be compatible with the provider.

The `--lead-molecule` argument is used to specify the lead molecule in SMILES format.

## Notes
- Ensure you have the required dependencies installed, including ChARGe and RDKit.
- Modify the prompts and tools as necessary to fit your specific use case.
- The `molecular_generation_server.py` file can be copied and customized to include additional RDKit functionalities as needed.
- When the `server_path` is not provided, the experiment class methods (if properly annotated) will be converted to tools automatically. This is disabled for this example.
