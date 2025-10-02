
# Legacy Experiments

This directory contains legacy experiments for molecule and reaction generation using MCP servers. The experiments utilize Google Gemini models for content generation.

## Setting Up Environment

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running Molecule Generation Experiment

```bash
python3 mol_client.py
```


## Running Reaction Generation Experiment

```bash
python3 reaction_client.py
```


## Running AutoGen Examples

```bash
python3 autogen_mol_client.py --backend <backend_name>
```
```bash
python3 autogen_reaction_client.py --backend <backend_name>
```

Possible backends are `gemini`, `openai`, `livchat`, and `ollama`. Make sure to set the appropriate API keys in your environment variables. `livchat` is an internal LLNL service. Ollama requires a local model server.

`Gemini` backend requires the `GOOGLE_API_KEY` environment variable to be set and the
`OpenAI` backend requires the `OPENAI_API_KEY` environment variable to be set.
