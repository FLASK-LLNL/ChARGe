import argparse
import asyncio
from charge.Experiment import Experiment
import httpx
from charge.clients.Client import Client
from charge.clients.autogen import AutoGenClient
import os
from typing import Optional

parser = argparse.ArgumentParser()
parser.add_argument("--lead-molecule", type=str, default="CC(=O)O[C@H](C)CCN")
parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8000/sse")

# Add standard CLI arguments
Client.add_std_parser_arguments(parser)

class UniqueMoleculeExperiment(Experiment):
    def __init__(self, lead_molecule: str):
        system_prompt = (
            "You are a world-class chemist. Your task is to generate unique molecules "
            "based on the lead molecule provided by the user. The generated molecules "
            "should be chemically valid and diverse, exploring different chemical spaces "
            "while maintaining some structural similarity to the lead molecule. "
            "Provide the final answer in a clear and concise manner."
        )

        user_prompt = (
            f"Generate 3 unique molecules based on the lead molecule {lead_molecule}. "
            +"Use tools to verify the molecule is valid. "
            + "Return only the SMILES strings in a Python list format."
        )

        super().__init__(system_prompt=system_prompt, user_prompt=user_prompt)
        print("UniqueMoleculeExperiment initialized with the provided prompts.")
        self.lead_molecule = lead_molecule
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt


if __name__ == "__main__":

    args = parser.parse_args()
    lead_molecule = args.lead_molecule
    server_url = args.server_url
    assert server_url is not None, "Server URL must be provided"
    assert server_url.endswith("/sse"), "Server URL must end with /sse"

    myexperiment = UniqueMoleculeExperiment(lead_molecule=lead_molecule)

    (model, backend, API_KEY, kwargs) = AutoGenClient.configure(args.model, args.backend)

    runner = AutoGenClient(
        experiment_type=myexperiment,
        backend=backend,
        model=model,
        api_key=API_KEY,
        model_kwargs=kwargs,
        server_url=server_url,
    )

    results = asyncio.run(runner.run())

    print(f"[{model} orchestrated] Experiment completed. Results: {results}")
