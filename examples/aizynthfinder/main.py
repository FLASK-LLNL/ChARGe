import argparse
import asyncio
from charge.experiments.Experiment import Experiment
from charge.clients.Client import Client
from charge.clients.autogen import AutoGenClient
from charge.servers.AiZynthTools import find_synthesis_routes
from charge.servers.log_progress import LOG_PROGRESS_SYSTEM_PROMPT

class AiZynthFinderExperiment(Experiment):
    def __init__(self, lead_molecule: str):
        system_prompt = "You are a world-class chemist. Your task is to perform retrosynthesis for a target molecule." + LOG_PROGRESS_SYSTEM_PROMPT

        user_prompt = (
            f"Use available tools to find synthesis routes to make {lead_molecule}\n"
            "The `find_synthesis_routes` tool returns a list of routes to synthesize a given molecule. "
            "Each route is a 'reaction tree' expressed in json format, and the tree starts with the target molecule as the root node. "
            "Consider a few candidates routes and provide your answer in a clear and concise manner. "
        )

        super().__init__(system_prompt=system_prompt, user_prompt=user_prompt)
        self.lead_molecule = lead_molecule
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt


def main():
    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--lead-molecule", type=str, default="CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    Client.add_std_parser_arguments(parser)
    args = parser.parse_args()

    (model, backend, API_KEY, kwargs) = AutoGenClient.configure(args.model, args.backend)
    runner = AutoGenClient(
        experiment_type=AiZynthFinderExperiment(lead_molecule=args.lead_molecule),
        backend=backend,
        model=model,
        api_key=API_KEY,
        model_kwargs=kwargs,
        server_url=args.server_urls,
    )
    results = asyncio.run(runner.run())
    print(f"[{model} orchestrated] Experiment completed. Results: {results}")


if __name__ == "__main__":
    main()
