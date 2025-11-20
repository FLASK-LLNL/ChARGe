import argparse
import asyncio
from charge.experiments.Experiment import Experiment
from charge.clients.Client import Client
from charge.clients.autogen import AutoGenClient

parser = argparse.ArgumentParser()

# Add standard CLI arguments
Client.add_std_parser_arguments(parser)

class ChargeMultiServerExperiment(Experiment):
    def __init__(
        self,
    ):
        system_prompt = (
            "You are a world-class chemist. Your task is to generate unique molecules "
            "based on the lead molecule provided by the user. The generated molecules "
            "should be chemically valid and diverse, exploring different chemical spaces "
            "while maintaining some structural similarity to the lead molecule. "
            "Provide the final answer in a clear and concise manner."
        )

        user_prompt = (
            "Generate a unique molecule based on the lead molecule provided. "
            " The lead molecule is CCO. Use SMILES format for the molecules. "
            "Ensure the generated molecule is chemically valid and unique,"
            " using the tools provided. Check the price of the generated molecule "
            "using the molecule pricing tool, and get a cheap molecule. "
        )
        super().__init__(system_prompt=system_prompt, user_prompt=user_prompt)
        print("ChargeMultiServerExperiment initialized with the provided prompts.")

        self.system_prompt = system_prompt
        self.user_prompt = user_prompt


if __name__ == "__main__":

    args = parser.parse_args()
    Client.enable_cmd_history_and_shell_integration(args.history)
    server_urls = args.server_urls
    assert server_urls is not None, "Server URLs must be provided"
    for url in server_urls:
        assert url.endswith("/sse"), f"Server URL {url} must end with /sse"

    exit
    myexperiment = ChargeMultiServerExperiment()

    (model, backend, API_KEY, kwargs) = AutoGenClient.configure(
        args.model, args.backend
    )

    server_path_1 = "stdio_server_1.py"
    server_path_2 = "stdio_server_2.py"

    runner = AutoGenClient(
        experiment_type=myexperiment,
        backend=backend,
        model=model,
        api_key=API_KEY,
        model_kwargs=kwargs,
        server_path=[server_path_1, server_path_2],
        server_url=server_urls,
    )

    results = asyncio.run(runner.run())

    print(f"Experiment completed. Results: {results}")
