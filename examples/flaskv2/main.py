import argparse
import asyncio
import os
from charge.tasks.Task import Task
from charge.clients.Client import Client


class RetrosynthesisTask(Task):
    def __init__(self, lead_molecules: list[str]):
        mols = '\n'.join(lead_molecules)
        system_prompt = (
            "You are an expert chemist. Your task is to perform retrosynthesis for a target molecule. "
            "If any tool is used, show the original tool output, followed by your overall answer. "
            "Provide your answer in a clear and concise manner. "
        )
        user_prompt = (
            "Use available tools to find synthesis routes to make the following product molecules:\n"
            f"{mols}"
        )
        super().__init__(system_prompt=system_prompt, user_prompt=user_prompt)
        self.lead_molecules = lead_molecules
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt


class ForwardSynthesisTask(Task):
    def __init__(self, lead_molecules: list[str]):
        mols = '\n'.join(lead_molecules)
        system_prompt = (
            "You are an expert chemist. Your task is to perform forward synthesis for a set of reactant molecules. "
            "If any tool is used, show the original tool output, followed by your overall answer. "
            "Provide your answer in a clear and concise manner. "
        )
        user_prompt = (
            "Use available tools to predict forward synthesis from the following reactant molecules:\n"
            f"{mols}"
        )
        super().__init__(system_prompt=system_prompt, user_prompt=user_prompt)
        self.lead_molecules = lead_molecules
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt


def main():
    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--lead-molecules", nargs='+', default=["CC(=O)O[C@H](C)CCN"])
    parser.add_argument("--retrosynthesis", action='store_true', default=False, help="Whether to perform a retrosynthesis task.")
    parser.add_argument("--client", type=str, default="autogen", choices=["autogen", "gemini"])
    Client.add_std_parser_arguments(parser)
    args = parser.parse_args()

    if args.client == "gemini":
        from charge.clients.gemini import GeminiClient
        client_key = os.getenv("GOOGLE_API_KEY")
        assert client_key is not None, "GOOGLE_API_KEY must be set in environment"
        runner = GeminiClient(task=mytask, api_key=client_key)
    elif args.client == "autogen":
        from charge.clients.autogen import AutoGenClient
        task = RetrosynthesisTask(args.lead_molecules) if args.retrosynthesis else ForwardSynthesisTask(args.lead_molecules)
        (model, backend, API_KEY, kwargs) = AutoGenClient.configure(args.model, args.backend)
        runner = AutoGenClient(
            task=task,
            model=model,
            backend=backend,
            api_key=API_KEY,
            model_kwargs=kwargs,
            server_url=args.server_urls,
        )
        results = asyncio.run(runner.run())
        print(f"[{model} orchestrated] Task completed. Results: {results}")


if __name__ == "__main__":
    main()
