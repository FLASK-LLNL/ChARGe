import argparse
import asyncio
import os
from charge.tasks.Task import Task
from charge.clients.Client import Client
from charge.clients.autogen import AutoGenPool


class RetrosynthesisTask(Task):
    def __init__(self, lead_molecules: list[str]):
        mols = "\n".join(lead_molecules)
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
        mols = "\n".join(lead_molecules)
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
    parser.add_argument("--lead-molecules", nargs="+", default=["CC(=O)O[C@H](C)CCN"])
    parser.add_argument(
        "--retrosynthesis",
        action="store_true",
        default=False,
        help="Whether to perform a retrosynthesis task.",
    )

    Client.add_std_parser_arguments(parser)
    args = parser.parse_args()

    agent_pool = AutoGenPool(model=args.model, backend=args.backend)

    task = (
        RetrosynthesisTask(args.lead_molecules)
        if args.retrosynthesis
        else ForwardSynthesisTask(args.lead_molecules)
    )
    runner = agent_pool.create_agent(task=task)

    results = asyncio.run(runner.run())
    print(f"[{args.model} orchestrated] Task completed. Results: {results}")


if __name__ == "__main__":
    main()
