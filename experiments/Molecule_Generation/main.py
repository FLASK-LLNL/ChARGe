import argparse
from charge.clients.gemini import GeminiClient
from LMOExperiment import LMOExperiment as LeadMoleculeOptimization
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--user-prompt", type=str, default="Generate a drug-like molecule"
    )
    parser.add_argument(
        "--system-prompt", type=str, default="You are a helpful assistant."
    )
    args = parser.parse_args()

    myexperiment = LeadMoleculeOptimization(
        system_prompt=args.system_prompt, user_prompt=args.user_prompt
    )

    client_key = os.getenv("GOOGLE_API_KEY")
    runner = GeminiClient(api_key=client_key)
    runner.setup_mcp_servers(myexperiment)
    results = runner.run(myexperiment)
    print(results)

    while True:
        cont = input("Additional refinement? ...")
        results = runner.refine(cont)
        print(results)
