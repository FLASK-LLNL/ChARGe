import argparse
import asyncio
import time
from charge.tasks.LMOTask import LMOTask as LeadMoleculeOptimization
import os
from charge.clients.Client import Client
import charge.utils.helper_funcs as helper_funcs
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument("--lead-molecule", type=str, default="CC(=O)O[C@H](C)CCN")
parser.add_argument(
    "--client", type=str, default="autogen", choices=["autogen", "gemini"]
)

parser.add_argument(
    "--server-path",
    type=str,
    default=None,
    help="Path to an existing MCP server script",
)
parser.add_argument(
    "--json_file",
    type=str,
    default="known_molecules.json",
    help="Path to the JSON file containing known molecules.",
)

parser.add_argument(
    "--max_iterations",
    type=int,
    default=5,
    help="Maximum number of iterations to run the task.",
)

# Add standard CLI arguments
Client.add_std_parser_arguments(parser)

args = parser.parse_args()


if __name__ == "__main__":

    mytask = LeadMoleculeOptimization(lead_molecule=args.lead_molecule)
    server_path = args.server_path
    server_urls = args.server_urls
    assert server_urls is not None, "Server URLs must be provided"
    for url in server_urls:
        assert url.endswith("/sse"), f"Server URL {url} must end with /sse"

    if args.client == "gemini":
        from charge.clients.gemini import GeminiClient

        client_key = os.getenv("GOOGLE_API_KEY")
        assert client_key is not None, "GOOGLE_API_KEY must be set in environment"
        runner = GeminiClient(task=mytask, api_key=client_key)
    elif args.client == "autogen":
        from charge.clients.autogen import AutoGenClient

        (model, backend, API_KEY, kwargs) = AutoGenClient.configure(
            args.model, args.backend
        )

        runner = AutoGenClient(
            task=mytask,
            model=model,
            backend=backend,
            api_key=API_KEY,
            model_kwargs=kwargs,
            server_path=server_path,
            server_url=server_urls,
        )

    mol_file_path = args.json_file

    lead_molecule_smiles = args.lead_molecule
    logger.info(f"Starting task with lead molecule: {lead_molecule_smiles}")
    parent_id = 0
    node_id = 0
    lead_molecule_data = helper_funcs.post_process_smiles(
        smiles=lead_molecule_smiles, parent_id=parent_id - 1, node_id=node_id
    )

    # Start the db with the lead molecule
    helper_funcs.save_list_to_json_file(
        data=[lead_molecule_data], file_path=mol_file_path
    )
    logger.info(f"Storing found molecules in {mol_file_path}")

    # Run the task in a loop
    new_molecules = helper_funcs.get_list_from_json_file(
        file_path=mol_file_path
    )  # Start with known molecules

    mol_data = [lead_molecule_data]

    max_iterations = args.max_iterations
    iteration = 0
    while True:
        try:
            iteration += 1
            if iteration >= max_iterations:
                logger.info("Reached maximum iterations. Ending task.")
                break
            results = asyncio.run(runner.run())
            results = results.as_list()  # Convert to list of strings
            logger.info(f"New molecules generated: {results}")
            processed_mol = helper_funcs.post_process_smiles(
                smiles=results[0], parent_id=parent_id, node_id=node_id
            )
            canonical_smiles = processed_mol["smiles"]
            if (
                canonical_smiles not in new_molecules
                and canonical_smiles != "Invalid SMILES"
            ):
                new_molecules.append(processed_mol["smiles"])
                mol_data.append(processed_mol)
                helper_funcs.save_list_to_json_file(
                    data=mol_data, file_path="known_molecules.json"
                )
                logger.info(f"New molecule added: {canonical_smiles}")

                node_id += 1
            else:
                logger.info(f"Duplicate molecule found: {canonical_smiles}")

            logger.info(f"Total unique molecules so far: {new_molecules}")

            # TODO: Update the prompt with the new lead molecule or maybe the
            # the best performing molecule.

            # reset the runner for the next iteration
            runner.reset()

            time.sleep(2)  # To avoid rate limiting

            if len(new_molecules) >= 5:  # Collect 5 new molecules then stop
                break
        except KeyboardInterrupt:
            logger.info("Task interrupted by user.")
            break
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            logger.info("Restarting the task...")
            runner.reset()
            continue

    logger.info(f"Task completed. Results: {new_molecules}")
