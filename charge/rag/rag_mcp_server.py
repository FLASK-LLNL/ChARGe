import argparse
import json
from enum import StrEnum

import datasets
from datasets import load_dataset
from loguru import logger
import itertools

datasets.disable_caching()

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LlamaForCausalLM,
        PreTrainedTokenizer,
    )
    import torch
    from trl import apply_chat_template

    HAS_FLASKV2 = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_FLASKV2 = False
    logger.warning(
        "Please install the flask support packages to use this module."
        "Install it with: pip install charge[flask]",
    )

from charge.rag import SmilesEmbedder, FaissDataRetriever
from charge.rag.rag_tokenizers import ChemformerTokenizer
from charge.rag.prompts import ReactionDataPrompt
from charge.servers.FLASKv2_reactions import (
    format_rxn_prompt,
    PRODUCT_KEYS,
    REAGENT_KEYS,
)

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


from fastmcp import FastMCP
from starlette.responses import JSONResponse

mcp = FastMCP("RAG Server", json_response=True)


class Role(StrEnum):
    PRODUCTS = "products"
    REACTANTS = "reactants"


def convert_dictoflists_to_listofdicts(dofl_dict: dict) -> dict:
    """Convert a dictionary of lists to a list of dicts.
    Args:
        dofl_dict (dict): Dictionary of lists. Each value should be the same length.
    Returns:
        list[dict]: List of dicts. Each dict represents a reaction.
    """
    return [dict(zip(dofl_dict.keys(), row)) for row in zip(*dofl_dict.values())]


@mcp.tool()
def search_similar_reactions(data: dict, forward: bool, k_r: int) -> dict:
    """Add similar reactions to a reaction data dictionary.

    This function retrieves similar reactions from a reaction database and
    populates the following fields within ``data``:
    - ``data['similar']``: A list of list of dictionaries. The inner lists are of length
      `k_r`, and each dictionary is of a retrieved similar reaction.

    Args:
        data (dict): data dictionary that will be augmented.
            Adds the `similar` key:
                similar (list[dict]): List populated with `k_r` retrieved similar
                    reactions.
            Additional keys may also be present but are not required or modified.
        forward (bool): Whether prediction is for forward synthesis
            (True) or retrosynthesis (False).
        k_r (int): Number of similar reactions to retrieve.

    Returns:
        dict: The updated reaction data dictionary with new fields populated.
    """
    return search_similar_reactions_impl(data, forward, k_r)


def search_similar_reactions_impl(data: dict, forward: bool, k_r: int) -> dict:
    """Add similar reactions to the reaction data dictionary.
    See search_similar_reactions for details
    """
    input_role = Role.REACTANTS if forward else Role.PRODUCTS
    return search_similar_reactions_by_role(data, input_role, k_r)


def search_similar_reactions_by_role(data: dict, role: str, k_r: int) -> dict:
    """Add similar reactions to a reaction data dictionary.

    This function retrieves similar reactions from a reaction database and
    populates the following fields within ``data``:
    - ``data['similar']``: A list of length `k_r` of dictionaries. Each dictionary is of a retrieved similar reaction.

    Args:
        data (dict): data dictionary that will be augmented.
            Example: (for forward synthesis) {'reactants': ['CCC']}
            Adds the `similar` key:
                similar (list[dict]): List populated with `k_r` retrieved similar
                    reactions.
            Additional keys may also be present but are not required or modified.
        role (str): The role to search similar reactions for. Must be one of 'reactants' or 'products'.
        k_r (int): Number of similar reactions to retrieve.

    Returns:
        dict: The updated reaction data dictionary with new fields populated.
    """
    input_role = role
    logger.debug(f"search_similar_reactions_by_role: data is {data}")
    assert isinstance(data[input_role], list), f"data[{input_role}] must be a list"
    assert not isinstance(
        data[input_role][0], list
    ), f"data[{input_role}] must not be a list of lists. This function is for a single reaction"
    data = search_similar_reactions_by_role_bulk([data], role, k_r)[0]
    return data


def search_similar_reactions_by_role_bulk(
    data: list[dict], role: str, k_r: int
) -> list[dict]:
    """See search_similar_reactions_by_role. data is a list of the dicts matching that function's data"""
    retriever = retro_retriever if role == Role.PRODUCTS else forward_retriever
    assert role in [
        Role.REACTANTS,
        Role.PRODUCTS,
    ], f"role must be one of 'reactants' or 'products', but got {role}"
    input_role = role
    query_smiles = [".".join(d[input_role]) for d in data]
    query_emb = embedder.embed_smiles(query_smiles)
    similar_dist, similar_idx, similar = retriever.search_similar(query_emb, k=k_r)
    for d, sim in zip(data, similar):
        d["similar"] = sim
    return data


def predict_reaction_internal(
    molecules: dict[str, list[str]], forward: bool, k_r: int = 3
) -> list[str]:
    """

    Args:
        molecules (dict[str, list[str]]): dict of list of SMILES strings for one side of a reaction.
          Example: {'reactants': ['CCC']}
        forward (bool): True if for forward synthesis, False for retrosynthesis
        k_r (int): Number of predictions. Defaults to 3.

    Returns: List of top-3 predictions of the product for forward synthesis, or reactants for retrosynthesis

    """
    # Copied from FLASKv2_reactions.py, b/c we both use global variables. Also, this accepts batches of molecules
    # Maybe that's dangerous? What if the model
    if not HAS_FLASKV2:
        raise ImportError(
            "Please install the [flask] optional packages to use this module."
        )
    if isinstance(molecules, list):
        logger.info(
            f"molecules should be a rxn dict of list of SMILES, not a list of smiles. Converting to a rxn dict"
        )
        molecules = (
            {Role.REACTANTS: molecules} if forward else {Role.PRODUCTS: molecules}
        )
    model = forward_expert_model if forward else retro_expert_model
    with torch.inference_mode():
        format_rxn_prompt(molecules, forward=forward)
        prompt = apply_chat_template(molecules, tokenizer=tokenizer)
        del molecules["prompt"]
        inputs = tokenizer(prompt["prompt"], return_tensors="pt", padding="longest").to(
            "cuda"
        )
        prompt_length = inputs["input_ids"].size(1)
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            num_return_sequences=k_r,
            # do_sample=True,
            num_beams=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,  # enable KV cache
        )
        processed_outputs = [
            tokenizer.decode(out[prompt_length:], skip_special_tokens=True)
            for out in outputs
        ]
    logger.debug(f'Model input: {prompt["prompt"]}')
    processed_outs = "\n".join(processed_outputs)
    logger.debug(f"Model output: {processed_outs}")
    return processed_outputs


def predict_reactions_internal(
    reaction_halves: list[dict[str, list[str]]], forward: bool, k_r: int = 3
) -> list[list[str]]:
    """
    Predict multiple reactions

    Args:
        reaction_halves (list[dict[str, list[str]]]): list of rxn dicts of SMILES strings. One dict per reaction. Dict
            values are lists of SMILES strings for each reaction role.
            Example: [{'reactants': ['CCC']}]
        forward (bool): True if for forward synthesis, False for retrosynthesis
        k_r (int): Number of predictions per reaction. Defaults to 3.

    Returns:
        List of product or reagent SMILES lists.
    """
    # Should there be some bounds on the max size of the list, or batched generation?
    if not HAS_FLASKV2:
        raise ImportError(
            "Please install the [flask] optional packages to use this module."
        )
    if isinstance(reaction_halves[0], list):
        logger.info(
            f"Reaction halves should be a list of rxn dicts, not a list of rxn dicts of list of SMILES. "
            f"Converting to a list of rxn dicts"
        )
        reaction_halves = [
            {Role.REACTANTS: half} if forward else {Role.PRODUCTS: half}
            for half in reaction_halves
        ]
    model = forward_expert_model if forward else retro_expert_model
    with torch.inference_mode():
        prompts = []
        for half in reaction_halves:
            format_rxn_prompt(half, forward=forward)
            prompts.append(apply_chat_template(half, tokenizer=tokenizer)["prompt"])
            del half["prompt"]

        inputs = tokenizer(prompts, return_tensors="pt", padding="longest").to("cuda")
        prompt_length = inputs["input_ids"].size(1)
        n_beams = 3
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            num_return_sequences=k_r,
            # do_sample=True,
            num_beams=n_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,  # enable KV cache
        )
    processed_outputs = []
    for i in range(0, len(outputs), k_r):
        processed_outputs.append(
            [
                tokenizer.decode(out[prompt_length:], skip_special_tokens=True)
                for out in outputs[i : i + k_r]
            ]
        )
    logger.debug(f"Model input: {prompts}")
    processed_outs = "\n".join(
        [item for sublist in processed_outputs for item in sublist]
    )
    logger.debug(f"Model output: {processed_outs}")
    return processed_outputs


def add_expert_predictions_on_similar_data(data: dict) -> dict:
    """
    Adds "expert predictions on similar data" to the reaction data dictionary. Will be a list of the same length as
    `data['similar']`.
    Args:
        data (dict): reaction data that contains information about similar reactions.
            A data retriever must be used beforehand to populate `data['similar']` and `data['similar idx']`.
    """
    expert_predictions = [
        _[0] for _ in predict_reactions_internal(data["similar"], forward=True, k_r=1)
    ]
    # Taking only the top-1 expert prediction per similar reaction
    data["expert predictions on similar data"] = expert_predictions
    return data


@mcp.tool()
def get_related_reaction_info(data: dict, forward=True, k_r: int = 3) -> dict:
    """
    Augment a single reaction data dictionary with additional information.
    This function will add an expert prediction and similar reactions with their expert predictions to the
    data dictionary. This function calls `search_similar_reactions` and `add_expert_predictions_on_similar_data`.
    Args:
        data (dict): reaction data dictionary.
          Example: (for forward synthesis) {'reactants': ['CCC'], }
          Example: (for retrosynthesis) {'products': ['CCC'], }
        forward (bool): Whether the prediction is for forward synthesis (True) or retrosynthesis (False). Defaults to True.
        k_r (int): Number of similar reactions to retrieve. Defaults to 3.

    Returns:
        dict: The updated reaction data dictionary with additional fields.
            Adds these fields:
            "expert_prediction" (str): Expert prediction for the reaction.
            "similar" (list[dict]): List populated with retrieved similar reactions, and their expert predictions
    """
    logger.debug(f"data is {data}, forward is {forward}")

    # Just in case: Remove extra fields to doubly make sure there's no data leakage
    to_delete = PRODUCT_KEYS if forward else REAGENT_KEYS
    for key in to_delete:
        if key in data:
            del data[key]

    search_similar_reactions_impl(data, forward=forward, k_r=k_r)
    add_expert_predictions_on_similar_data(data)

    data["expert_prediction"] = predict_reaction_internal(
        molecules=data, forward=forward, k_r=1
    )[0]
    sim_expert_preds = data["expert predictions on similar data"]
    assert len(sim_expert_preds) == len(
        data["similar"]
    ), f"Expected {len(data['similar'])} expert predictions, but got {len(sim_expert_preds)}"
    for i, sim_pred in enumerate(sim_expert_preds):
        data["similar"][i]["expert_prediction"] = sim_pred
    del data["expert predictions on similar data"]
    return data


embedder: SmilesEmbedder = None
forward_retriever: FaissDataRetriever = None
retro_retriever: FaissDataRetriever = None
forward_expert_model = None
retro_expert_model = None
tokenizer = None


def main():
    global embedder, forward_retriever, retro_retriever, forward_expert_model, retro_expert_model, tokenizer
    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database-path",
        type=str,
        help="Path to database file (JSON) for similarity search",
    )
    parser.add_argument("--embedder-path", type=str, help="Path to embedding model")
    parser.add_argument(
        "--embedder-vocab-path",
        type=str,
        help="Path to embedding model vocab json file",
    )
    parser.add_argument(
        "--forward-embedding-path",
        type=str,
        help="Path to embedding file (NPY) corresponding to similarity search",
    )
    parser.add_argument(
        "--retro-embedding-path",
        type=str,
        help="Path to embedding file (NPY) corresponding to similarity search",
    )
    parser.add_argument(
        "--all-mols-database-path",
        type=str,
        help="Path to database file (txt) of one SMILES per line for similarity search",
    )
    parser.add_argument(
        "--all-mols-embedding-path",
        type=str,
        help="Path to embedding file (NPY) corresponding to similarity search",
    )
    parser.add_argument(
        "--forward-expert-model-path", type=str, help="Path to forward expert model"
    )
    parser.add_argument(
        "--retro-expert-model-path", type=str, help="Path to retro expert model"
    )
    # parser.add_argument('--k_e', type=int, help='Number of expert predictions (per input data) to show in prompt')
    parser.add_argument(
        "--k_r",
        type=int,
        help="Number of similar reactions (per input data) to retrieve",
        default=3,
    )
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k} = {v}", flush=True)

    # Init RAG components
    embedder = SmilesEmbedder(
        model_path=args.embedder_path,
        tokenizer=ChemformerTokenizer(
            vocab_path=args.embedder_vocab_path,
        ),
    )
    forward_retriever = FaissDataRetriever(
        data_path=args.database_path,
        emb_path=args.forward_embedding_path,
    )
    retro_retriever = FaissDataRetriever(
        data_path=args.database_path,
        emb_path=args.retro_embedding_path,
    )
    # retriever = forward_retriever if forward else retro_retriever

    tokenizer = AutoTokenizer.from_pretrained(
        args.forward_expert_model_path or args.retro_expert_model_path,
        padding_side="left",
    )
    tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
    if args.forward_expert_model_path is not None:
        forward_expert_model = AutoModelForCausalLM.from_pretrained(
            args.forward_expert_model_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
    if args.retro_expert_model_path is not None:
        retro_expert_model = AutoModelForCausalLM.from_pretrained(
            args.retro_expert_model_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )

    if forward_expert_model is not None:

        @mcp.tool()
        def get_expert_forward_synthesis_predictions(reactants: list[str]) -> list[str]:
            """
            Given a set of reactants and possibly reagent molecules, predict the likely product molecule(s).

            Args:
                reactants (list[str]): A list of smiles of reactant and reagent molecules in SMILES representation, for
                    one reaction.
                    Example: ['CCO(=O)', 'CCO'], which is acetic acid and ethanol which produces ethyl acetate.
            Returns:
                list[str]: A list of product predictions, each of which is  json string listing the predicted product molecule(s) in SMILES.
            """
            logger.debug("Calling `predict_reaction_products`")
            logger.debug(f"Input reactions: {reactants}")
            res = predict_reaction_internal(molecules=reactants, forward=True)
            logger.debug(f"Output predictions: {res}")
            return res

    if retro_expert_model is not None:

        @mcp.tool()
        def get_expert_retro_synthesis_predictions(products: list[str]) -> list[str]:
            """
            Given a product molecule, predict the likely reactants and other chemical species (e.g., agents, solvents).

            Args:
                products (list[str]): a list of product molecules in SMILES representation, for one reaction.
            Returns:
                list[str]: a list of predictions, each of which is a json string listing the predicted reactant molecule(s) in SMILES,
                    as well as potential (re)agents and solvents used in the reaction.
            """
            logger.debug("Calling `predict_reaction_reactants`")
            return predict_reaction_internal(molecules=products, forward=False)


if __name__ == "__main__":
    main()
    mcp.run(
        transport="sse",
    )
