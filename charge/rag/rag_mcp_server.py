import argparse
import json

import datasets
from datasets import load_dataset
from loguru import logger

datasets.disable_caching()

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, PreTrainedTokenizer
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
from charge.servers.FLASKv2_reactions import format_rxn_prompt

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


from mcp.server.fastmcp import FastMCP

mcp = FastMCP("RAG Server", json_response=True)


def generate_generic_reaction_prompt(data: dict, forward: bool, reaction_prompt: ReactionDataPrompt) -> dict:
    # Determine input and output roles
    lhs_roles = ['reactants', 'agents', 'solvents', 'catalysts', 'atmospheres']  # lhs = left hand side
    input_roles  = lhs_roles    if forward else ['products']
    output_roles = ['products'] if forward else lhs_roles
    all_roles = lhs_roles + ['products']
    reaction_prompt.sections['input data'] = json.dumps({k: data[k] for k in input_roles if data.get(k)})
    data['messages'] = [{'role': 'user', 'content': str(reaction_prompt)}]
    return data


def generate_expert_only_reaction_prompt(data: dict, forward: bool, reaction_prompt: ReactionDataPrompt) -> dict:
    # Determine input and output roles
    lhs_roles = ['reactants', 'agents', 'solvents', 'catalysts', 'atmospheres']  # lhs = left hand side
    input_roles  = lhs_roles    if forward else ['products']
    output_roles = ['products'] if forward else lhs_roles
    reaction_prompt.sections['input data'] = json.dumps({k: data[k] for k in input_roles if data.get(k)})
    reaction_prompt.sections['expert prediction'] = data['expert predictions']['responses'][0]
    data['messages'] = [{'role': 'user', 'content': str(reaction_prompt)}]
    return data



@mcp.tool()
def search_similar_reactions(data: dict, forward: bool, k_r: int) -> dict:
    """Add similar reactions to the reaction data dictionary.

    This function retrieves similar reactions from a reaction database and
    populates the following fields within ``data``:
    - ``data['similar']``: A list of list of dictionaries. The inner lists are of length
      `k_r`, and each dictionary is of a retrieved similar reaction.

    Args:
        data (dict): Reaction data dictionary that will be augmented.
            Adds the similar key:
                similar (list[list[dict]]): List populated with retrieved similar
                    reactions.
            Additional keys may also be present but are not required or modified.
        forward (bool): Whether prediction is for forward synthesis
            (True) or retrosynthesis (False).
        k_r (int): Number of similar reactions to retrieve.

    Returns:
        dict: The updated reaction data dictionary with new fields populated.
    """
    input_role = 'reactants' if forward else 'products'
    return search_similar_reactions_by_role(data, input_role, k_r)


def search_similar_reactions_by_role(data: dict, role: str, k_r: int) -> dict:
    """Add similar reactions to the reaction data dictionary.

    This function retrieves similar reactions from a reaction database and
    populates the following fields within ``data``:
    - ``data['similar']``: A list of list of dictionaries. The inner lists are of length
      `k_r`, and each dictionary is of a retrieved similar reaction.

    Args:
        data (dict): Reaction data dictionary that will be augmented.
            Adds the similar key:
                similar (list[list[dict]]): List populated with retrieved similar
                    reactions.
            Additional keys may also be present but are not required or modified.
        role (str): The role to search similar reactions for. Must be one of 'reactants' or 'products'.
        k_r (int): Number of similar reactions to retrieve.

    Returns:
        dict: The updated reaction data dictionary with new fields populated.
    """
    assert role in ['reactants', 'products'], f"role must be one of 'reactants' or 'products', but got {role}"
    input_role = role
    logger.info(f'data is {data}')
    assert isinstance(data[input_role], list) and isinstance(data[input_role][0], list), \
        "This data processing function must be called in batched mode, e.g., using `dataset.map(..., batched=True)`."
    num_reactions = len(data[input_role])
    query_smiles = ['.'.join(data[input_role][i]) for i in range(num_reactions)]
    query_emb = embedder.embed_smiles(query_smiles)
    similar_dist, similar_idx, similar = retriever.search_similar(query_emb, k=k_r)
    data['similar'] = similar  # list of list of dicts. Each list has k_r dicts, each dict is a similar reaction.
    return data


def predict_reaction_internal(molecules: list[str], forward: bool) -> list[str]:
    """

    Args:
        molecules (list[str]): list of SMILES strings for one side of a reaction
        forward (bool): True if for forward synthesis, False for retrosynthesis

    Returns:

    """
    # Copied from FLASKv2_reactions.py, b/c we both use global variables.
    if not HAS_FLASKV2:
        raise ImportError(
            "Please install the [flask] optional packages to use this module."
        )
    model = forward_expert_model if forward else retro_expert_model
    data = {'reactants': molecules} if forward else {'products': molecules}
    with torch.inference_mode():
        prompt = format_rxn_prompt(data, forward=forward)
        prompt = apply_chat_template(prompt, tokenizer=tokenizer)
        inputs = tokenizer(prompt["prompt"], return_tensors="pt", padding="longest").to('cuda')
        prompt_length = inputs["input_ids"].size(1)
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            num_return_sequences=3,
            # do_sample=True,
            num_beams=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,  # enable KV cache
        )
        processed_outputs = [tokenizer.decode(out[prompt_length:], skip_special_tokens=True) for out in outputs]
    logger.debug(f'Model input: {prompt["prompt"]}')
    processed_outs = "\n".join(processed_outputs)
    logger.debug(f'Model output: {processed_outs}')
    return processed_outputs


def add_expert_predictions_on_similar_data(data: dict, expert_predictions: datasets.Dataset) -> dict:
    """
    Args:
        data (dict): reaction data that contains information about similar reactions.
            A data retriever must be used beforehand to populate `data['similar']` and `data['similar idx']`.
        expert_predictions (datasets.Dataset): expert preditions that correspond to the database used by the retriever, e.g., USPTO-50k train split.
            We currently assume that `expert_predictions` has two fields: 'prompt' and 'responses'.
            - prompt: the input prompt that was fed to the expert model.
            - responses: a list of LLM-generated strings, where each string is a prediction output. 
    """
    indices = data['similar idx']
    # Taking only the top-1 expert prediction per similar reaction
    data['expert predictions on similar data'] = [expert_predictions[i]['responses'][0] for i in indices]
    return data


# @mcp.tool()
def generate_ragv1_reaction_prompt(data: dict, forward: bool, reaction_prompt: ReactionDataPrompt, k_r: int = 3) -> dict:
    """Generates a RAG prompt string, with v1 format for one reaction.
    Example of data format: {"reactants": ["CCCC"], }
    """
    retrieve_similar_reactions(data, forward, k_r=k_r)
    return _generate_ragv1_reaction_prompt(data, forward, reaction_prompt)

def _generate_ragv1_reaction_prompt(data: dict, forward: bool, reaction_prompt: ReactionDataPrompt) -> dict:
    # Ex of data format: {"reactants": [["CCCC"]], "similar": [[{"products": ["CCCO"], "reactants": ["CCC", "O"]}], []...]}
    # Determine input and output roles
    assert len(data['similar']) == 1, 'This function is not for batched mode'

    lhs_roles = ['reactants', 'agents', 'solvents', 'catalysts', 'atmospheres']  # lhs = left hand side
    input_roles  = lhs_roles    if forward else ['products']
    output_roles = ['products'] if forward else lhs_roles

    # Add similar reaction input-output pairs
    reaction_prompt.sections['data table'] = "Input | Ground truth output\n"
    for s in data['similar'][0]:
        similar_input = json.dumps({k: s[k] for k in input_roles  if s.get(k)}) 
        true_output   = json.dumps({k: s[k] for k in output_roles if s.get(k)})
        reaction_prompt.sections['data table'] += f'{similar_input} | {true_output}\n'

    # Add this last line: actual input | ????
    actual_input = json.dumps({k: data[k] for k in input_roles if data.get(k)})
    reaction_prompt.sections['data table'] += f'{actual_input} | ???'
    
    data['messages'] = [{'role': 'user', 'content': str(reaction_prompt)}]
    del data['similar_idx']  # Not helpful to the LLM
    return data


def generate_ragv3_reaction_prompt(data: dict, forward: bool, reaction_prompt: ReactionDataPrompt) -> dict:
    # Determine input and output roles
    lhs_roles = ['reactants', 'agents', 'solvents', 'catalysts', 'atmospheres']  # lhs = left hand side
    input_roles  = lhs_roles    if forward else ['products']
    output_roles = ['products'] if forward else lhs_roles

    # Add similar reaction input-output and expert prediction
    reaction_prompt.sections['data table'] = "Input | Ground truth output | Predicted output\n"
    for s, e in zip(data['similar'], data['expert predictions on similar data']):
        similar_input = json.dumps({k: s[k] for k in input_roles  if s.get(k)}) 
        true_output   = json.dumps({k: s[k] for k in output_roles if s.get(k)})
        pred_output   = e
        reaction_prompt.sections['data table'] += f'{similar_input} | {true_output} | {pred_output}\n'

    # Add this last line: actual input | ???? | expert prediction
    actual_input = json.dumps({k: data[k] for k in input_roles if data.get(k)})
    pred_output  = data['expert predictions']['responses'][0]  # only choose the first expert prediction
    reaction_prompt.sections['data table'] += f'{actual_input} | ??? | {pred_output}'
    
    data['messages'] = [{'role': 'user', 'content': str(reaction_prompt)}]
    return data


def generate_ragv4_reaction_prompt(data: dict, forward: bool, reaction_prompt: ReactionDataPrompt) -> dict:
    # Determine input and output roles
    lhs_roles = ['reactants', 'agents', 'solvents', 'catalysts', 'atmospheres']  # lhs = left hand side
    input_roles  = lhs_roles    if forward else ['products']
    output_roles = ['products'] if forward else lhs_roles

    # Add rows with this format: Input | Ground truth output | Predicted output | Neighbor distance
    reaction_prompt.sections['data table'] = "Input | Ground Truth Output | Predicted Output | Neighbor Distance\n"
    for s, pred_output, dist in zip(data['similar'], data['expert predictions on similar data'], data['similar dist']):
        similar_input = json.dumps({k: s[k] for k in input_roles  if s.get(k)}) 
        true_output   = json.dumps({k: s[k] for k in output_roles if s.get(k)})
        reaction_prompt.sections['data table'] += f'{similar_input} | {true_output} | {pred_output} | {dist:.3f}\n'

    # Add this last line: actual input | ???? | expert prediction
    actual_input = json.dumps({k: data[k] for k in input_roles if data.get(k)})
    pred_output  = data['expert predictions']['responses'][0]  # only choose the first expert prediction
    reaction_prompt.sections['data table'] += f'{actual_input} | ??? | {pred_output} | 0'
    
    data['messages'] = [{'role': 'user', 'content': str(reaction_prompt)}]
    return data


embedder : SmilesEmbedder = None
retriever : FaissDataRetriever = None
forward_expert_model = None
retro_expert_model = None
tokenizer = None

def main():
    global embedder, retriever, forward_expert_model, retro_expert_model, tokenizer
    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--vllm-url',   type=str, help='vLLM server URL', default='http://192.168.128.34:8011/v1/chat/completions')
    parser.add_argument('--model-path', type=str, help='Model path on vLLM server', default='/p/vast1/flask/models/marathe1/gpt-oss-120b')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--eval-data',  type=str, help='Eval data file in JSONL format')
    parser.add_argument('--expert-pred-train-path', type=str, help='Path to expert prediction file (train set)')
    parser.add_argument('--expert-pred-test-path',  type=str, help='Path to expert prediction file (test set)')
    parser.add_argument('--database-path', type=str, help='Path to database file (JSON) for similarity search')
    parser.add_argument('--forward-embedding-path', type=str, help='Path to embedding file (NPY) corresponding to similarity search')
    parser.add_argument('--retro-embedding-path', type=str, help='Path to embedding file (NPY) corresponding to similarity search')
    parser.add_argument('--forward-expert-model-path', type=str, help='Path to forward expert model')
    parser.add_argument('--retro-expert-model-path', type=str, help='Path to retro expert model')
    parser.add_argument('--reasoning-effort', type=str, choices=['low', 'medium', 'high'])
    # parser.add_argument('--k_e', type=int, help='Number of expert predictions (per input data) to show in prompt')
    parser.add_argument('--k_r', type=int, help='Number of similar reactions (per input data) to retrieve', default=3)
    parser.add_argument('--retrosynthesis', action='store_true', help='Whether the context is retrosynthesis. Otherwise it is forward synthesis.')
    rag_version_group = parser.add_mutually_exclusive_group()
    rag_version_group.add_argument('--rag-version', type=int, help='RAG version')
    rag_version_group.add_argument('--non-rag-version', type=str, help='Non-RAG version', choices=['basic', 'expert_only'], default=None)
    args = parser.parse_args()    
    for k, v in vars(args).items():
        print(f'{k} = {v}', flush=True)

    ## Load datasets
    # dataset = load_dataset(
    #     'json',
    #     data_files={'test': args.eval_data},
    # )
    # test_set = dataset['test']
    # expert_predictions = load_dataset(
    #     'json',
    #     data_files={
    #         'train': args.expert_pred_train_path,
    #         'test': args.expert_pred_test_path,
    #     },
    # )

    ## Run inference on only the first n samples
    # test_set = test_set.select(range(1000))
    # expert_predictions['test'] = expert_predictions['test'].select(range(1000))
    expert_predictions = None

    # Generate prompt
    forward = (not args.retrosynthesis)

    # Init RAG components
    embedder = SmilesEmbedder(
        model_path='/p/vast1/flask/team/tim/models/chemformer/chemformer_encoder.ts',
        tokenizer=ChemformerTokenizer(vocab_path='/p/vast1/flask/team/tim/models/chemformer/bart_vocab.json')
    )
    forward_retriever = FaissDataRetriever(
        data_path=args.database_path,
        emb_path=args.forward_embedding_path,
    )
    retro_retriever = FaissDataRetriever(
        data_path=args.database_path,
        emb_path=args.retro_embedding_path,
    )
    retriever = forward_retriever if forward else retro_retriever

    tokenizer = AutoTokenizer.from_pretrained(args.forward_expert_model_path or args.retro_expert_model_path, padding_side="left")
    tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
    if args.forward_expert_model_path is not None:
        forward_expert_model = AutoModelForCausalLM.from_pretrained(
            args.forward_expert_model_path,
            device_map='cuda',
            torch_dtype=torch.bfloat16,
        )
    if args.retro_expert_model_path is not None:
        retro_expert_model = AutoModelForCausalLM.from_pretrained(
            args.retro_expert_model_path,
            device_map='cuda',
            torch_dtype=torch.bfloat16,
        )

    if forward_expert_model is not None:
        @mcp.tool()
        def get_expert_forward_synthesis_predictions(reactants: list[str]) -> list[str]:
            """
            Given a set of reactants and possibly reagent molecules, predict the likely product molecule(s).

            Args:
                reactants (list[str]): A list of smiles of reactant and reagent molecules in SMILES representation, for
                    one reaction
            Returns:
                list[str]: A list of product predictions, each of which is  json string listing the predicted product molecule(s) in SMILES.
            """
            logger.debug('Calling `predict_reaction_products`')
            logger.debug(f'Input reactions: {reactants}')
            res = predict_reaction_internal(molecules=reactants, forward=True)
            logger.debug(f'Output predictions: {res}')
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
            logger.debug('Calling `predict_reaction_reactants`')
            return predict_reaction_internal(molecules=products, forward=False)

    # Retrieve similar reactions
    # test_set = test_set.map(
    #     retrieve_similar_reactions,
    #     fn_kwargs=dict(forward=forward, embedder=embedder, retriever=retriever, k_r=args.k_r),
    #     batched=True,
    #     batch_size=32,
    # )

    match args.rag_version:
        case 1:
            from charge.rag.prompts import ReactionDataPrompt_RAG
            reaction_prompt = ReactionDataPrompt_RAG(forward=forward)
            generate_prompt_func = generate_ragv1_reaction_prompt
        case 2:
            raise NotImplementedError
        case 3:
            raise NotImplementedError('Will need expert model set up as a server')
            from charge.rag.prompts import ReactionDataPrompt_RAGv3
            reaction_prompt = ReactionDataPrompt_RAGv3(forward=forward)
            generate_prompt_func = generate_ragv3_reaction_prompt
            test_set = test_set.add_column('expert predictions', expert_predictions['test'])
            test_set = test_set.map(
                add_expert_predictions_on_similar_data,
                fn_kwargs=dict(expert_predictions=expert_predictions['train']),
                num_proc=8,
            )
        case 4:
            raise NotImplementedError('Will need expert model set up as a server')
            from charge.rag.prompts import ReactionDataPrompt_RAGv4
            reaction_prompt = ReactionDataPrompt_RAGv4(forward=forward)
            generate_prompt_func = generate_ragv4_reaction_prompt
            test_set = test_set.add_column('expert predictions', expert_predictions['test'])
            test_set = test_set.map(
                add_expert_predictions_on_similar_data,
                fn_kwargs=dict(expert_predictions=expert_predictions['train']),
                num_proc=8,
            )
        case _:
            raise NotImplementedError


if __name__ == '__main__':
    main()
    #mcp.run(transport="streamable-http")
    mcp.run(transport="sse", )
    from charge.rag.prompts import ReactionDataPrompt_RAG
    # reaction_prompt = ReactionDataPrompt_RAG(forward=True)
    # res = generate_ragv1_reaction_prompt({'reactants': [["CCC"]]}, forward=True , reaction_prompt=reaction_prompt, k_r=3)
    # print(res)
    
