import click
from loguru import logger
import json
from mcp.server.fastmcp import FastMCP
from typing import Optional

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, PreTrainedTokenizer
    from peft import PeftModel
    from trl import apply_chat_template
    import torch
    HAS_FLASKV2 = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_FLASKV2 = False
    logger.warning(
        "Please install the flask support packages to use this module."
        "Install it with: pip install charge[flask]",
    )

from charge.servers.server_utils import update_mcp_network, get_hostname

def format_rxn_prompt(data: dict, forward: bool) -> dict:
    required_keys = ['reactants', 'products', 'agents', 'solvents', 'catalysts', 'atmospheres']
    non_product_keys = [k for k in required_keys if k != 'products']
    if forward:
        d = {k: data[k] for k in non_product_keys if data.get(k, None)}
        prompt = json.dumps(d)
    else:
        d = {'products': data['products']}
        prompt = json.dumps(d)
    data['prompt'] = [{'role': 'user', 'content': prompt}]
    return data


def predict_reaction_internal(molecules: list[str], retrosynthesis: bool) -> list[str]:
    if not HAS_FLASKV2:
        raise ImportError(
            "Please install the [flask] optional packages to use this module."
        )
    model = retro_model if retrosynthesis else fwd_model
    data = {'products': molecules} if retrosynthesis else {'reactants': molecules}
    with torch.inference_mode():
        prompt = format_rxn_prompt(data, forward=(not retrosynthesis))
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


@click.command()
@click.option("--model-dir-fwd", envvar="FLASKV2_MODEL_FWD", help="Path to flaskv2 model")
@click.option("--model-dir-retro", envvar="FLASKV2_MODEL_RETRO", help="Path to flaskv2 model for retrosynthesis")
@click.option("--adapter-weights-fwd", help="LoRA adapter weights, if used")
@click.option("--adapter-weights-retro", help="LoRA adapter weights for retrosynthesis model, if used")
@click.option("--transport", type=click.Choice(['stdio', 'streamable-http', 'sse']), help="MCP transport type", default="sse")
@click.option("--port", type=int, default=8125, help="Port to run the server on")
@click.option("--host", type=str, default=None, help="Host to run the server on")
def main(model_dir_fwd: str, model_dir_retro: str, adapter_weights_fwd: str, adapter_weights_retro: str, transport: str, port: str, host: Optional[str]):
    if not HAS_FLASKV2:
        raise ImportError(
            "Please install the [flask] optional packages to use this module."
        )
    if not model_dir_fwd and not model_dir_retro:
        raise ValueError("At least one model has to be given to the MCP server")

    if host is None:
        _, host = get_hostname()

    mcp = FastMCP("FLASKv2 Reaction Predictor",
                  port=port,
                  website_url=f"{host}",
    )

    # Init MCP server
    mcp = FastMCP("FLASKv2 Reaction Predictor", host=host, port=port)

    # Make HF models and tokenizer global objects
    global fwd_model, retro_model, tokenizer
    fwd_model = None
    retro_model = None

    # Load tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(model_dir_fwd or model_dir_retro, padding_side="left")
    tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
    if model_dir_fwd:
        fwd_model = AutoModelForCausalLM.from_pretrained(
            model_dir_fwd,
            device_map='cuda',
            torch_dtype=torch.bfloat16,
        )
        if adapter_weights_fwd is not None:
            fwd_model = PeftModel.from_pretrained(fwd_model, adapter_weights_fwd)
            fwd_model = fwd_model.merge_and_unload()
    if model_dir_retro:
        retro_model = AutoModelForCausalLM.from_pretrained(
            model_dir_retro,
            device_map='cuda',
            torch_dtype=torch.bfloat16,
        )
        if adapter_weights_retro is not None:
            retro_model = PeftModel.from_pretrained(retro_model, adapter_weights_retro)
            retro_model = retro_model.merge_and_unload()

    # Enable model optimizations
    if fwd_model is not None:
        fwd_model.eval()
        if hasattr(fwd_model, "config") and hasattr(fwd_model.config, "use_cache"):
            fwd_model.config.use_cache = True  # enable KV caching
    if retro_model is not None:
        retro_model.eval()
        if hasattr(retro_model, "config") and hasattr(retro_model.config, "use_cache"):
            retro_model.config.use_cache = True  # enable KV caching

    # Dynamic tool creation based on input models
    available_tools = []
    if fwd_model is not None:
        available_tools.append("Forward Prediction")

        @mcp.tool()
        def predict_reaction_products(reactants: list[str]) -> list[str]:
            """
            Given a set of reactant molecules, predict the likely product molecule(s).

            Args:
                reactants (list[str]): a list of reactant molecules in SMILES representation.
            Returns:
                list[str]: a list of predictions, each of which is a json string listing the predicted product molecule(s) in SMILES.
            """
            logger.debug('Calling `predict_reaction_products`')
            return predict_reaction_internal(reactants, False)

    if retro_model is not None:
        available_tools.append("Single-Step Retrosynthesis")

        @mcp.tool()
        def predict_reaction_reactants(products: list[str]) -> list[str]:
            """
            Given a product molecule, predict the likely reactants and other chemical species (e.g., agents, solvents).

            Args:
                products (list[str]): a list of product molecules in SMILES representation.
            Returns:
                list[str]: a list of predictions, each of which is a json string listing the predicted reactant molecule(s) in SMILES,
                    as well as potential (re)agents and solvents used in the reaction.
            """
            logger.debug('Calling `predict_reaction_reactants`')
            return predict_reaction_internal(products, True)

    logger.info(f"Available tools: {', '.join(available_tools)}")

    # Run MCP server
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
