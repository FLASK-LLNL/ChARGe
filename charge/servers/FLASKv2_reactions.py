"""
Persistent server that serves one or two FLASKv2 LLMs for forward reaction prediction and retrosynthesis
"""

import click
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, PreTrainedTokenizer
    from peft import PeftModel
    from trl import apply_chat_template
    import torch
except ImportError:
    raise ImportError(
        "Please install the [flask] optional packages to use this module."
    )
from mcp.server.fastmcp import FastMCP
from typing import List


from loguru import logger
import logging

logging.basicConfig(level=logging.INFO)

mcp = FastMCP("FLASKv2 Reaction Predictor")


def format_rxn_prompt(data: list[str], forward: bool) -> dict[str, list[dict[str, str]]]:
    if forward:
        reactants = [f"`{smi}`" for smi in data]
        user_prompt = (
            "Predict the chemical reaction product that would be synthesized based on the following reactant molecule(s) in SMILES representation. "
            "Your response must start with the string 'Product: ' followed by the SMILES of the product molecule. "
            "Each SMILES string must be surrounded with the '`' symbol. "
            "Do not output anything else.\n"
        )
        user_prompt += "\n".join(reactants)
    else:
        products = [f"`{smi}`" for smi in data]
        user_prompt = (
            "Predict the chemical reactants needed for synthesizing the following product molecule(s) in SMILES representation. "
            "Your response must be one or multiple lines of text, each starting with the string 'Reactant: ' followed by the SMILES of a reactant molecule. "
            "Each SMILES string must be surrounded with the '`' symbol. "
            "Do not output anything else.\n"
        )
        user_prompt += "\n".join(products)

    data: dict[str, list[dict[str, str]]] = {}
    data["prompt"] = [{"role": "user", "content": user_prompt}]
    return data


# Keep global state
fwd_model: LlamaForCausalLM = None
retro_model: LlamaForCausalLM = None
tokenizer: PreTrainedTokenizer = None
device = None


def predict_reaction_internal(data: List[str], retrosynthesis: bool) -> List[str]:
    SEQS = 3
    model = fwd_model if not retrosynthesis else retro_model
    with torch.inference_mode():
        prompt = format_rxn_prompt(data, forward=not retrosynthesis)
        prompt = apply_chat_template(prompt, tokenizer=tokenizer)
        inputs = tokenizer(prompt["prompt"], return_tensors="pt", padding="longest").to(device)
        prompt_length = inputs["input_ids"].size(1)

        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            num_return_sequences=SEQS,
            do_sample=False,
            num_beams=SEQS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,  # enable KV cache
        )
        processed_outputs = [tokenizer.decode(out[prompt_length:], skip_special_tokens=True) for out in outputs]

    result: list[str] = []
    for line in processed_outputs[0].splitlines():
        result.append(line.strip().removeprefix("Reactant: `" if retrosynthesis else "Product: `").removesuffix("`"))

    logger.info(f'{"RETRO" if retrosynthesis else "FORWARD"}: Got {data}, returned {result}')

    return result


@click.command()
@click.option("--model-dir-fwd", envvar="FLASKV2_MODEL_FWD", help="Path to flaskv2 model")
@click.option("--model-dir-retro", envvar="FLASKV2_MODEL_RETRO", help="Path to flaskv2 model for retrosynthesis")
@click.option("--adapter-weights-fwd", help="LoRA adapter weights, if used")
@click.option("--adapter-weights-retro", help="LoRA adapter weights for retrosynthesis model, if used")
def main(model_dir_fwd: str, adapter_weights_fwd: str, model_dir_retro: str, adapter_weights_retro: str):
    if not model_dir_fwd and not model_dir_retro:
        raise ValueError("At least one model has to be given to the MCP server")
    global fwd_model, retro_model, tokenizer, device, retrosynthesis
    device = torch.device("cuda")

    print("Loading tokenizer...", flush=True, end="")
    tokenizer = AutoTokenizer.from_pretrained(model_dir_fwd or model_dir_retro, padding_side="left")
    tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
    print("Done", flush=True)

    print("Loading models...", flush=True)
    if model_dir_fwd:
        fwd_model = AutoModelForCausalLM.from_pretrained(
            model_dir_fwd,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        if adapter_weights_fwd is not None:
            fwd_model = PeftModel.from_pretrained(fwd_model, adapter_weights_fwd)
            fwd_model = fwd_model.merge_and_unload()

    if model_dir_retro:
        retro_model = AutoModelForCausalLM.from_pretrained(
            model_dir_retro,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        if adapter_weights_retro is not None:
            retro_model = PeftModel.from_pretrained(retro_model, adapter_weights_retro)
            retro_model = retro_model.merge_and_unload()
    print("Models loaded")

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
        def predict_reaction_products(reactants: List[str]) -> List[str]:
            """
            Analyzes a set of molecules to compute what likely product molecules of a
            chemical reaction would be. The reactants and products are both given as
            SMILES strings.

            Args:
                reactants (List[str]): A list of reactant molecules given as SMILES strings.
            Returns:
                List[str]: A list of product molecules, given as SMILES strings.
            """
            return predict_reaction_internal(reactants, False)

    if retro_model is not None:
        available_tools.append("Single-Step Retrosynthesis")

        @mcp.tool()
        def predict_reaction_reactants(products: List[str]) -> List[str]:
            """
            Performs a retrosynthetic analysis to compute, for a given set of products
            in a chemical reaction, a set of potential reactants. The products and
            reactants are both given as SMILES strings.

            Args:
                products (List[str]): A list of product molecules given as SMILES strings.
            Returns:
                List[str]: A list of reactant molecules, given as SMILES strings.
            """
            return predict_reaction_internal(products, True)

    logger.info(f"Available tools: {', '.join(available_tools)}")

    # Run server
    mcp.run(
        transport="sse",
    )


if __name__ == "__main__":
    main()
