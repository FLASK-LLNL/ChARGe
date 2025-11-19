import click
from loguru import logger
import json
from mcp.server.fastmcp import FastMCP
from typing import Optional

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LlamaForCausalLM,
        PreTrainedTokenizer,
    )
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
from charge.servers.ServerToolkit import ServerToolkit


def format_rxn_prompt(data: dict, forward: bool) -> dict:
    required_keys = [
        "reactants",
        "products",
        "agents",
        "solvents",
        "catalysts",
        "atmospheres",
    ]
    non_product_keys = [k for k in required_keys if k != "products"]
    if forward:
        d = {k: data[k] for k in non_product_keys if data.get(k, None)}
        prompt = json.dumps(d)
    else:
        d = {"products": data["products"]}
        prompt = json.dumps(d)
    data["prompt"] = [{"role": "user", "content": prompt}]
    return data


class FlaskV2ReactionServer(ServerToolkit):
    def __init__(
        self,
        mcp: FastMCP,
        model_dir_fwd: Optional[str],
        model_dir_retro: Optional[str],
        adapter_weights_fwd: Optional[str],
        adapter_weights_retro: Optional[str],
    ):
        super().__init__(mcp)

        if not HAS_FLASKV2:
            raise ImportError(
                "Please install the [flask] optional packages to use this module."
            )
        assert (
            model_dir_fwd or model_dir_retro
        ), "At least one model has to be given to the MCP server"

        # Load tokenizer and models
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir_fwd or model_dir_retro, padding_side="left"
        )
        self.tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
        self.fwd_model = self._load_model(model_dir_fwd, adapter_weights_fwd)
        self.retro_model = self._load_model(model_dir_retro, adapter_weights_retro)

        self.available_tools = []
        if self.fwd_model is not None:
            self.available_tools.append("Forward Prediction")
            self._register_single_method(self.predict_reaction_products)
        if self.retro_model is not None:
            self.available_tools.append("Single-Step Retrosynthesis")
            self._register_single_method(self.predict_reaction_reactants)

    def _load_model(self, model_dir: Optional[str], adapter_weights: Optional[str]):
        if model_dir is None:
            return None
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        if adapter_weights is not None:
            model = PeftModel.from_pretrained(model, adapter_weights)
            model = model.merge_and_unload()

        model.eval()
        # Enable model optimizations
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = True  # enable KV caching
        return model

    def predict_reaction_products(self, reactants: list[str]) -> list[str]:
        """
        Given a set of reactant molecules, predict the likely product molecule(s).

        Args:
            reactants (list[str]): a list of reactant molecules in SMILES representation.

        Returns:
            list[str]: a list of predictions, each of which is a json string listing the predicted product molecule(s) in SMILES.
        """

        return self._predict_reaction_internal(reactants, False)

    def predict_reaction_reactants(self, products: list[str]) -> list[str]:
        """
        Given a set of product molecules, predict the likely reactant molecule(s).

        Args:
            products (list[str]): a list of product molecules in SMILES representation.

        Returns:
            list[str]: a list of predictions, each of which is a json string listing the predicted reactant molecule(s) in SMILES.
        """
        return self._predict_reaction_internal(products, True)

    def _predict_reaction_internal(
        self, molecules: list[str], retrosynthesis: bool
    ) -> list[str]:

        model = self.retro_model if retrosynthesis else self.fwd_model
        data = {"products": molecules} if retrosynthesis else {"reactants": molecules}
        with torch.inference_mode():
            prompt = format_rxn_prompt(data, forward=(not retrosynthesis))
            prompt = apply_chat_template(prompt, tokenizer=self.tokenizer)
            inputs = self.tokenizer(
                prompt["prompt"], return_tensors="pt", padding="longest"
            ).to("cuda")
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
            processed_outputs = [
                tokenizer.decode(out[prompt_length:], skip_special_tokens=True)
                for out in outputs
            ]
        logger.debug(f'Model input: {prompt["prompt"]}')
        processed_outs = "\n".join(processed_outputs)
        logger.debug(f"Model output: {processed_outs}")
        return processed_outputs

    def get_available_tools(self) -> list[str]:
        return self.available_tools


@click.command()
@click.option(
    "--model-dir-fwd", envvar="FLASKV2_MODEL_FWD", help="Path to flaskv2 model"
)
@click.option(
    "--model-dir-retro",
    envvar="FLASKV2_MODEL_RETRO",
    help="Path to flaskv2 model for retrosynthesis",
)
@click.option("--adapter-weights-fwd", help="LoRA adapter weights, if used")
@click.option(
    "--adapter-weights-retro",
    help="LoRA adapter weights for retrosynthesis model, if used",
)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http", "sse"]),
    help="MCP transport type",
    default="sse",
)
@click.option("--port", type=int, default=8125, help="Port to run the server on")
@click.option("--host", type=str, default=None, help="Host to run the server on")
def main(
    model_dir_fwd: str,
    model_dir_retro: str,
    adapter_weights_fwd: str,
    adapter_weights_retro: str,
    transport: str,
    port: str,
    host: Optional[str],
):
    if not HAS_FLASKV2:
        raise ImportError(
            "Please install the [flask] optional packages to use this module."
        )
    if not model_dir_fwd and not model_dir_retro:
        raise ValueError("At least one model has to be given to the MCP server")

    if host is None:
        _, host = get_hostname()

    mcp = FastMCP(
        "FLASKv2 Reaction Predictor",
        port=port,
        website_url=f"{host}",
    )

    # Init MCP server
    mcp = FastMCP("FLASKv2 Reaction Predictor", host=host, port=port)

    server = FlaskV2ReactionServer(
        mcp, model_dir_fwd, model_dir_retro, adapter_weights_fwd, adapter_weights_retro
    )

    logger.info(f"Available tools: {', '.join(server.get_available_tools())}")

    # Run MCP server
    server.run(transport=transport)


if __name__ == "__main__":
    main()
