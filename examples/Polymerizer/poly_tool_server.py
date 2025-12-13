# server.py
# pip install "mcp[cli]" rdkit-pypi
from typing_extensions import TypedDict
from typing import Literal, List
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp import Context

# import your existing module (the one we’ve been building)
import polymer_rules as pr

mcp = FastMCP(
    "Polymerizer",
    json_response=True,
    # description=(
    # "Expose monomer→polymer repeat transforms via MCP tools. "
    # "Tools: polymerize_explicit, polymerize_auto, suggest_rules."
)


# ----- Structured outputs for nicer tool schemas -----
class Suggestion(TypedDict):
    strategy: str
    confidence: float
    reason: str


class PolymerizeResult(TypedDict):
    repeat_smiles: str
    strategy: str
    rationale: str


# ----- Tools -----


@mcp.tool()
def polymerize_explicit(
    monomer_smiles: str,
    strategy: Literal[
        "vinyl",
        "acrylate",
        "rop_thf",
        "rop_epoxide",
        "ketene",
        "cond_alpha_hydroxy_acid",
        "cond_diphenol",
        "rop_lactam",
        "cond_omega_amino_acid",
        "alkyne",
        "polyacetylene",  # optional pretty-printer for C#C
    ],
    bigsmiles_wrap: bool = False,
) -> str:
    """
    Apply a specific polymerization rule and return the repeat-unit SMILES with [*] endpoints.
    Set bigsmiles_wrap=True to wrap in a simple {…} block.
    """
    rep = pr.monomer_to_repeat_smiles(monomer_smiles, strategy=strategy)
    return pr.wrap_bigsmiles_like(rep) if bigsmiles_wrap else rep


@mcp.tool()
def suggest_rules(monomer_smiles: str, top_k: int = 5) -> List[Suggestion]:
    """
    Inspect a monomer and return ranked candidate strategies with reasons.
    This does NOT perform any transformation.
    """
    ranked = pr.suggest_polymerization_rules(monomer_smiles)
    out = [
        {"strategy": s.strategy, "confidence": float(s.confidence), "reason": s.reason}
        for s in ranked[:top_k]
    ]
    return out


@mcp.tool()
def polymerize_auto(
    monomer_smiles: str,
    min_confidence: float = 0.80,
    allow_fallback_to_lower_confidence: bool = True,
    bigsmiles_wrap: bool = False,
) -> PolymerizeResult:
    """
    Auto-select and apply a single-monomer rule.
    Returns repeat SMILES, chosen strategy, and rationale.
    Raises a helpful error if a comonomer is required or the case is ambiguous.
    """
    rep, strat, why = pr.monomer_to_repeat_auto(
        monomer_smiles,
        min_confidence=min_confidence,
        allow_fallback_to_lower_confidence=allow_fallback_to_lower_confidence,
    )
    rep_out = pr.wrap_bigsmiles_like(rep) if bigsmiles_wrap else rep
    return {"repeat_smiles": rep_out, "strategy": strat, "rationale": why}


# ----- Run the server -----
if __name__ == "__main__":
    # Streamable HTTP transport is easy to test & works with hosted MCP tools
    mcp.run(transport="sse")  # defaults to http://localhost:8000/mcp
