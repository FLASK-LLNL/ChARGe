# polymer_rules.py
# pip install rdkit-pypi
from dataclasses import dataclass
from typing import List, Tuple, Optional

from rdkit import Chem
from rdkit.Chem import AllChem

# ============================================================
# ===============  Polymerization Reaction Rules  =============
# ============================================================

# ---- Vinyl addition (head-to-tail) --------------------------
# Converts terminal CH2=CH–X to repeat [*]CH2–CH(X)[*]
RXN_VINYL_HT = AllChem.ReactionFromSmarts("[CH2:1]=[C:2]>>[*:9][CH2:1]-[C:2]([*:10])")

# ---- Acrylate / Methacrylate selective rule ----------------

# Unsubstituted acrylate head-to-tail:
# [CH2]=[CH]-C(=O)X  ->  [*]CH2-[CH]([*])-C(=O)X
RXN_ACRYLATE_HT = AllChem.ReactionFromSmarts(
    "[CH2:1]=[CH:2]-[C:3](=O)[O,N:4]>>[*:9][CH2:1]-[CH:2]([*:10])-[C:3](=O)[O,N:4]"
)

# β-substituted (meth)acrylate (covers itaconates):
# [CH2]=[C](R)-C(=O)X  ->  [*]CH2-[C](R)([*])-C(=O)X
RXN_METHACRYLATE_HT = AllChem.ReactionFromSmarts(
    "[CH2:1]=[C:2]([#6:5])-[C:3](=O)[O,N:4]>>[*:9][CH2:1]-[C:2]([#6:5])([*:10])-[C:3](=O)[O,N:4]"
)

# ---- Ring-opening polymerization of THF ---------------------
# Opens a saturated 5-member cyclic ether (tetrahydrofuran)
RXN_ROP_THF = AllChem.ReactionFromSmarts(
    "[O;X2;R:1]1-[C;R:2]-[C;R:3]-[C;R:4]-[C;R:5]-1"
    ">>"
    "[*:9][O:1]-[C:2]-[C:3]-[C:4]-[C:5][*:10]"
)

# ---- Ring-opening polymerization of epoxides ----------------
# Converts 3-membered cyclic ethers (ethylene oxide, propylene oxide)
RXN_ROP_EPOXIDE = AllChem.ReactionFromSmarts(
    "[O;X2;R:1]1[C;R:2][C;R:3]1>>[*:9][O:1]-[C:2]-[C:3][*:10]"
)

# ---- Ketene polymerization (oxyethylene formation) ----------
# Converts ketene C=C=O into –CH2–CH2–O– repeat
RXN_KETENE_TO_PEO = AllChem.ReactionFromSmarts(
    "[C:1]=[C:2]=[O:3]>>[*:9][CH2:1][CH2:2][O:3][*:10]"
)

# ---- Condensation: α-hydroxy acid → polyester (PLA) ---------
# Example: lactic acid CC(O)C(=O)O -> [*]OC(C)C(=O)[*]
RXN_COND_ALPHA_HA = AllChem.ReactionFromSmarts(
    "[O;H1:1]-[C:2]-[C:3](=O)[O;H1:4]>>[*:9][O:1]-[C:2]-[C:3](=O)[*:10]"
)

# ---- Di-phenol self-condensation → poly(arylene ether) ------
# A) Hydroquinone-specific (para-diphenol) fast path:
RXN_DIPHENOL_HQ = AllChem.ReactionFromSmarts(
    "[O;H1:1]c1ccc([O;H1:2])cc1>>[*:9][O:1]c1ccc([O:2][*:10])cc1"
)
# B) General fallback: replace one phenolic –OH with [*]O– per application
RXN_DIPHENOL_GENERIC_ONE_END = AllChem.ReactionFromSmarts(
    "[O;H1:1]-[a:2]>>[*:9][O:1]-[a:2]"
)

RXN_LACTAM_R7_TO_N6 = AllChem.ReactionFromSmarts(
    # Map ring atoms explicitly so we lay out NH-(CH2)5-CO in order
    "[O:8]=[C:2]1[N:1][C:3][C:4][C:5][C:6][C:7]1"
    ">>"
    "[*:9][N:1]-[C:3]-[C:4]-[C:5]-[C:6]-[C:7]-[C:2](=O)[*:10]"
)
RXN_LACTAM_GENERIC = AllChem.ReactionFromSmarts(
    "[N;R:1]-[C;R:2](=O)>>[*:9][N:1]-[C:2](=O)[*:10]"
)
RXN_OMEGA_AMINO_ACID_N6 = AllChem.ReactionFromSmarts(
    "[N:1][C:2][C:3][C:4][C:5][C:6][C:7](=O)[O;H1:8]"
    ">>"
    "[*:9][N:1]-[C:2]-[C:3]-[C:4]-[C:5]-[C:6]-[C:7](=O)[*:10]"
)
RXN_OMEGA_AMINO_ACID_GENERIC = AllChem.ReactionFromSmarts(
    "[N:1]-[CH2:2]-[CH2:3]-[CH2:4]-[CH2:5]-[CH2:6]-[C:7](=O)[O;H1:8]"
    ">>"
    "[*:9][N:1]-[CH2:2]-[CH2:3]-[CH2:4]-[CH2:5]-[CH2:6]-[C:7](=O)[*:10]"
)
RXN_ALKYNE_TO_POLYALKENYLENE = AllChem.ReactionFromSmarts(
    "[C:1]#[C:2]>>[*:9][C:1]=[C:2][*:10]"
)
# ============================================================
# ===============  Helpers & Public API  =====================
# ============================================================


def first_valid_product(mol, rxn):
    """Run a reaction and return the first sanitized product molecule that succeeds."""
    out = rxn.RunReactants((mol,))
    for prods in out:
        try:
            prod = Chem.Mol(prods[0])
            Chem.SanitizeMol(prod)
            return prod
        except Exception:
            continue
    return None


def count_phenolic_OH(mol):
    """Count phenolic OH groups: O(H) attached to an aromatic carbon."""
    patt = Chem.MolFromSmarts("[O;H1]-a")
    return len(mol.GetSubstructMatches(patt))


def same_aromatic_component(mol):
    """Check the two phenolic OH are on the same aromatic connected component."""
    phenol_matches = mol.GetSubstructMatches(Chem.MolFromSmarts("[O;H1]-a"))
    if len(phenol_matches) != 2:
        return False
    _, a1 = phenol_matches[0]
    _, a2 = phenol_matches[1]
    arom = Chem.MolFromSmarts("a")
    arom_set = set(a for (a,) in mol.GetSubstructMatches(arom))
    # BFS across aromatic atoms
    visited = {a1}
    stack = [a1]
    while stack:
        cur = stack.pop()
        if cur == a2:
            return True
        atom = mol.GetAtomWithIdx(cur)
        for nbr in atom.GetNeighbors():
            j = nbr.GetIdx()
            if j in arom_set and j not in visited:
                visited.add(j)
                stack.append(j)
    return False


def monomer_to_repeat_smiles(monomer_smiles: str, strategy: str) -> str:
    """
    Transform a monomer into a polymer repeat-unit SMILES with wildcard endpoints [*].
    strategy ∈ {'vinyl','acrylate','rop_thf','rop_epoxide','ketene',
                'cond_alpha_hydroxy_acid','cond_diphenol', rop_lactam','cond_omega_amino_acid'}
    """
    mol = Chem.MolFromSmiles(monomer_smiles)
    if mol is None:
        raise ValueError("Invalid monomer SMILES.")

    if strategy == "vinyl":
        prod = first_valid_product(mol, RXN_VINYL_HT)
    elif strategy == "acrylate":
        prod = run_acrylate_head_to_tail(mol)
    elif strategy == "alkyne":
        prod = first_valid_product(mol, RXN_ALKYNE_TO_POLYALKENYLENE)
    elif strategy == "rop_thf":
        prod = first_valid_product(mol, RXN_ROP_THF)
    elif strategy == "rop_epoxide":
        prod = first_valid_product(mol, RXN_ROP_EPOXIDE)
    elif strategy == "ketene":
        prod = first_valid_product(mol, RXN_KETENE_TO_PEO)
    elif strategy == "cond_alpha_hydroxy_acid":
        prod = first_valid_product(mol, RXN_COND_ALPHA_HA)
    elif strategy == "rop_lactam":
        # Try specific ε-caprolactam first, then generic lactam
        prod = None
        if Chem.MolFromSmiles(monomer_smiles).HasSubstructMatch(PATT_LACTAM_R7_EXACT):
            prod = first_valid_product(
                Chem.MolFromSmiles(monomer_smiles), RXN_LACTAM_R7_TO_N6
            )
        if prod is None and Chem.MolFromSmiles(monomer_smiles).HasSubstructMatch(
            PATT_LACTAM_GENERIC
        ):
            prod = first_valid_product(
                Chem.MolFromSmiles(monomer_smiles), RXN_LACTAM_GENERIC
            )
    elif strategy == "cond_omega_amino_acid":
        # Try Nylon-6 specific A-B case, then generic ω-amino acid fallback
        mol = Chem.MolFromSmiles(monomer_smiles)
        if mol.HasSubstructMatch(PATT_OMEGA_AMINO_ACID_N6):
            prod = first_valid_product(mol, RXN_OMEGA_AMINO_ACID_N6)
        else:
            prod = first_valid_product(mol, RXN_OMEGA_AMINO_ACID_GENERIC)
    elif strategy == "cond_diphenol":
        # HQ-specific
        prod = first_valid_product(mol, RXN_DIPHENOL_HQ)
        if prod is None:
            if count_phenolic_OH(mol) == 2 and same_aromatic_component(mol):
                tmp = mol
                # Apply one-end replacement twice
                for _ in range(2):
                    p = first_valid_product(tmp, RXN_DIPHENOL_GENERIC_ONE_END)
                    if p is None:
                        break
                    tmp = p
                if count_phenolic_OH(tmp) == 0:
                    prod = tmp

    else:
        raise NotImplementedError(f"Unknown strategy: {strategy}")

    if prod is None:
        raise ValueError(f"No applicable polymerization site found for '{strategy}'.")

    return Chem.MolToSmiles(prod, isomericSmiles=True)


def wrap_bigsmiles_like(smiles_with_wildcards: str) -> str:
    """Optional: wrap the repeat unit in a simple BigSMILES-like block."""
    return "{" + smiles_with_wildcards + "}"


# ============================================================
# ===============  Rule Suggester (no transform)  =============
# ============================================================


@dataclass
class Suggestion:
    strategy: str  # e.g., 'vinyl', 'rop_epoxide', ...
    confidence: float  # 0..1 heuristic score
    reason: str  # short human-readable rationale


# SMARTS detectors (conservative)
PATT_TERMINAL_VINYL = Chem.MolFromSmarts("[CH2]=[C]")  # styrene-like, acrylates too
# β-substituted (meth)acrylate: CH2=C(R)-C(=O)X, where R is any carbon substituent
PATT_METHACRYLATE = Chem.MolFromSmarts("C=C([#6])-[C](=O)[O,N]")
# Unsubstituted acrylate: CH2=CH-C(=O)X
PATT_ACRYLATE = Chem.MolFromSmarts("[CH2]=[CH]-[C](=O)[O,N]")
PATT_EPOXIDE = Chem.MolFromSmarts("[O;X2;R]1[C;R][C;R]1")  # 3-membered cyclic ether
PATT_THF = Chem.MolFromSmarts(
    "[O;X2;R]1[C;R][C;R][C;R][C;R]1"
)  # saturated 5-member cyclic ether
PATT_KETENE = Chem.MolFromSmarts("C=C=O")  # ketene
PATT_ALPHA_HA = Chem.MolFromSmarts("[O;H1]-[C]-C(=O)[O;H1]")  # HO-CH(R)-C(=O)OH
PATT_PHENOL = Chem.MolFromSmarts("[O;H1]-a")  # phenolic OH
PATT_DIOL_1_2_3_ = Chem.MolFromSmarts(
    "[O;H1]-[CX4]-[CX4]-[O;H1]"
)  # e.g., OCCO (ethylene glycol)
# Itaconate diester: CH2=C(-C(=O)OR)C(=O)OR  → vinyl-like addition
PATT_ITACONATE = Chem.MolFromSmarts("[CH2]=[C](-C(=O)O[#6])C(=O)O[#6]")
PATT_LACTAM_R7_EXACT = Chem.MolFromSmarts("O=C1NCCCCC1")  # ε-caprolactam
PATT_LACTAM_GENERIC = Chem.MolFromSmarts("[N;R]-[C;R](=O)")
PATT_OMEGA_AMINO_ACID_N6 = Chem.MolFromSmarts("NCCCCC C(=O)O")  # spaces for readability
PATT_OMEGA_AMINO_ACID_GENERIC = Chem.MolFromSmarts(
    "N-[CH2]-[CH2]-[CH2]-[CH2]-[CH2,$([CH2][CH2])]-C(=O)O"
)
PATT_LACTAM_R7_HINT = Chem.MolFromSmarts(
    "O=C1N@@@@@1"
)  # very permissive “ring amide” hint
PATT_LACTAM_ANY_RING = Chem.MolFromSmarts("[N;R]-[C;R](=O)")
PATT_AB_OMEGA_GENERIC = PATT_OMEGA_AMINO_ACID_GENERIC
# Matches any C≡C (terminal or internal); substituents are preserved automatically.
PATT_ALKYNE_ANY = Chem.MolFromSmarts("[#6]#[#6]")


def suggest_polymerization_rules(monomer_smiles: str) -> List[Suggestion]:
    """
    Inspect a monomer and suggest plausible single-monomer rules (ranked).
    Returns a list of Suggestion(strategy, confidence, reason). Does NOT transform.
    """
    mol = Chem.MolFromSmiles(monomer_smiles)
    if mol is None:
        raise ValueError("Invalid monomer SMILES.")

    suggestions: List[Suggestion] = []
    if mol.HasSubstructMatch(PATT_ITACONATE):
        suggestions.append(
            Suggestion(
                "acrylate",
                0.93,
                "Itaconate diester detected — vinyl (head-to-tail) chain growth at CH2.",
            )
        )

    if mol.HasSubstructMatch(PATT_EPOXIDE):
        suggestions.append(
            Suggestion(
                "rop_epoxide",
                0.95,
                "Epoxide ring detected (3-membered cyclic ether) → ROP to poly(alkylene oxide).",
            )
        )
    if mol.HasSubstructMatch(PATT_LACTAM_ANY_RING):
        suggestions.append(
            Suggestion(
                "rop_lactam",
                0.92,
                "Lactam ring detected → ROP to polyamide (e.g., Nylon-6).",
            )
        )

    if mol.HasSubstructMatch(PATT_AB_OMEGA_GENERIC):
        suggestions.append(
            Suggestion(
                "cond_omega_amino_acid",
                0.88,
                "ω-Amino acid detected → A–B self-condensation to polyamide (nylon-n).",
            )
        )

    if mol.HasSubstructMatch(PATT_THF):
        suggestions.append(
            Suggestion(
                "rop_thf",
                0.85,
                "Tetrahydrofuran-like 5-member cyclic ether detected → ROP to poly(tetramethylene oxide).",
            )
        )
    if mol.HasSubstructMatch(PATT_ALKYNE_ANY):
        suggestions.append(
            Suggestion(
                "alkyne",
                0.88,
                "Alkyne (C≡C) detected → conjugated poly(alkenylene) repeat.",
            )
        )
    if mol.HasSubstructMatch(PATT_ACRYLATE):
        suggestions.append(
            Suggestion(
                "acrylate",
                0.90,
                "α,β-unsaturated carbonyl (acrylate/amide) → chain-growth at C=C.",
            )
        )

    if mol.HasSubstructMatch(PATT_TERMINAL_VINYL):
        suggestions.append(
            Suggestion(
                "vinyl",
                0.70,
                "Terminal vinyl group present; generic chain-growth possible (check selectivity).",
            )
        )

    if mol.HasSubstructMatch(PATT_KETENE):
        suggestions.append(
            Suggestion(
                "ketene", 0.80, "Ketene C=C=O fragment → conceptual oxyethylene repeat."
            )
        )

    if mol.HasSubstructMatch(PATT_ALPHA_HA):
        suggestions.append(
            Suggestion(
                "cond_alpha_hydroxy_acid",
                0.75,
                "α-hydroxy acid pattern → self-condensation polyester (e.g., PLA).",
            )
        )

    phenolic_matches = mol.GetSubstructMatches(PATT_PHENOL)
    if len(phenolic_matches) == 2:
        suggestions.append(
            Suggestion(
                "cond_diphenol",
                0.70,
                "Two phenolic –OH groups detected on an aromatic system → poly(arylene ether) by self-condensation.",
            )
        )

    # Detect plain diol motifs: mark that a comonomer is needed (guidance only)
    if mol.HasSubstructMatch(PATT_DIOL_1_2_3_):
        suggestions.append(
            Suggestion(
                "needs_comonomer",
                0.95,
                "Diol detected (e.g., HO–(CH2)n–OH). Typically requires a co-monomer (e.g., diacid → polyester, dihalide → polyether).",
            )
        )

    # De-duplicate by strategy (keep highest confidence)
    best = {}
    for s in suggestions:
        if s.strategy not in best or s.confidence > best[s.strategy].confidence:
            best[s.strategy] = s

    ranked = sorted(best.values(), key=lambda s: s.confidence, reverse=True)
    return ranked


# ============================================================
# ===============  Auto Selection & Apply  ===================
# ============================================================

# Specificity priority (higher = more specific); used to break ties/overlaps
SPECIFICITY_ORDER = {
    "rop_epoxide": 90,
    "rop_thf": 80,
    "acrylate": 70,
    "alkyne": 72,
    "vinyl": 50,
    "ketene": 60,
    "cond_alpha_hydroxy_acid": 75,
    "cond_diphenol": 65,
    "needs_comonomer": 0,  # guidance only
}


def choose_strategy_auto(
    monomer_smiles: str,
    min_confidence: float = 0.80,
    allow_fallback_to_lower_confidence: bool = True,
) -> Tuple[str, str]:
    """
    Decide which single-monomer rule to apply.
    Returns (strategy, reason). Raises ValueError on ambiguity or 'needs_comonomer'.

    Logic:
      1) Rank suggestions by confidence; filter >= min_confidence.
      2) If none and fallback allowed, consider next tier (>= 0.65).
      3) If multiple remain, pick the one with highest SPECIFICITY_ORDER.
      4) If still ambiguous (same specificity & close scores), raise with details.
      5) If winner is 'needs_comonomer', raise with guidance.
    """
    ranked = suggest_polymerization_rules(monomer_smiles)

    # primary tier
    candidates = [s for s in ranked if s.confidence >= min_confidence]

    # fallback tier
    if not candidates and allow_fallback_to_lower_confidence:
        candidates = [s for s in ranked if s.confidence >= 0.65]

    if not candidates:
        raise ValueError(
            "No suitable single-monomer polymerization rule detected for this monomer."
        )

    # If the top candidate is 'needs_comonomer' and there's nothing else comparable, raise.
    top_conf = candidates[0].confidence
    top_like = [s for s in candidates if abs(s.confidence - top_conf) < 0.05]
    # remove 'needs_comonomer' from contention if there is a real rule with comparable score
    real_rules = [s for s in top_like if s.strategy != "needs_comonomer"]
    if not real_rules and candidates[0].strategy == "needs_comonomer":
        raise ValueError(candidates[0].reason)

    # prefer non-'needs_comonomer'
    candidates = [
        s for s in candidates if s.strategy != "needs_comonomer"
    ] or candidates

    # break ties by specificity
    best_spec = max(SPECIFICITY_ORDER.get(s.strategy, 0) for s in candidates)
    specific = [
        s for s in candidates if SPECIFICITY_ORDER.get(s.strategy, 0) == best_spec
    ]

    # if still more than one, pick highest confidence among those
    specific.sort(key=lambda s: s.confidence, reverse=True)
    winner = specific[0]

    # final ambiguity check: if there are two with same specificity and |Δscore| < 0.03 → ambiguous
    if (
        len(specific) > 1
        and abs(specific[0].confidence - specific[1].confidence) < 0.03
    ):
        names = ", ".join(f"{s.strategy}({s.confidence:.2f})" for s in specific[:3])
        raise ValueError(
            f"Ambiguous polymerization class candidates: {names}. Please specify a strategy."
        )

    return winner.strategy, winner.reason


def monomer_to_repeat_auto(
    monomer_smiles: str,
    min_confidence: float = 0.80,
    allow_fallback_to_lower_confidence: bool = True,
) -> Tuple[str, str, str]:
    """
    Auto-select and apply a single-monomer polymerization rule.
    Returns (repeat_smiles, chosen_strategy, rationale).
    Raises ValueError when comonomer required or ambiguous.
    """
    strategy, reason = choose_strategy_auto(
        monomer_smiles,
        min_confidence=min_confidence,
        allow_fallback_to_lower_confidence=allow_fallback_to_lower_confidence,
    )
    repeat = monomer_to_repeat_smiles(monomer_smiles, strategy=strategy)
    return repeat, strategy, reason


def run_acrylate_head_to_tail(mol: Chem.Mol) -> Chem.Mol | None:
    """Head-to-tail mapping for (meth)acrylates. Try β-substituted first, then unsubstituted."""
    # β-substituted (itaconates, methacrylates, etc.)
    if mol.HasSubstructMatch(PATT_METHACRYLATE):
        prods = RXN_METHACRYLATE_HT.RunReactants((mol,))
        for p in prods:
            try:
                m = Chem.Mol(p[0])
                Chem.SanitizeMol(m)
                return m
            except Exception:
                pass
    # Unsubstituted acrylates
    if mol.HasSubstructMatch(PATT_ACRYLATE):
        prods = RXN_ACRYLATE_HT.RunReactants((mol,))
        for p in prods:
            try:
                m = Chem.Mol(p[0])
                Chem.SanitizeMol(m)
                return m
            except Exception:
                pass
    return None


# ============================================================
# ===============  Example / Smoke Tests  =====================
# ============================================================
if __name__ == "__main__":
    tests = [
        # clear single-monomer cases
        ("C=CC1=CC=CC=C1", "styrene"),
        ("CC(=C)C(=O)OC", "methyl methacrylate"),
        ("C1CCOC1", "THF"),
        ("C1CO1", "ethylene oxide"),
        ("C=C=O", "ketene"),
        ("CC(O)C(=O)O", "lactic acid"),
        ("Oc1ccc(O)cc1", "hydroquinone"),
        ("C=C(CC(=O)OCCCC)C(=O)OCCCC", "itaconate"),
        ("O=C1NCCCCC1", "caprolactam"),
        ("NCCCCCC(=O)O", "nylon"),
        ("C#Cc1ccccc1", "alkyne"),
        ("C#C", "alkyne"),
        # needs comonomer
        ("OCCO", "ethylene glycol"),
    ]

    for s, name in tests:
        try:
            rep, strat, why = monomer_to_repeat_auto(s)
            print(
                f"{name:22s} -> {wrap_bigsmiles_like(rep):28s}  via {strat:22s} | {why}"
            )
        except Exception as e:
            print(f"{name:22s} -> ERROR: {e}")
