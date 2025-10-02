import charge
from charge.Experiment import Experiment
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer


class LMOExperiment(Experiment):
    def __init__(
        self,
        system_prompt,
        user_prompt,
        verification_prompt=None,
        refinement_prompt=None,
    ):
        super().__init__(
            system_prompt,
            user_prompt,
            verification_prompt,
            refinement_prompt,
        )
        print("LMOExperiment initialized with the provided prompts.")

        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.verification_prompt = verification_prompt
        self.refinement_prompt = refinement_prompt

    @charge.hypothesis
    def canonicalize_smiles(self, smiles: str) -> str:
        """
        Canonicalize a SMILES string.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol)
        except Exception as e:
            return smiles

    @charge.hypothesis
    def verify_smiles(self, smiles: str) -> bool:
        """
        Verify if a SMILES string is valid.
        """
        try:
            Chem.MolFromSmiles(smiles)
            return True
        except Exception as e:
            return False

    @charge.hypothesis
    def get_synthesizability(self, smiles: str) -> float:
        """
        Calculate the synthesizability of a molecule given its SMILES string.
        Values range from 1.0 (highly synthesizable) to 10.0 (not synthesizable).
        """

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 10.0  # Default value for invalid SMILES
            score = sascorer.calculateScore(mol)
            return score
        except Exception as e:
            return 10.0
