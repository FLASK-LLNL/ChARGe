# ChARGe
**Ch**emical tool **A**ugmented **R**easoning models for **Ge**nerating molecules and reactions

## Installation

To install the package, clone the repository and run:
```bash
pip install -e .
charge-install --extras all
```

Or install directly from GitHub:
```bash
pip install git+https://github.com/FLASK-LLNL/ChARGe.git
charge-install --extras all
```


## Running an Experiment

```python

import argparse
from charge.clients import GeminiClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-prompt", type=str, default="Generate a drug-like molecule")
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    args = parser.parse_args()

    myexperiment = LeadMoleculeOptimization(
        system_prompt=args.system_prompt,
        hypothesis_prompt=args.user_prompt
    )

    runner = GeminiClient(experiment_type=myexperiment)
    results = runner.run()
    print(results)

    while True:
        cont = input("Additional refinement? ...")
        results = runner.refine(cont)
        print(results)

```

## Defining an Experiment

```python
from charge.experiments import Experiment


class LeadMoleculeOptimization(Experiment):
    def __init__(
        self,
        system_prompt,
        hypothesis_prompt=None,
        verification_prompt=None,
        refinement_prompt=None,
    ):
        super().__init__(
            system_prompt, hypothesis_prompt, verification_prompt, refinement_prompt
        )
        self._experiment_global_variable = "Put useful info like API endpoints here"
        self._min_density = 0.8
        self._max_sascore = 1.2

    @hypothesis
    def verifySMILES(self, smiles_string: str) -> bool:
        """ Given a SMILES string, verify if it is valid. 
            Return True if valid, False otherwise.
            Arguments:
                smiles_string: A string representing a molecule in SMILES format.
            Returns:
                bool: True if the SMILES string is valid, False otherwise.
        """
        return isValidSMILES(smiles_string)
    
    @hypothesis
    def calculateSAScore(self, smiles_string: str) -> float:
        """ 
            Given a SMILES string, calculate the synthetic accessibility score (SAScore).
            Arguments:
                smiles_string: A string representing a molecule in SMILES format.
            Returns:
                float: The SAScore of the molecule.
        """
        return calculateSAScore(smiles_string)
    
    @hypothesis
    def calculateDensity(self, smiles_string: str) -> float:
        """ 
            Given a SMILES string, calculate the density of the molecule.
            Arguments:
                smiles_string: A string representing a molecule in SMILES format.
            Returns:
                float: The density of the molecule.
        """
        return calculateDensity(smiles_string)
    
    @verification
    def verifyMolecule(self, smiles_string: str) -> bool:
        """ 
            Given a SMILES string, verify if the molecule meets the criteria to
            be a solution.
            Arguments:
                smiles_string: A string representing a molecule in SMILES format.
            Returns:
                bool: True if the molecule meets the criteria, False otherwise.
            
        """
        density = self.calculateDensity(smiles_string)
        sascore = self.calculateSAScore(smiles_string)
        if density >= self._min_density and sascore <= self._max_sascore:
            return True
        return False

```

# License

Copyright (c) 2025, Lawrence Livermore National Security, LLC. and Binghamton University.

Produced at the Lawrence Livermore National Laboratory and Binghamton University.

SPDX-License-Identifier: Apache-2.0

LLNL-CODE-2006345
