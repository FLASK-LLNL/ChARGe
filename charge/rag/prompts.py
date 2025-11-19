from dataclasses import dataclass


class ReactionDataPrompt:
    def __init__(self, forward : bool) -> None:
        """
        Args:
            forward (bool): whether the prompt is for forward synthesis context
        """
        self.sections = dict()
        self.sections['role'] = 'You are an expert chemist.'
        self.sections['task'] = 'Forward synthesis prediction' if forward else 'Retrosynthesis prediction'
        self.sections['instruction'] = \
            "Given the input data in [INPUT DATA], perform your task and make your predictions, which must follow [OUTPUT FORMAT]."
        self.sections['output format'] = ''.join([
            "Each prediction must be a JSON string on a newline, formatted as: ",
            '{\"products\": [...]}' if forward else '{\"reactants\": [...], \"agents\": [...], \"solvents\": [...]}',
            ", ",
            "where [...] is a list of SMILES strings. ",
            "In case of forward synthesis the products can often be a list of a single molecule. ",
            "In case of retrosynthesis the agents and the solvents can be absent."
        ])

    def to_string(self) -> str:
        strings = []
        for k, v in self.sections.items():
            section_title = f'[{k.upper()}]'
            strings.append(section_title + '\n' + str(v) + '\n\n')
        return ''.join(strings)

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        sections_str = ', '.join([f'{k} = {v}' for k, v in self.sections.items()])
        return f'{self.__class__.__name__}({sections_str})'


class ReactionDataPrompt_RAG(ReactionDataPrompt):
    def __init__(self, forward: bool) -> None:
        super().__init__(forward=forward)
        self.sections['instruction'] = (
            "Given the input data in [INPUT DATA], a list of similar reactions is provided in [SIMILAR REACTIONS].\n"
            "Consider these similar reactions and make you prediction, which must follow [OUTPUT FORMAT]."
        )
        self.sections['similar reactions'] = ''


class ReactionDataPrompt_RAGv2(ReactionDataPrompt):
    def __init__(self, forward: bool) -> None:
        super().__init__(forward=forward)
        input_role = 'reactants' if forward else 'products'
        output_role = 'products' if forward else 'reactants'
        self.sections['instruction'] = ''.join([
            "Given the input data in [INPUT DATA], a list of expert predictions is provided in [EXPERT PREDICTIONS], ",
            "and a list of support examples is provided in [SUPPORT EXAMPLES].\n",
            f"Each line in [EXPERT PREDICTIONS] is a prediction for [INPUT DATA], and is based on an expert chemical reaction model.\n",
            "Each line in [SUPPORT EXAMPLES] consists of three columns:\n",
            f"(1) {input_role} similar to the input data, retrieved from a chemical reaction database,\n",
            f"(2) ground truth {output_role} for column (1), retrieved from the same database,\n",
            f"(3) expert-predicted {output_role} for column (1), based on the same expert model used in [EXPERT PREDICTIONS].\n",
            "Learn from the pattern in [SUPPORT EXAMPLES] to correct or improve [EXPERT PREDICTIONS]. ",
            "If none of the [EXPERT PREDICTIONS] are chemically plausible or consistent with the learned patterns, ",
            "construct a corrected or hybrid prediction guided by [SUPPORT EXAMPLES] and your chemistry expertise. ",
            "Your prediction output must follow [OUTPUT FORMAT].",
        ])
        self.sections['expert predictions'] = ''
        self.sections['support examples'] = ''


class ReactionDataPrompt_RAGv3(ReactionDataPrompt):
    def __init__(self, forward: bool) -> None:
        super().__init__(forward=forward)
        self.sections['instruction'] = ''.join([
            "You are given a data table in [DATA TABLE]. In this table, each row/line consists of three columns: \n",
            "(1) reaction data input,\n",
            "(2) ground truth output retrieved from a database,\n",
            "(3) predicted output from an expert chemical reaction model.\n",
            "In one line the ground truth output is missing as denoted by '???'. ",
            "Make several predictions (e.g., 3 to 5 different predictions) for this missing value. ",
            "Your predictions must follow [OUTPUT FORMAT]. ",
            "Observe the pattern in the table and the expert predictions as guidance. ",
            "Note that the expert predictions may or may not be biased.",
        ])