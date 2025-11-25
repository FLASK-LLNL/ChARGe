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
        self.sections['instruction'] = ''.join([
            "Given the input data in [INPUT DATA], perform your task and make several predictions (e.g., 3 to 5 different predictions), ",
            "which must follow [OUTPUT FORMAT]."
        ])
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


class ReactionDataPrompt_ExpertOnly(ReactionDataPrompt):
    def __init__(self, forward: bool) -> None:
        super().__init__(forward=forward)
        self.sections['instruction'] = ''.join([
            "Given the input data in [INPUT DATA], a predicted output is provided in [EXPERT PREDICTION], ",
            "which is based on an expert chemical reaction model. ",
            "Consider this expert prediction and make your own predictions (e.g., 3 to 5 different predictions). ",
            "Your predictions must follow [OUTPUT FORMAT].",
        ])


class ReactionDataPrompt_CopyExpert(ReactionDataPrompt):
    def __init__(self, forward: bool) -> None:
        super().__init__(forward=forward)
        self.sections['instruction'] = ''.join([
            "Given the input data in [INPUT DATA], a list of predicted outputs is provided in [EXPERT PREDICTION], ",
            "which is based on an expert chemical reaction model. ",
            "In your response, please simply copy these expert predictions, which should already follow [OUTPUT FORMAT].",
        ])


class ReactionDataPrompt_RAG(ReactionDataPrompt):
    def __init__(self, forward: bool) -> None:
        super().__init__(forward=forward)
        self.sections['instruction'] = ''.join([
            "You are given a data table in [DATA TABLE]. In this table, each row/line consists of two columns: \n",
            "(1) reaction data input,\n",
            "(2) ground truth output retrieved from a database.\n",
            "In one line the ground truth output is missing as denoted by '???'. ",
            "Make several predictions (e.g., 3 to 5 different predictions) for this missing value. ",
            "Your predictions must follow [OUTPUT FORMAT]. ",
            "Observe the pattern in the table as guidance for your predictions.",
        ])


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


class ReactionDataPrompt_RAGv4(ReactionDataPrompt):
    """
    RAGv4 builds on RAGv3 by:
      - Adding a per-row Neighbor Distance column in [DATA TABLE]
      - Defining Neighbor Distance explicitly as the distance between each row's input and the target row's input
      - Including that definition directly within the instruction text
      - Maintaining backward compatibility with previous output formats
    """
    def __init__(self, forward: bool) -> None:
        super().__init__(forward=forward)

        neighbor_distance_definition = (
            "The Neighbor Distance is a non-negative value representing the distance between this row's input "
            "and the target row's input (the row with '???'). Smaller distances indicate greater similarity, "
            "and 0 represents the closest possible match under the chosen similarity metric."
        )

        self.sections['instruction'] = ''.join([
            "You are given a data table in [DATA TABLE]. Each row has four columns:\n",
            "(1) reaction data input,\n",
            "(2) ground truth output retrieved from a database,\n",
            "(3) predicted output from an expert chemical reaction model,\n",
            "(4) Neighbor Distance between this row's input and the input in the final row (the row with '???').\n",
            "Exactly one row has the ground truth output missing, denoted by '???'. ",
            "Make several predictions (e.g., 3 to 5 different predictions) for this missing value. ",
            "Your predictions must follow [OUTPUT FORMAT]. ",
            "Use the expert predictions and the examples in the table as guidance, but note that expert predictions may be biased. ",
            "Use the Neighbor Distance column to prioritize the most relevant rows when inferring patterns.\n\n",
            f"{neighbor_distance_definition}\n\n",
            "Guidelines for using the table:\n",
            "- Treat smaller Neighbor Distances as stronger evidence; 0 represents the closest possible match under the similarity metric.\n",
            "- Favor patterns that recur among multiple rows with small Neighbor Distances.\n",
            "- Cross-check expert predictions against nearby (low-distance) ground truths to detect and correct biases.\n",
            "- If no directly consistent pattern is found, synthesize a corrected or hybrid prediction using nearest-neighbor ground truths, ",
            "chemical reasoning, and your expertise.",
        ])