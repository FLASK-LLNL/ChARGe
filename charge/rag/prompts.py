from dataclasses import dataclass


@dataclass
class ReactionDataPrompt:
    role: str
    task: str
    instruction: str
    output_format: str
    input_data: str

    def __init__(self, forward : bool) -> None:
        """
        Args:
            forward (bool): whether the prompt is for forward synthesis context
        """
        self.role = 'You are an expert chemist.'
        self.task = 'Forward synthesis prediction' if forward else 'Retrosynthesis prediction'
        self.instruction = \
            "Given the input data in [INPUT DATA], perform your task and make your prediction, which must follow [OUTPUT FORMAT]."
        self.output_format = ''.join([
            "A single line of JSON string formatted as:\n",
            '{"products": [...]}' if forward else '{"reactants": [...], "reagents": [...], "solvents": [...]}',
            ",\n",
            "where [...] is a list of SMILES strings.",
        ])
        self.input_data = ''

    def to_string(self) -> None:
        strings = []
        for k, v in self.__dict__.items():
            section_title = f"[{k.replace('_', ' ').upper()}]"
            strings.append(section_title + '\n' + str(v) + '\n\n')
        return ''.join(strings)

    def __str__(self) -> None:
        return self.to_string()


@dataclass
class ReactionDataPrompt_RAG(ReactionDataPrompt):
    similar_reactions: str
    
    def __init__(self, forward: bool) -> None:
        super().__init__(forward=forward)
        self.instruction = (
            "Given the input data in [INPUT DATA], a list of similar reactions is provided in [SIMILAR REACTIONS].\n"
            "Consider these similar reactions and make you prediction, which must follow [OUTPUT FORMAT]."
        )
        self.similar_reactions = ''


@dataclass
class ReactionDataPrompt_RAGv2(ReactionDataPrompt):
    expert_predictions: str
    support_data: str
    
    def __init__(self, forward: bool) -> None:
        super().__init__(forward=forward)
        input_role = 'reactants' if forward else 'products'
        output_role = 'products' if forward else 'reactants'
        self.instruction = ''.join([
            "Given the input data in [INPUT DATA], a list of expert predictions is provided in [EXPERT PREDICTIONS], ",
            "and a list of support examples is provided in [SUPPORT EXAMPLES].\n",
            f"Each line in [EXPERT PREDICTIONS] is expert-predicted {output_role} for the input data based on an expert chemical reaction model.\n",
            "Each line in [SUPPORT EXAMPLES] consists of three columns:\n",
            f"(1) {input_role} similar to the input data, retrieved from a chemical reaction database,\n",
            f"(2) ground truth {output_role} for column (1), retrieved from the same database,\n",
            f"(3) expert-predicted {output_role} for column (1), based on the same expert model used in [EXPERT PREDICTIONS].\n",
            "Learn from the pattern in [SUPPORT EXAMPLES] to correct or improve [EXPERT PREDICTIONS]. ",
            "If none of the [EXPERT PREDICTIONS] are chemically plausible or consistent with the learned patterns, ",
            "construct a corrected or hybrid prediction guided by [SUPPORT EXAMPLES] and your chemistry expertise, ",
            "Your prediction output must follow [OUTPUT FORMAT].",
        ])
        self.expert_predictions = ''
        self.support_data = ''