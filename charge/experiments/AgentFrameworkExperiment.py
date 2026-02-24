################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from charge.experiments import Experiment
from charge.clients.AgentPool import Agent, AgentPool
from charge.tasks.Task import Task
from typing import List, Union, Optional

try:
    from charge.clients.agentframework import AgentFrameworkPool
    from charge.clients.agentframework_utils import AgentFrameworkMemory
except ImportError:
    raise ImportError(
        "The agent-framework package is required for AgentFrameworkExperiment. "
        "Please install it via 'pip install charge[agentframework]'."
    )


class AgentFrameworkExperiment(Experiment):
    """
    Experiment implementation for Agent Framework.

    This class manages sequential multi-task execution with shared conversation context
    using Agent Framework agents and sessions.

    Args:
        task: Single task or list of tasks to execute sequentially.
        agent_pool: AgentFrameworkPool for creating agents.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        task: Optional[Union[Task, List[Task]]],
        agent_pool: AgentFrameworkPool,
        *args,
        **kwargs,
    ):
        super().__init__(task=task, agent_pool=agent_pool, *args, **kwargs)
        # Initialize Agent Framework specific parameters
        self.model_context = AgentFrameworkMemory()
        self.agent_pool: AgentFrameworkPool

    def save_agent_state(self, agent: Agent):
        """
        Save the state of an Agent Framework agent.

        Args:
            agent: The agent whose state to save.

        Returns:
            None (state is managed via model_context)
        """
        # Agent Framework agents are stateless by default
        # State is managed through AgentSession which is tracked in model_context
        pass

    async def add_to_context(self, agent: Agent, task: Task, result):
        """
        Add the result to the shared experiment context.

        This allows subsequent tasks to access the results of previous tasks,
        enabling multi-turn reasoning across different agents.

        Args:
            agent: The agent that produced the result.
            task: The task that was executed.
            result: The result from the task execution.
        """
        instruction = task.get_user_prompt()
        result_str = str(result)

        # Format as instruction-response pair for context
        content = f"Instruction: {instruction}\nResponse: {result_str}"

        name = getattr(agent, "agent_name", "Agent")
        await self.model_context.add(content, source_agent=name)

    async def save_state(self) -> str:
        """
        Save the complete state of the experiment.

        This includes all conversation history, task results, and agent states
        that have been accumulated during the experiment execution.

        Returns:
            str: JSON string representation of the experiment state.
        """
        # Serialize the memory content which contains all task results and context
        serialized_memory = self.model_context.serialize_memory_content()
        return serialized_memory

    async def load_state(self, state: str):
        """
        Load a previously saved experiment state.

        This allows resuming an experiment from a checkpoint or replaying
        a previous experiment execution.

        Args:
            state: JSON string representation of the experiment state.
        """
        # Create fresh memory and load the state
        self.model_context = AgentFrameworkMemory()
        self.model_context.load_memory_content(state)

    def create_agent_with_experiment_state(self, task: Task, **kwargs) -> Agent:
        """
        Create an agent with access to the shared experiment state.

        This ensures that the agent can access the conversation history and
        results from previous tasks in the experiment.

        Args:
            task: The task for the agent to execute.
            **kwargs: Additional keyword arguments for agent creation.

        Returns:
            Agent: The created agent with experiment state.
        """
        # Create agent with shared memory for context continuity
        return self.agent_pool.create_agent(
            task=task, memory=[self.model_context], **kwargs
        )

    def reset(self):
        """
        Reset the experiment state, clearing all accumulated context.

        This is useful when starting a new experiment or rerunning tasks
        from a clean slate.
        """
        super().reset()
        self.model_context = AgentFrameworkMemory()

    def get_conversation_history(self) -> List[dict]:
        """
        Get the complete conversation history from the experiment.

        Returns:
            List[dict]: List of message dictionaries with content and metadata.
        """
        return self.model_context.get_messages()

    def get_source_agents(self) -> List[str]:
        """
        Get the list of agents that contributed to the conversation.

        Returns:
            List[str]: List of agent names in order of their contributions.
        """
        return self.model_context.get_source_agents()
