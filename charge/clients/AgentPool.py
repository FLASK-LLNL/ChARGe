from charge.tasks.Task import Task


class Agent:
    def __init__(self, task: Task, **kwargs):
        self.task = task
        self.kwargs = kwargs

    def run(self, **kwargs):
        raise NotImplementedError("Method 'run' is not implemented.")


class AgentPool:
    def __init__(self):
        pass

    def create_agent(
        self,
        task: Task,
        **kwargs,
    ):
        raise NotImplementedError("Method 'create_agent' is not implemented.")
