from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from charge.tasks.task import Task


class Memory(ABC):
    """Abstract base class for memory storage in experiments."""

    @abstractmethod
    def add_to_context(self, task: "Task", item: Any) -> None:
        """Add an item to the memory context."""
        pass

    @abstractmethod
    def to_json(self) -> Dict[str, Any]:
        """Serialize memory to JSON-compatible dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, data: Dict[str, Any]) -> "Memory":
        """Deserialize memory from JSON-compatible dictionary."""
        pass

    @abstractmethod
    def to_list_of_tasks_and_results(self) -> list[tuple["Task", str]]:
        """Returns a list of tasks and corresponding results as strings."""
        pass


class ListMemory(Memory):
    """Simple memory implementation using a list of strings."""

    def __init__(self, items: Optional[list[tuple["Task", str]]] = None):
        self.items: list[tuple["Task", str]] = items if items is not None else []

    def add_to_context(self, task: "Task", item: str) -> None:
        """Add a string item to the memory list."""
        self.items.append((task, str(item)))

    def to_json(self) -> Dict[str, Any]:
        """Serialize memory to JSON-compatible dictionary."""
        # Serialize each (task, result) tuple
        serialized_items = []
        for task, result in self.items:
            serialized_items.append({"task": task.to_json(), "result": result})
        return {"items": serialized_items}

    def to_list_of_tasks_and_results(self) -> list[tuple["Task", str]]:
        return self.items

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ListMemory":
        """Deserialize memory from JSON-compatible dictionary."""
        from charge.tasks.task import Task
        import importlib

        items = []
        for item_data in data.get("items", []):
            # Deserialize the task
            task_data = item_data["task"]
            module_name = task_data.get("module")
            class_name = task_data.get("class_name")

            # Dynamically import the task class
            if module_name and class_name:
                module = importlib.import_module(module_name)
                task_class = getattr(module, class_name)
                task = task_class.from_json(task_data)
            else:
                # Fallback to base Task class
                task = Task.from_json(task_data)

            result = item_data["result"]
            items.append((task, result))

        return cls(items=items)
