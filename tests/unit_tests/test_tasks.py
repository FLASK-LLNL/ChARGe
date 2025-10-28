import json
import pytest


@pytest.mark.parametrize(
    "input_data, expected_output",
    [
        ("User Prompt", "user_prompt"),
        ("   user   prompt   ", "user_prompt"),
        ("User-Prompt", "user_prompt"),
        ("User--Prompt", "user_prompt"),
        ("SyStem PrompT", "system_prompt"),
        ("__System__Prompt__", "system_prompt"),
    ],
)
def test_normalize_string(input_data, expected_output):
    from charge.tasks.Task import normalize_string

    assert normalize_string(input_data) == expected_output


def test_Task_create_no_tools():
    from charge.tasks.Task import Task

    task = Task(
        system_prompt="System Prompt",
        user_prompt="User Prompt",
        verification_prompt="Verification Prompt",
        refinement_prompt="Refinement Prompt",
    )
    assert task.get_system_prompt() == "System Prompt"
    assert task.get_user_prompt() == "User Prompt"
    assert task.has_refinement_prompt() is True
    assert task.get_verification_prompt() == "Verification Prompt"
    assert task.get_refinement_prompt() == "Refinement Prompt"
    assert task.has_structured_output_schema() is False


@pytest.fixture
def task_with_schema():
    from charge.tasks.Task import Task
    from pydantic import BaseModel

    class SampleSchema(BaseModel):
        field1: str
        field2: int

    task = Task(
        system_prompt="System Prompt",
        user_prompt="User Prompt",
        structured_output_schema=SampleSchema,
    )
    return task, SampleSchema


def test_Task_with_structured_output_schema(task_with_schema):
    task, SampleSchema = task_with_schema

    assert task.has_structured_output_schema() is True
    schema = task.get_structured_output_schema()
    assert schema == SampleSchema


def test_Task_check_output_formatting(task_with_schema):
    task, SampleSchema = task_with_schema

    sample_output = SampleSchema(field1="test", field2=42).model_dump_json()
    assert task.check_output_formatting(sample_output) is True


def test_Task_check_output_formatting_invalid(task_with_schema):
    import pydantic_core

    task, SampleSchema = task_with_schema
    invalid_output = json.dumps({"field1": "test", "field2": "not_an_int"})

    with pytest.raises(pydantic_core._pydantic_core.ValidationError):
        SampleSchema.model_validate_json(invalid_output)

    with pytest.warns(UserWarning, match="Output formatting check failed with error:"):
        assert task.check_output_formatting(invalid_output) is False
