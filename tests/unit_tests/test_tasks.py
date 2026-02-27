import json
import pytest
import requests_mock


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
    from charge.utils.system_utils import normalize_string

    assert normalize_string(input_data) == expected_output


def test_Task_create_no_tools():
    from charge.tasks.task import Task

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
    from charge.tasks.task import Task
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


def test_Task_no_url_raise_error():
    from charge.tasks.task import Task

    # When CHARGE_RAISE_ON_MISSING_SERVER is set to True, ValueError should be raised
    pytest.MonkeyPatch().setenv("CHARGE_ERROR_ON_MISSING_SERVER", "1")
    with pytest.raises(ValueError) as excinfo:
        with pytest.warns(UserWarning):
            task = Task(
                system_prompt="System Prompt",
                user_prompt="User Prompt",
                server_urls="http://non_existent_server_url/sse",
            )

            assert task.server_urls == []  # No valid server URLs found

    # When CHARGE_RAISE_ON_MISSING_SERVER is set to False, a warning should be issued
    pytest.MonkeyPatch().setenv("CHARGE_ERROR_ON_MISSING_SERVER", "0")
    with pytest.warns(UserWarning):
        task = Task(
            system_prompt="System Prompt",
            user_prompt="User Prompt",
            server_urls="http://non_existent_server_url/sse",
        )

        assert task.server_urls == []  # No valid server URLs found


def test_Task_with_valid_and_invalid_urls():
    from charge.tasks.task import Task

    valid_url = "http://valid_server_url/sse"
    invalid_url = "http://invalid_server_url/sse"

    with requests_mock.Mocker() as m:
        m.get(valid_url, status_code=200)
        m.get(invalid_url, status_code=404)

        with pytest.warns(UserWarning):
            task = Task(
                system_prompt="System Prompt",
                user_prompt="User Prompt",
                server_urls=[valid_url, invalid_url],
            )

            assert task.server_urls == [
                valid_url
            ]  # Only the valid URL should be retained
