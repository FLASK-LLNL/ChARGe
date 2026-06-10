from charge.clients.agentframework import AgentFrameworkAgent
from charge.tasks.task import Task


def test_agentframework_prepares_multimodal_prompt_for_attachments():
    task = Task(
        user_prompt="Describe this image.",
        attachments=[
            {
                "id": "image-1",
                "name": "sample.png",
                "mimeType": "image/png",
                "sizeBytes": 68,
                "dataUrl": "data:image/png;base64,ZmFrZQ==",
                "createdAt": "2026-05-22T00:00:00Z",
            }
        ],
    )
    agent = AgentFrameworkAgent(
        task=task,
        client=object(),
        agent_key="test",
        model="test-model",
    )

    prompt = agent._prepare_task_prompt()

    assert isinstance(prompt, list)
    assert prompt[0].type == "text"
    assert prompt[0].text == "Describe this image."
    assert prompt[1].type == "data"
    assert prompt[1].media_type == "image/png"
    assert prompt[1].uri == "data:image/png;base64,ZmFrZQ=="


def test_agentframework_signature_does_not_include_turn_attachments():
    task = Task(
        system_prompt="System",
        user_prompt="Describe this image.",
        attachments=[
            {
                "id": "image-1",
                "name": "sample.png",
                "mimeType": "image/png",
                "sizeBytes": 68,
                "dataUrl": "data:image/png;base64,ZmFrZQ==",
                "createdAt": "2026-05-22T00:00:00Z",
            }
        ],
    )
    agent = AgentFrameworkAgent(
        task=task,
        client=object(),
        agent_key="test",
        model="test-model",
    )

    signature_with_attachment = agent._get_agent_signature()
    task.attachments = []

    assert agent._get_agent_signature() == signature_with_attachment
