from charge.tasks.task import Task


def test_task_serializes_attachments():
    attachments = [
        {
            "id": "image-1",
            "name": "molecule.png",
            "mimeType": "image/png",
            "sizeBytes": 68,
            "dataUrl": "data:image/png;base64,ZmFrZQ==",
            "createdAt": "2026-05-22T00:00:00Z",
        }
    ]
    task = Task(system_prompt="System", user_prompt="User", attachments=attachments)

    restored = Task.from_json(task.to_json())

    assert restored.attachments == attachments
