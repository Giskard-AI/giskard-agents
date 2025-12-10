import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel

from giskard.agents.templates import MessageTemplate, PromptsManager


@pytest.fixture
def prompts_manager():
    return PromptsManager(
        default_prompts_path=Path(__file__).parent / "data" / "prompts"
    )


async def test_message_template():
    template = MessageTemplate(
        role="user",
        content_template="Hello, {{ name }}!",
    )

    message = template.render(name="Orlande de Lassus")

    assert message.role == "user"
    assert message.content == "Hello, Orlande de Lassus!"


async def test_multi_message_template_parsing(prompts_manager):
    messages = await prompts_manager.render_template(
        "multi_message.j2",
        {
            "theory": "Normandy is actually the center of the universe because its perfect balance of rain, cheese, and cider creates a quantum field that bends space-time."
        },
    )

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert (
        "You are an impartial evaluator of scientific theories" in messages[0].content
    )


async def test_invalid_template(prompts_manager):
    with pytest.raises(ValueError):
        await prompts_manager.render_template("invalid.j2")


async def test_simple_template(prompts_manager):
    messages = await prompts_manager.render_template("simple.j2")

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert (
        messages[0].content
        == "This is a simple prompt that should be rendered as a single user message."
    )


def test_pydantic_json_rendering_inline():
    class Book(BaseModel):
        title: str
        description: str

    template = MessageTemplate(
        role="user",
        content_template="Hello, consider this content:\n{{ book }}!",
    )

    book = Book(
        title="The Great Gatsby",
        description="The Great Gatsby is a novel by F. Scott Fitzgerald.",
    )

    message = template.render(book=book)

    assert message.role == "user"
    expected_json = """{
    "title": "The Great Gatsby",
    "description": "The Great Gatsby is a novel by F. Scott Fitzgerald."
}"""
    assert message.content == f"Hello, consider this content:\n{expected_json}!"


async def test_pydantic_json_rendering_with_prompts_manager():
    class Book(BaseModel):
        title: str
        description: str

    book = Book(
        title="The Great Gatsby",
        description="The Great Gatsby is a novel by F. Scott Fitzgerald.",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        prompts_manager = PromptsManager(default_prompts_path=tmp_dir)

        template_path = Path(tmp_dir) / "book.j2"
        template_path.write_text("Here is a book:\n{{ book }}")

        messages = await prompts_manager.render_template("book.j2", {"book": book})

        assert len(messages) == 1
        assert messages[0].role == "user"
        expected_json = """{
    "title": "The Great Gatsby",
    "description": "The Great Gatsby is a novel by F. Scott Fitzgerald."
}"""
        assert messages[0].content == f"Here is a book:\n{expected_json}"
