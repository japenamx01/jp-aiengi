"""Template tests for the v3 prompt: project metadata block and attachments."""

import re

from app.prompts.loader import render_estimation_prompt, render_session_prompt
from app.schemas.estimation import EstimationRequest
from app.schemas.session import SessionTurn
from app.services.attachments import ExtractedAttachment
from app.services.sessions import ProjectMetadata

TURN = SessionTurn(transcript="Add offline mode to the app.")
METADATA_BLOCK = re.compile(r"<project_metadata>\n(.*?)</project_metadata>", re.DOTALL)


def _metadata_block(system: str) -> str:
    match = METADATA_BLOCK.search(system)
    assert match, "the <project_metadata> block must always be present"
    return match.group(1)


def test_empty_metadata_renders_an_empty_block() -> None:
    system, _ = render_session_prompt(TURN, ProjectMetadata())
    assert _metadata_block(system) == ""


def test_metadata_facts_are_rendered_only_when_known() -> None:
    metadata = ProjectMetadata(
        project_name="VetCare", mentioned_technologies=["React Native", "Stripe"]
    )

    block = _metadata_block(render_session_prompt(TURN, metadata)[0])

    assert "- Project name: VetCare" in block
    assert "- Mentioned technologies: React Native, Stripe" in block
    assert "team size" not in block
    assert "Agreed scope" not in block


def test_attachments_are_appended_with_clear_separators() -> None:
    attachments = [
        ExtractedAttachment(filename="spec.pdf", text="Must support SSO.", truncated=False),
        ExtractedAttachment(filename="notes.docx", text="Budget 20k.", truncated=False),
    ]

    _, user = render_session_prompt(TURN, ProjectMetadata(), attachments)

    assert user.startswith(
        "<project_description>\nAdd offline mode to the app.\n</project_description>"
    )
    assert (
        "--- attachment: spec.pdf ---\nMust support SSO.\n--- end of attachment: spec.pdf ---"
        in user
    )
    assert user.index("spec.pdf") < user.index("notes.docx")


def test_without_attachments_there_are_no_markers() -> None:
    _, user = render_session_prompt(TURN, ProjectMetadata())
    assert "--- attachment" not in user


def test_v3_also_renders_without_a_session() -> None:
    request = EstimationRequest(
        description="Booking app for padel courts with payments.",
        project_type="mobile_app",
        detail_level="summary",
        output_format="narrative",
    )

    system, user = render_estimation_prompt(request, version="v3")

    assert _metadata_block(system) == ""
    assert "executive summary" in system
    assert "{{" not in system and "{%" not in system and "#}" not in system
    assert "--- attachment" not in user
