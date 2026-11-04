"""v2 differs from v1 in one deliberate way: a client-facing tone."""

from app.prompts.loader import available_versions, render_estimation_prompt
from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType

REQUEST = EstimationRequest(
    description="Portal interno para que el equipo de ventas registre visitas a clientes.",
    project_type=ProjectType.INTERNAL_TOOL,
    detail_level=DetailLevel.MEDIUM,
    output_format=OutputFormat.LINE_ITEMS,
)


def test_both_versions_are_available() -> None:
    assert {"v1", "v2"} <= set(available_versions())


def test_v2_adds_client_facing_tone_and_keeps_the_rest() -> None:
    v1_system, v1_user = render_estimation_prompt(REQUEST, version="v1")
    v2_system, v2_user = render_estimation_prompt(REQUEST, version="v2")

    assert "executive summary" in v2_system
    assert "executive summary" not in v1_system
    assert "business owner" in v2_system
    # Same user message and same examples: only the instructions changed.
    assert v1_user == v2_user
    assert (
        v1_system.split("## Reference estimations")[1]
        == v2_system.split("## Reference estimations")[1]
    )
