"""Template tests for the v1 estimation prompt.

They check what the rendered text contains for each combination of form
parameters — not what the model answers. No network, no LLM: milliseconds.
"""

import pytest
from jinja2 import UndefinedError

from app.prompts import loader
from app.prompts.loader import UnknownPromptVersionError, render_estimation_prompt
from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType

DESCRIPTION = "Marketplace web para que talleres mecánicos publiquen citas disponibles."
ASSUMPTIONS_INSTRUCTION = "list the assumptions for each phase"


def _request(**overrides) -> EstimationRequest:
    fields = {
        "description": DESCRIPTION,
        "project_type": ProjectType.WEB_SAAS,
        "detail_level": DetailLevel.MEDIUM,
        "output_format": OutputFormat.PHASES_TABLE,
    }
    return EstimationRequest(**{**fields, **overrides})


def test_description_is_rendered_verbatim_inside_its_block() -> None:
    _, user = render_estimation_prompt(_request())
    assert f"<project_description>\n{DESCRIPTION}\n</project_description>" in user


def test_description_goes_only_in_the_user_message() -> None:
    system, _ = render_estimation_prompt(_request())
    assert DESCRIPTION not in system


def test_phases_table_format_keyword_only_with_that_format() -> None:
    table_system, _ = render_estimation_prompt(_request(output_format=OutputFormat.PHASES_TABLE))
    narrative_system, _ = render_estimation_prompt(_request(output_format=OutputFormat.NARRATIVE))

    assert "phases_table" in table_system
    assert "confidence_pct" in table_system
    # Also checks the few-shot examples follow the requested format.
    assert "phases_table" not in narrative_system
    assert "confidence_pct" not in narrative_system


def test_detailed_level_adds_assumptions_per_phase() -> None:
    detailed_system, _ = render_estimation_prompt(_request(detail_level=DetailLevel.DETAILED))
    summary_system, _ = render_estimation_prompt(_request(detail_level=DetailLevel.SUMMARY))

    assert ASSUMPTIONS_INSTRUCTION in detailed_system
    assert ASSUMPTIONS_INSTRUCTION not in summary_system


@pytest.mark.parametrize("project_type", list(ProjectType))
@pytest.mark.parametrize("detail_level", list(DetailLevel))
@pytest.mark.parametrize("output_format", list(OutputFormat))
def test_every_combination_renders_with_examples(
    project_type: ProjectType, detail_level: DetailLevel, output_format: OutputFormat
) -> None:
    system, user = render_estimation_prompt(
        _request(project_type=project_type, detail_level=detail_level, output_format=output_format)
    )
    assert f"## Output format: {output_format.value}" in system
    assert "--- EXAMPLE 3:" in system
    assert "{{" not in system and "{%" not in system
    assert user


def test_unknown_version_is_rejected() -> None:
    with pytest.raises(UnknownPromptVersionError, match="v99"):
        render_estimation_prompt(_request(), version="v99")
    # Only directories on disk are accepted, so path tricks never reach Jinja.
    with pytest.raises(UnknownPromptVersionError):
        render_estimation_prompt(_request(), version="../estimation/v1")


def test_missing_template_variable_fails_loudly() -> None:
    template = loader._env.from_string("Estimate {{ descripton }}")  # deliberate typo
    with pytest.raises(UndefinedError):
        template.render(description=DESCRIPTION)
