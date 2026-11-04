"""Template tests for the v1 estimation prompt.

They check what the rendered text contains for each combination of form
parameters — not what the model answers. No network, no LLM: milliseconds.

NOTA: qué cubre un test de template (y qué NO)
----------------------------------------------
SÍ cubre: "dado este formulario, ¿el texto que mandamos al modelo es el que
queremos?". Es lógica nuestra (los if/include del .j2) y es determinista:
mismo input → mismo texto, siempre. Por eso se puede testear como cualquier
función, sin mocks ni red.

NO cubre: si el modelo HACE CASO al prompt (que devuelva la tabla, que acierte
las horas, que responda en español). Eso no es determinista y depende del
proveedor; se mide con evals y guardrails, no con tests unitarios.

Patrón que se repite abajo: renderizar dos variantes y comprobar que una
contiene algo y la otra NO. Probar la ausencia es tan importante como probar
la presencia: atrapa el bug típico de un {% endif %} mal colocado que deja
pasar un bloque a todas las variantes.
"""

import pytest
from jinja2 import UndefinedError

from app.prompts import loader
from app.prompts.loader import UnknownPromptVersionError, render_estimation_prompt
from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType

DESCRIPTION = "Marketplace web para que talleres mecánicos publiquen citas disponibles."
ASSUMPTIONS_INSTRUCTION = "list the assumptions for each phase"


# NOTA: helper con valores por defecto; cada test sobrescribe solo el campo
# que le interesa. Así queda claro qué parámetro se está probando.
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


# NOTA: comprueba la separación system/user: los datos del usuario no deben
# mezclarse con las instrucciones.
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


# NOTA: tres parametrize apilados = producto cartesiano: pytest genera
# 4 × 3 × 3 = 36 tests a partir de esta única función. Es una red de seguridad
# barata: si una combinación rompe el template (variable inexistente, include
# roto), salta aquí antes de llegar a producción.
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
    # NOTA: si queda "{{", "{%" o "#}" en el texto final, algo no se renderizó
    # bien. Caso real: un comentario que contenía otro cierre de comentario
    # dentro se cerró antes de tiempo y dejó texto y un "#}" suelto en el prompt.
    assert "{{" not in system and "{%" not in system and "#}" not in system
    assert user


def test_unknown_version_is_rejected() -> None:
    with pytest.raises(UnknownPromptVersionError, match="v99"):
        render_estimation_prompt(_request(), version="v99")
    # Only directories on disk are accepted, so path tricks never reach Jinja.
    with pytest.raises(UnknownPromptVersionError):
        render_estimation_prompt(_request(), version="../estimation/v1")


# NOTA: demuestra para qué sirve StrictUndefined: un typo en el nombre de la
# variable (descripton) lanza error en vez de renderizar un hueco vacío.
def test_missing_template_variable_fails_loudly() -> None:
    template = loader._env.from_string("Estimate {{ descripton }}")  # deliberate typo
    with pytest.raises(UndefinedError):
        template.render(description=DESCRIPTION)
