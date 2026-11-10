"""Unit tests of the session state: sliding window and project metadata merge."""

import pytest

from app.services.sessions import (
    ConversationHistory,
    ProjectMetadata,
    SessionNotFoundError,
    SessionStore,
)


def _fill(history: ConversationHistory, turns: int) -> None:
    for n in range(1, turns + 1):
        history.add_turn(f"user {n}", f"assistant {n}")


def test_window_drops_the_oldest_turns() -> None:
    history = ConversationHistory(max_turns=3)
    _fill(history, 5)

    assert len(history) == 3
    assert [t.user for t in history.turns] == ["user 3", "user 4", "user 5"]


def test_system_prompt_is_always_first_and_regenerated() -> None:
    history = ConversationHistory(max_turns=2)
    _fill(history, 4)

    messages = history.to_messages_list("system v2")

    assert messages[0] == {"role": "system", "content": "system v2"}
    assert [m["role"] for m in messages[1:]] == ["user", "assistant", "user", "assistant"]


# NOTA: el turno pendiente cuenta dentro de la ventana: con max_turns=3 se
# envían 2 turnos guardados + el mensaje nuevo, nunca 4 mensajes de usuario.
def test_pending_message_counts_towards_the_window() -> None:
    history = ConversationHistory(max_turns=3)
    _fill(history, 5)

    messages = history.to_messages_list("system", pending_user="new question")
    users = [m["content"] for m in messages if m["role"] == "user"]

    assert users == ["user 4", "user 5", "new question"]


def test_single_turn_window_sends_only_the_new_message() -> None:
    history = ConversationHistory(max_turns=1)
    _fill(history, 2)

    messages = history.to_messages_list("system", pending_user="only this")

    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "only this"},
    ]


def test_max_turns_must_be_positive() -> None:
    with pytest.raises(ValueError):
        ConversationHistory(max_turns=0)


def test_metadata_merge_keeps_known_facts_and_accumulates_technologies() -> None:
    current = ProjectMetadata(
        project_name="VetCare", assumed_team_size=3, mentioned_technologies=["React", "Stripe"]
    )
    update = ProjectMetadata(mentioned_technologies=["stripe", "PostgreSQL"], agreed_scope="MVP")

    merged = current.merge(update)

    assert merged.project_name == "VetCare"
    assert merged.assumed_team_size == 3
    assert merged.mentioned_technologies == ["React", "Stripe", "PostgreSQL"]
    assert merged.agreed_scope == "MVP"


def test_metadata_merge_overwrites_with_new_values() -> None:
    merged = ProjectMetadata(project_name="Old", assumed_team_size=2).merge(
        ProjectMetadata(project_name="New", assumed_team_size=4)
    )
    assert (merged.project_name, merged.assumed_team_size) == ("New", 4)


def test_empty_metadata() -> None:
    assert ProjectMetadata().is_empty()
    assert not ProjectMetadata(project_name="X").is_empty()


def test_store_creates_unique_sessions_and_rejects_unknown_ids() -> None:
    store = SessionStore(max_turns=4)
    first, second = store.create(), store.create()

    assert first.session_id != second.session_id
    assert store.get(first.session_id) is first
    assert first.history.max_turns == 4
    with pytest.raises(SessionNotFoundError):
        store.get("missing")
