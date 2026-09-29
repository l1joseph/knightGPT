"""Unit tests for _split_latest_user_message(), the pure helper that splits
an OpenAI-style messages array into (latest user content, prior history)
for /v1/chat/completions. Confirmed live on kl-remote that without this
split feeding AgentOrchestrator.run(history=...), every follow-up question
in a multi-turn Open WebUI chat got answered as a brand-new conversation
with no memory of what was said before."""

import pytest

from src.api.main import _split_latest_user_message


@pytest.mark.unit
def test_single_user_message_has_no_history():
    user_message, history = _split_latest_user_message(
        [{"role": "user", "content": "what microbes are associated with AD?"}]
    )
    assert user_message == "what microbes are associated with AD?"
    assert history == []


@pytest.mark.unit
def test_follow_up_question_carries_prior_turns_as_history():
    messages = [
        {"role": "user", "content": "what microbes are associated with AD?"},
        {"role": "assistant", "content": "Several taxa have been studied..."},
        {"role": "user", "content": "so what are the microbes"},
    ]
    user_message, history = _split_latest_user_message(messages)

    assert user_message == "so what are the microbes"
    assert history == [
        {"role": "user", "content": "what microbes are associated with AD?"},
        {"role": "assistant", "content": "Several taxa have been studied..."},
    ]


@pytest.mark.unit
def test_system_messages_are_dropped_from_history():
    """The orchestrator has its own GENERATOR_SYSTEM_PROMPT -- a
    client-supplied system message must not leak into history."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]
    user_message, history = _split_latest_user_message(messages)

    assert user_message == "second question"
    assert history == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]


@pytest.mark.unit
def test_no_user_message_returns_none_and_empty_history():
    user_message, history = _split_latest_user_message(
        [{"role": "system", "content": "You are a helpful assistant."}]
    )
    assert user_message is None
    assert history == []


@pytest.mark.unit
def test_empty_messages_returns_none_and_empty_history():
    assert _split_latest_user_message([]) == (None, [])
