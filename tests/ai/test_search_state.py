"""SearchState: equality, hashing, immutability, state_key."""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError

import pytest

from agent.ai import SearchState, state_key
from agent.ai.search_state import state_key as module_state_key
from agent.state import Farm, PlayerState

from ._build import make_game, make_search_state


def test_equality_is_value_based() -> None:
    a = make_search_state(money=1000)
    b = SearchState(make_game(money=1000))
    assert a == b
    assert a.game == b.game


def test_equality_detects_difference() -> None:
    a = make_search_state(money=1000)
    b = make_search_state(money=2000)
    assert a != b


def test_state_key_is_deterministic_and_value_based() -> None:
    a = make_search_state(money=1000)
    b = SearchState(make_game(money=1000))
    assert module_state_key(a) == module_state_key(b) == a.state_key()
    assert hash(a) == hash(b)


def test_state_key_differs_for_different_states() -> None:
    a = make_search_state(money=1000)
    b = make_search_state(money=2000)
    assert a.state_key() != b.state_key()


def test_state_key_is_independent_of_identity() -> None:
    a = make_search_state(money=500)
    b = SearchState(a.game)  # different object, same value
    assert a is not b
    assert state_key(a) == state_key(b)


def test_state_key_is_cached() -> None:
    state = make_search_state()
    first = state.state_key()
    assert state.state_key() == first  # second call uses the cache


def test_search_state_is_immutable() -> None:
    state = make_search_state()
    with pytest.raises(FrozenInstanceError):
        state.game = make_game()  # type: ignore[misc]


def test_search_state_size_is_small() -> None:
    # The wrapper is just a reference + a cached int key; the bulk is the
    # underlying GameState, which is shared/reused by the immutable model.
    state = make_search_state()
    wrapper_size = sys.getsizeof(state)
    assert wrapper_size < 128
    assert state.state_key() is not None
