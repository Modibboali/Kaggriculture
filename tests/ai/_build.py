"""Shared synthetic-state builder for AI tests (no Kaggle dependency)."""

from __future__ import annotations

from agent.ai import ActionGenerator, Evaluator, SearchState, SimulatorAdapter, Terminal
from agent.simulator import GameConfig
from agent.simulator.game_config import DEFAULT_MARKET_PARAMS, MARKET_I0, PRODUCT_ITEMS
from agent.state import (
    EmptyTile,
    Farm,
    GameState,
    Inventory,
    Market,
    PlayerState,
    Position,
    Quadrant,
    Seeds,
    Tile,
    Town,
    Worker,
)

FARMER_POS = Position(1, 1)


def make_game(
    *,
    board_size: int = 4,
    money: int = 3000,
    step: int = 0,
    day: int = 0,
    hour: int = 0,
) -> GameState:
    """A minimal ``board_size`` x ``board_size`` state with NW unlocked."""
    tiles: tuple[tuple[Tile, ...], ...] = tuple(
        tuple(EmptyTile() for _ in range(board_size)) for _ in range(board_size)
    )
    farmer = Worker(0, FARMER_POS, Inventory.empty(), True)
    farm = Farm(
        money=money,
        tiles=tiles,
        farmer=farmer,
        workers=(),
        unlocked_quadrants=frozenset({Quadrant.NW}),
        hires_today=0,
    )
    market = Market(
        Inventory({item: MARKET_I0 for item in PRODUCT_ITEMS}),
        {item: params.base for item, params in DEFAULT_MARKET_PARAMS.items()},
    )
    player = PlayerState(farm, Inventory.empty(), Seeds.empty(), (farmer,))
    return GameState(
        day=day,
        hour=hour,
        step=step,
        market=market,
        town=Town(frozenset()),
        players=(player, player),
        current_player=0,
    )


def make_search_state(board_size: int = 4, **kw: int) -> SearchState:
    return SearchState(make_game(board_size=board_size, **kw))


def make_config(board_size: int = 4) -> GameConfig:
    return GameConfig(board_size=board_size)


def make_components(
    board_size: int = 4,
) -> tuple[GameConfig, SimulatorAdapter, ActionGenerator, Evaluator, Terminal]:
    """Shared search components bound to a board size."""
    config = make_config(board_size)
    adapter = SimulatorAdapter(count_transitions=True)
    generator = ActionGenerator(config)
    evaluator = Evaluator(config)
    terminal = Terminal(config)
    return config, adapter, generator, evaluator, terminal
