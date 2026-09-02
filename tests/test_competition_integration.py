
import pytest

from agents.value_seeking_agent import ValueSeekingAgent


class MockEnv:
    def __init__(self):
        self.agents = [
            "agent_1",
            "agent_2",
        ]

        self.positions = {
            "agent_1": (0, 0),
            "agent_2": (9, 9),
        }

        self.resources = {
            (2, 0): 50.0,
            (8, 8): 50.0,
        }


def make_agent():
    env = MockEnv()

    return ValueSeekingAgent(
        env=env,
        agent_name="agent_1",
    )


# ============================================================
# Core capability
# ============================================================

def test_competition_is_core_agent_capability():
    agent = make_agent()

    assert hasattr(
        agent,
        "competition",
    )

    assert hasattr(
        agent,
        "competition_strategy",
    )


# ============================================================
# Competition API
# ============================================================

def test_agent_can_evaluate_resource_competition():
    agent = make_agent()

    state = agent.get_competition_state(
        resource_value=50.0,
        resource_location=(2, 0),
    )

    assert isinstance(
        state,
        dict,
    )

    assert "competition" in state
    assert "risk" in state
    assert "strategy" in state


# ============================================================
# Strategy API
# ============================================================

def test_agent_can_get_competition_strategy():
    agent = make_agent()

    strategy = agent.get_competition_strategy(
        resource_value=50.0,
        resource_location=(2, 0),
    )

    assert strategy in {
        "approach",
        "race",
        "pursue",
        "avoid",
        "defensive",
    }


# ============================================================
# Competition affects score
# ============================================================

def test_competition_changes_resource_score():
    agent = make_agent()

    position = (5, 5)
    value = 50.0

    # بدون رقیب
    agent.env.agents = [
        "agent_1",
    ]

    score_without_competitor = (
        agent.get_resource_score(
            position,
            value,
        )
    )

    # رقیب بسیار نزدیک به Resource
    agent.env.agents = [
        "agent_1",
        "agent_2",
    ]

    agent.env.positions[
        "agent_2"
    ] = position

    score_with_competitor = (
        agent.get_resource_score(
            position,
            value,
        )
    )

    assert (
        score_with_competitor
        != score_without_competitor
    )


# ============================================================
# Competition can change target selection
# ============================================================

def test_competition_can_change_target_selection():
    agent = make_agent()

    # هر دو Resource ارزش و فاصله مشابه دارند.
    agent.env.resources = {
        (2, 0): 50.0,
        (0, 2): 50.0,
    }

    # هیچ رقیبی نداریم.
    agent.env.agents = [
        "agent_1",
    ]

    target_without_competition = (
        agent.select_target()
    )

    # رقیب را کنار Resource اول قرار می‌دهیم.
    agent.env.agents = [
        "agent_1",
        "agent_2",
    ]

    agent.env.positions[
        "agent_2"
    ] = (2, 0)

    target_with_competition = (
        agent.select_target()
    )

    assert (
        target_without_competition
        != target_with_competition
    )

    assert (
        target_with_competition
        == (0, 2)
    )


# ============================================================
# High competition should not increase preference
# ============================================================

def test_high_competition_reduces_resource_preference():
    agent = make_agent()

    resource = (4, 0)
    value = 50.0

    # بدون رقیب
    agent.env.agents = [
        "agent_1",
    ]

    score_without_competition = (
        agent.get_resource_score(
            resource,
            value,
        )
    )

    # رقیب روی Resource
    agent.env.agents = [
        "agent_1",
        "agent_2",
    ]

    agent.env.positions[
        "agent_2"
    ] = resource

    score_with_competition = (
        agent.get_resource_score(
            resource,
            value,
        )
    )

    assert (
        score_with_competition
        < score_without_competition
    )


# ============================================================
# Race strategy
# ============================================================


# ============================================================
# Avoid strategy
# ============================================================

def test_high_risk_competition_can_avoid():
    agent = make_agent()

    strategy = (
        agent.competition_strategy.choose(
            competition=0.90,
            risk=0.90,
            resource_value=80.0,
        )
    )

    assert strategy in {
        "avoid",
        "defensive",
    }

from agents.competition_strategy import CompetitionStrategy


def test_low_competition_means_approach():
    strategy = CompetitionStrategy()

    result = strategy.choose(
        competition=0.10,
        risk=0.05,
        resource_value=50.0,
    )

    assert result == "approach"


def test_medium_competition_means_alternative():
    strategy = CompetitionStrategy()

    result = strategy.choose(
        competition=0.45,
        risk=0.20,
        resource_value=50.0,
    )

    assert result == "alternative"


def test_competition_creates_race():
    strategy = CompetitionStrategy()

    result = strategy.choose(
        competition=0.60,
        risk=0.40,
        resource_value=80.0,
    )

    assert result == "race"


def test_high_competition_with_manageable_risk_pursues():
    strategy = CompetitionStrategy()

    result = strategy.choose(
        competition=0.80,
        risk=0.40,
        resource_value=80.0,
    )

    assert result == "pursue"


def test_high_risk_overrides_competition_strategy():
    strategy = CompetitionStrategy()

    result = strategy.choose(
        competition=0.80,
        risk=0.90,
        resource_value=80.0,
    )

    assert result == "avoid"


def test_strategy_has_priority_modifier():
    strategy = CompetitionStrategy()

    assert strategy.get_priority_modifier(
        "avoid"
    ) < strategy.get_priority_modifier(
        "approach"
    )
