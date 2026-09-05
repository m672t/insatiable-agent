import pytest

from agents.value_seeking_agent import ValueSeekingAgent
from agents.competition_strategy import CompetitionStrategy


class MockEnv:
    def __init__(self):
        self.agents = [
            "agent_1",
        ]

        self.positions = {
            "agent_1": (0, 0),
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


def test_successful_experience_increases_risk_tolerance():
    agent = make_agent()

    agent.motivation_model.lack = 0.0
    agent.motivation_model.satisfaction = 0.0

    baseline = agent.get_risk_tolerance(
        position=(2, 0)
    )

    for _ in range(5):
        agent.internal_state.memory.record(
            action=0,
            reward=10.0,
            info={
                "position": (2, 0),
                "resource_location": (2, 0),
                "resource_value": 50.0,
                "collected_resource": 1.0,
                "outcome": "success",
            },
        )

    experienced = agent.get_risk_tolerance(
        position=(2, 0)
    )

    assert experienced > baseline


def test_failed_experience_reduces_risk_tolerance():
    agent = make_agent()

    agent.motivation_model.lack = 0.0
    agent.motivation_model.satisfaction = 0.0

    baseline = agent.get_risk_tolerance(
        position=(2, 0)
    )

    for _ in range(5):
        agent.internal_state.memory.record(
            action=0,
            reward=-10.0,
            info={
                "position": (2, 0),
                "resource_location": (2, 0),
                "resource_value": 50.0,
                "collected_resource": 0.0,
                "outcome": "failure",
            },
        )

    experienced = agent.get_risk_tolerance(
        position=(2, 0)
    )

    assert experienced < baseline


def test_experience_is_location_specific():
    agent = make_agent()

    agent.motivation_model.lack = 0.0
    agent.motivation_model.satisfaction = 0.0

    baseline = agent.get_risk_tolerance(
        position=(8, 8)
    )

    for _ in range(5):
        agent.internal_state.memory.record(
            action=0,
            reward=10.0,
            info={
                "position": (2, 0),
                "resource_location": (2, 0),
                "resource_value": 50.0,
                "collected_resource": 1.0,
                "outcome": "success",
            },
        )

    experienced = agent.get_risk_tolerance(
        position=(8, 8)
    )

    assert experienced == pytest.approx(
        baseline
    )


def test_experience_and_lack_both_affect_risk_tolerance():
    agent = make_agent()

    agent.motivation_model.lack = 0.0
    agent.motivation_model.satisfaction = 0.0

    baseline = agent.get_risk_tolerance(
        position=(2, 0)
    )

    for _ in range(5):
        agent.internal_state.memory.record(
            action=0,
            reward=10.0,
            info={
                "position": (2, 0),
                "resource_location": (2, 0),
                "resource_value": 50.0,
                "collected_resource": 1.0,
                "outcome": "success",
            },
        )

    experienced = agent.get_risk_tolerance(
        position=(2, 0)
    )

    agent.motivation_model.lack = 1.0

    experienced_and_hungry = (
        agent.get_risk_tolerance(
            position=(2, 0)
        )
    )

    assert experienced > baseline
    assert experienced_and_hungry > experienced
    
def test_risk_tolerance_depends_on_personality():
    risk_seeking = make_agent()
    conservative = make_agent()

    risk_seeking.personality = "risk_seeking"
    conservative.personality = "conservative"

    seeking_tolerance = risk_seeking.get_risk_tolerance()
    conservative_tolerance = conservative.get_risk_tolerance()

    assert seeking_tolerance > conservative_tolerance


def test_neutral_personality_has_middle_risk_tolerance():
    risk_seeking = make_agent()
    neutral = make_agent()
    conservative = make_agent()

    risk_seeking.personality = "risk_seeking"
    neutral.personality = "neutral"
    conservative.personality = "conservative"

    seeking = risk_seeking.get_risk_tolerance()
    neutral_value = neutral.get_risk_tolerance()
    conservative_value = conservative.get_risk_tolerance()

    assert seeking > neutral_value
    assert neutral_value > conservative_value


def test_personality_and_lack_both_affect_risk_tolerance():
    risk_seeking = make_agent()
    conservative = make_agent()

    risk_seeking.personality = "risk_seeking"
    conservative.personality = "conservative"

    risk_seeking.internal_state.lack = 0.8
    conservative.internal_state.lack = 0.8

    seeking = risk_seeking.get_risk_tolerance()
    conservative_value = conservative.get_risk_tolerance()

    assert seeking > conservative_value

# ============================================================
# Social State -> Risk Tolerance
# ============================================================

def test_social_support_increases_risk_tolerance():
    agent = make_agent()

    agent.motivation_model.lack = 0.0
    agent.motivation_model.satisfaction = 0.0

    agent.personality = "neutral"

    agent.internal_state.memory.clear()

    agent.social_state = {
        "support": 0.0,
        "isolation": 0.0,
    }

    baseline = agent.get_risk_tolerance()

    agent.social_state = {
        "support": 1.0,
        "isolation": 0.0,
    }

    supported = agent.get_risk_tolerance()

    assert supported > baseline


def test_social_isolation_reduces_risk_tolerance():
    agent = make_agent()

    agent.motivation_model.lack = 0.0
    agent.motivation_model.satisfaction = 0.0

    agent.personality = "neutral"

    agent.internal_state.memory.clear()

    agent.social_state = {
        "support": 0.0,
        "isolation": 0.0,
    }

    baseline = agent.get_risk_tolerance()

    agent.social_state = {
        "support": 0.0,
        "isolation": 1.0,
    }

    isolated = agent.get_risk_tolerance()

    assert isolated < baseline


def test_social_state_is_independent_of_location():
    agent = make_agent()

    agent.motivation_model.lack = 0.0
    agent.motivation_model.satisfaction = 0.0

    agent.personality = "neutral"

    agent.internal_state.memory.clear()

    agent.social_state = {
        "support": 1.0,
        "isolation": 0.0,
    }

    tolerance_a = agent.get_risk_tolerance(
        position=(2, 0)
    )

    tolerance_b = agent.get_risk_tolerance(
        position=(8, 8)
    )

    assert tolerance_a == pytest.approx(
        tolerance_b
    )


def test_social_support_and_isolation_have_opposite_effects():
    agent = make_agent()

    agent.motivation_model.lack = 0.0
    agent.motivation_model.satisfaction = 0.0

    agent.personality = "neutral"

    agent.internal_state.memory.clear()

    agent.social_state = {
        "support": 1.0,
        "isolation": 0.0,
    }

    supported = agent.get_risk_tolerance()

    agent.social_state = {
        "support": 0.0,
        "isolation": 1.0,
    }

    isolated = agent.get_risk_tolerance()

    assert supported > isolated


def test_social_state_combines_with_lack():
    agent = make_agent()

    agent.personality = "neutral"
    agent.internal_state.memory.clear()

    agent.motivation_model.satisfaction = 0.0

    agent.motivation_model.lack = 0.0

    agent.social_state = {
        "support": 1.0,
        "isolation": 0.0,
    }

    supported = agent.get_risk_tolerance()

    agent.motivation_model.lack = 1.0

    supported_and_hungry = (
        agent.get_risk_tolerance()
    )

    assert supported_and_hungry > supported
    
def test_high_risk_tolerance_allows_pursue_strategy():
    agent = make_agent()

    agent.personality = "risk_seeking"

    agent.motivation_model.lack = 1.0
    agent.motivation_model.satisfaction = 0.0

    agent.social_state = {
        "support": 1.0,
        "isolation": 0.0,
    }

    strategy = agent.competition_strategy.choose(
        competition=0.8,
        risk=0.5,
    )

    assert strategy == "pursue"


def test_low_risk_tolerance_favors_avoidance():
    agent = make_agent()

    agent.personality = "conservative"

    agent.motivation_model.lack = 0.0
    agent.motivation_model.satisfaction = 1.0

    agent.social_state = {
        "support": 0.0,
        "isolation": 1.0,
    }

    strategy = agent.competition_strategy.choose(
        competition=0.8,
        risk=0.9,
    )

    assert strategy == "avoid"


def test_risk_tolerance_changes_competition_decision():
    agent = make_agent()

    agent.motivation_model.lack = 0.0
    agent.motivation_model.satisfaction = 0.0

    agent.social_state = {
        "support": 0.0,
        "isolation": 0.0,
    }

    agent.personality = "risk_seeking"

    high_tolerance = agent.get_risk_tolerance()

    agent.personality = "conservative"

    low_tolerance = agent.get_risk_tolerance()

    assert high_tolerance > low_tolerance
    
def test_high_risk_tolerance_can_change_strategy():
    agent = make_agent()

    agent.competition_strategy = CompetitionStrategy()

    # شرایط رقابت و ریسک ثابت
    competition = 0.8
    risk = 0.65

    # Agent ریسک‌پذیر
    agent.personality = "risk_seeking"
    agent.motivation_model.lack = 1.0
    agent.motivation_model.satisfaction = 0.0

    agent.social_state = {
        "support": 1.0,
        "isolation": 0.0,
    }

    high_tolerance = agent.get_risk_tolerance()

    high_strategy = agent.competition_strategy.choose(
        competition=competition,
        risk=risk,
    )

    # Agent محافظه‌کار
    agent.personality = "conservative"
    agent.motivation_model.lack = 0.0
    agent.motivation_model.satisfaction = 1.0

    agent.social_state = {
        "support": 0.0,
        "isolation": 1.0,
    }

    low_tolerance = agent.get_risk_tolerance()

    low_strategy = agent.competition_strategy.choose(
        competition=competition,
        risk=risk,
    )

    assert high_tolerance > low_tolerance

    assert high_strategy != low_strategy
    
def test_agent_strategy_uses_risk_tolerance():
    agent = make_agent()

    agent.competition_strategy = CompetitionStrategy()

    agent.personality = "risk_seeking"
    agent.motivation_model.lack = 1.0
    agent.motivation_model.satisfaction = 0.0

    agent.social_state = {
        "support": 1.0,
        "isolation": 0.0,
    }

    high_strategy = agent.get_competition_strategy(
        resource_value=80.0,
        resource_location=(2, 0),
    )

    agent.personality = "conservative"
    agent.motivation_model.lack = 0.0
    agent.motivation_model.satisfaction = 1.0

    agent.social_state = {
        "support": 0.0,
        "isolation": 1.0,
    }

    low_strategy = agent.get_competition_strategy(
        resource_value=80.0,
        resource_location=(2, 0),
    )

    assert high_strategy != low_strategy
