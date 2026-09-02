from agents.memory import MemorySystem
from agents.learning_system import LearningSystem


def make_learning():
    memory = MemorySystem()
    learning = LearningSystem(memory)
    return memory, learning


def test_learning_reads_reward_from_memory():
    memory, learning = make_learning()

    memory.record(
        action=1,
        reward=10.0,
    )

    assert learning.get_reward_learning() > 0.0


def test_action_value_is_learned():
    memory, learning = make_learning()

    learning.update_action_value(
        action=2,
        reward=10.0,
    )

    assert learning.get_action_value(2) > 0.0


def test_recent_experience_is_available():
    memory, learning = make_learning()

    memory.record(
        action=1,
        reward=5.0,
    )

    recent = learning.get_recent_experience()

    assert len(recent) == 1
    assert recent[0]["action"] == 1


def test_recency_weight_exists():
    memory, learning = make_learning()

    memory.record(
        action=1,
        reward=1.0,
    )

    memory.record(
        action=2,
        reward=10.0,
    )

    assert (
        learning.get_recency_weight(1)
        >= learning.get_recency_weight(0)
    )


def test_location_learning():
    memory, learning = make_learning()

    learning.update_location_value(
        position=(2, 3),
        reward=10.0,
    )

    assert (
        learning.get_location_value((2, 3))
        > 0.0
    )


def test_learning_signal_exists():
    memory, learning = make_learning()

    signal = learning.get_learning_signal(
        reward=10.0
    )

    assert signal > 0.0


def test_failure_learning_exists():
    memory, learning = make_learning()

    memory.record(
        action=1,
        reward=-10.0,
    )

    assert (
        learning.get_failure_learning()
        > 0.0
    )


def test_strategy_changes_after_experience():
    memory, learning = make_learning()

    before = learning.get_strategy_bias()

    learning.learn_from_experience(
        {
            "action": 1,
            "reward": 10.0,
            "position": (0, 0),
        }
    )

    after = learning.get_strategy_bias()

    assert before != after


def test_environment_adapts():
    memory, learning = make_learning()

    memory.record(
        action=1,
        reward=10.0,
    )

    state = learning.update_environment_model()

    assert state["average_reward"] == 10.0
    assert state["success_rate"] == 1.0


def test_learning_state_exists():
    memory, learning = make_learning()

    memory.record(
        action=1,
        reward=5.0,
    )

    state = learning.get_state()

    assert "reward_learning" in state
    assert "action_values" in state
    assert "recent_experience" in state
    assert "recency_weight" in state
    assert "location_values" in state
    assert "learning_signal" in state
    assert "failure_learning" in state
    assert "strategy_bias" in state
    assert "environment_model" in state
