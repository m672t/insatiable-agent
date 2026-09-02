
from agents.base_agent import BaseAgent


class DummyEnv:
    """حداقل محیط لازم برای ساخت BaseAgent."""

    def __init__(self):
        self.positions = {
            "test_agent": (0, 0),
        }

        self.resources = {}

        self.agents = [
            "test_agent",
        ]


def make_agent():
    env = DummyEnv()

    return BaseAgent(
        env=env,
        agent_name="test_agent",
    )


def test_episode_preserves_memory():
    agent = make_agent()

    # ثبت یک تجربه
    agent.record_experience(
        action=3,
        reward=10.0,
        info={
            "position": (1, 0),
            "collected_resource": 10.0,
        },
    )

    memory_before = agent.internal_state.get_memory()

    assert len(memory_before) == 1

    # پایان Episode
    agent.reset_episode()

    memory_after = agent.internal_state.get_memory()

    # Memory بلندمدت نباید پاک شود
    assert len(memory_after) == 1
    assert memory_after[0]["reward"] == 10.0


def test_agent_episode_reset_preserves_long_term_memory():
    agent = make_agent()

    # وضعیت کوتاه‌مدت را تغییر بده
    agent.record_action(3)

    agent.record_experience(
        action=3,
        reward=20.0,
        info={
            "position": (1, 0),
            "collected_resource": 20.0,
        },
    )

    assert len(agent.action_history) == 1
    assert len(agent.internal_state.get_memory()) == 1

    # پایان Episode
    agent.reset_episode()

    # Action history متعلق به Episode است
    assert agent.action_history == []

    # Memory متعلق به Agent است و باید باقی بماند
    assert len(agent.internal_state.get_memory()) == 1

    # State کوتاه‌مدت باید reset شده باشد
    state = agent.get_internal_state()

    assert state["lack"] == 0.0
    assert state["desire"] == 0.0
    assert state["satisfaction"] == 0.0


def test_memory_accumulates_across_episodes():
    agent = make_agent()

    # =========================================================
    # Episode 1
    # =========================================================

    agent.record_experience(
        action=3,
        reward=10.0,
        info={
            "position": (1, 0),
            "collected_resource": 10.0,
        },
    )

    assert len(agent.internal_state.get_memory()) == 1

    agent.reset_episode()

    # Memory باید باقی مانده باشد
    assert len(agent.internal_state.get_memory()) == 1

    # =========================================================
    # Episode 2
    # =========================================================

    agent.record_experience(
        action=1,
        reward=30.0,
        info={
            "position": (1, 1),
            "collected_resource": 30.0,
        },
    )

    memory = agent.internal_state.get_memory()

    # هر دو تجربه باید حفظ شده باشند
    assert len(memory) == 2

    assert memory[0]["reward"] == 10.0
    assert memory[1]["reward"] == 30.0

    # =========================================================
    # Episode 3
    # =========================================================

    agent.reset_episode()

    agent.record_experience(
        action=2,
        reward=50.0,
        info={
            "position": (0, 1),
            "collected_resource": 50.0,
        },
    )

    memory = agent.internal_state.get_memory()

    # سه تجربه از سه Episode
    assert len(memory) == 3

    assert memory[0]["reward"] == 10.0
    assert memory[1]["reward"] == 30.0
    assert memory[2]["reward"] == 50.0
