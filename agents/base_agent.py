from agents.internal_state import InternalState
from agents.motivation import MotivationModel


class BaseAgent:
    """
    کلاس پایه تمام Agentها.

    مسئول نگهداری:
    - Environment
    - Agent Name
    - InternalState
    - MotivationModel
    - Action History
    - Long-Term Memory
    """

    def __init__(
        self,
        env,
        agent_name,
        internal_state=None,
        motivation_model=None,
    ):
        self.env = env
        self.agent_name = agent_name

        self.internal_state = (
            internal_state
            if internal_state is not None
            else InternalState()
        )

        self.motivation_model = (
            motivation_model
            if motivation_model is not None
            else MotivationModel()
        )

        self.action_history = []

    # =========================================================
    # Episode
    # =========================================================

    def reset_episode(self):
        """
        Reset وضعیت کوتاه‌مدت Episode.

        Memory عمداً حفظ می‌شود.
        """

        self.action_history = []
        self.internal_state.reset()

    # =========================================================
    # Action
    # =========================================================

    def record_action(self, action):
        """ثبت Action."""

        try:
            action = int(action)
        except (TypeError, ValueError):
            action = 4

        self.action_history.append(action)

    # =========================================================
    # Experience
    # =========================================================

    def record_experience(
        self,
        action,
        reward,
        info=None,
    ):
        """
        ثبت تجربه واقعی Agent.

        این متد باید بعد از env.step()
        فراخوانی شود.
        """

        info = (
            info.copy()
            if isinstance(info, dict)
            else {}
        )

        collected_resource = info.get(
            "collected_resource",
            0.0,
        )

        self.internal_state.record_experience(
            action=action,
            reward=reward,
            info=info,
        )

        self.internal_state.update(
            reward=reward,
            collected_resource=collected_resource,
        )

    # =========================================================
    # State
    # =========================================================

    def get_internal_state(self):
        """دریافت وضعیت داخلی."""

        return self.internal_state.get_state()

    # =========================================================
    # Motivation
    # =========================================================

    def get_motivation_state(self):
        """دریافت وضعیت انگیزشی."""

        state = self.internal_state.get_state()

        return self.motivation_model.get_state(
            lack=state["lack"],
            satisfaction=state["satisfaction"],
        )

    # =========================================================
    # Decision Context
    # =========================================================

    def get_decision_context(self):
        """Context استاندارد تصمیم‌گیری."""

        motivation = self.get_motivation_state()

        return {
            "lack": motivation["lack"],
            "desire": motivation["desire"],
            "satisfaction": motivation["satisfaction"],
            "urgency": motivation["urgency"],
        }

    def build_decision_context(self, observation):
        """ساخت Context کامل Agent."""

        return {
            "observation": observation,
            "internal_state": self.get_decision_context(),
        }
