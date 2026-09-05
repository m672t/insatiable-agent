from agents.internal_state import InternalState
from agents.motivation import MotivationModel
from agents.memory import MemorySystem
from agents.learning_system import LearningSystem
from agents.competition import CompetitionSystem
from agents.competition_strategy import CompetitionStrategy


class BaseAgent:
    """
    Base class for all agents.

    Responsibilities:
        - Environment
        - Agent identity
        - Internal state
        - Motivation model
        - Unified memory system
        - Action history
    """

    def __init__(
        self,
        env,
        agent_name,
        internal_state=None,
        motivation_model=None,
        memory=None,
    ):
        self.env = env
        self.agent_name = agent_name

        # =====================================================
        # Unified Memory
        # =====================================================
        #
        # MemorySystem تنها حافظه Agent است.
        #
        # اگر Memory از بیرون داده شود، همان instance
        # حفظ می‌شود تا امکان share / injection وجود داشته باشد.
        #

        self.memory = (
            memory
            if memory is not None
            else MemorySystem()
        )

        # =====================================================
        # Internal State
        # =====================================================

        self.internal_state = (
            internal_state
            if internal_state is not None
            else InternalState(
                memory=self.memory
            )
        )
        
        self.learning = LearningSystem(
            self.internal_state.memory
        )
        
        self.competition = CompetitionSystem()

        self.competition_strategy = CompetitionStrategy()

        # اگر InternalState از بیرون داده شده باشد،
        # حافظه Unified را به آن متصل می‌کنیم.
        if hasattr(self.internal_state, "set_memory"):
            self.internal_state.set_memory(
                self.memory
            )
        else:
            self.internal_state.memory = self.memory

        # =====================================================
        # Motivation
        # =====================================================

        self.motivation_model = (
            motivation_model
            if motivation_model is not None
            else MotivationModel()
        )

        # =====================================================
        # Episode-local history
        # =====================================================

        self.action_history = []

    # =========================================================
    # Episode
    # =========================================================

    def reset_episode(self):
        """
        Reset وضعیت کوتاه‌مدت Episode.

        Long-term memory هرگز با reset_episode پاک نمی‌شود.
        """

        self.action_history = []

        # فقط وضعیت درونی کوتاه‌مدت
        self.internal_state.reset()

        # MemorySystem حافظه بلندمدت را نگه می‌دارد.
        reset_memory = getattr(
            self.memory,
            "reset_episode",
            None,
        )

        if callable(reset_memory):
            reset_memory()

    # =========================================================
    # Action
    # =========================================================

    def record_action(self, action):
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
        info = (
            info.copy()
            if isinstance(info, dict)
            else {}
        )

        competition_state = self.evaluate_competition(
            resource_value=info.get(
                "resource_value",
                0.0,
            ),
            resource_location=info.get(
                "resource_location",
            ),
            agent_position=info.get(
                "position",
            ),
            other_agents=info.get(
                "other_agents",
                [],
            ),
        )

        info["competition"] = (
            competition_state["competition"]
        )

        info["competition_risk"] = (
            competition_state["risk"]
        )

        info["competition_strategy"] = (
            competition_state["strategy"]
        )

        experience = self.internal_state.memory.record(
            action=action,
            reward=reward,
            info=info,
        )

        self.learning.learn_from_experience(
            experience
        )

        return experience
    
    # =========================================================
    # Memory API
    # =========================================================

    def get_memory(self):
        """
        دریافت حافظه واحد Agent.
        """

        return self.memory.get_memory()

    def get_short_term_memory(self, window=10):
        return self.memory.get_short_term(
            window=window
        )

    def get_long_term_memory(self):
        return self.memory.get_long_term()

        # =========================================================
    # Competition
    # =========================================================

    def evaluate_competition(
        self,
        resource_value=0.0,
        resource_location=None,
        agent_position=None,
        other_agents=None,
    ):
        """
        ارزیابی وضعیت رقابتی فعلی Agent.
        """

        return self.competition.evaluate(
            resource_value=resource_value,
            resource_location=resource_location,
            agent_position=agent_position,
            other_agents=other_agents,
        )

    def get_competition_state(self):
        """
        آخرین وضعیت رقابتی Agent.
        """

        return self.competition.get_state()
    # =========================================================
    # State
    # =========================================================

    def get_internal_state(self):
        return self.internal_state.get_state()

    # =========================================================
    # Motivation
    # =========================================================

    def get_motivation_state(self):
        return self.motivation_model.get_state(
            lack=self.motivation_model.lack,
            satisfaction=self.motivation_model.satisfaction,
            urgency=self.motivation_model.urgency,
        )

    # =========================================================
    # Decision Context
    # =========================================================

    def get_decision_context(self):
        motivation = self.get_motivation_state()

        return {
            "lack": motivation["lack"],
            "desire": motivation["desire"],
            "satisfaction": motivation["satisfaction"],
            "urgency": motivation["urgency"],
            "motives": motivation.get(
                "motives",
                {},
            ),
            "conflicts": motivation.get(
                "conflicts",
                {},
            ),
            "conflict_pressure": motivation.get(
                "conflict_pressure",
                0.0,
            ),
            "hidden_motives": motivation.get(
                "hidden_motives",
                {},
            ),
        }

    def evaluate_resource_competition(
        self,
        resource,
        competitors,
        agent_position=None,
    ):
        """
        ارزیابی Competition برای یک Resource.
        """

        if agent_position is None:
            state = self.get_internal_state()
            agent_position = state.get(
                "position"
            )

        if agent_position is None:
            return {
                "distance": float("inf"),
                "competition": 0.0,
                "risk": 0.0,
                "priority": 0.0,
                "strategy": "ignore",
            }

        return self.competition.evaluate_resource(
            agent_position=agent_position,
            resource=resource,
            competitors=competitors,
        )


    def choose_resource(
        self,
        resources,
        competitors,
        agent_position=None,
    ):
        """
        انتخاب Resource بر اساس ارزش،
        فاصله و Competition.
        """

        if agent_position is None:
            state = self.get_internal_state()
            agent_position = state.get(
                "position"
            )

        return self.competition.choose_resource(
            agent_position=agent_position,
            resources=resources,
            competitors=competitors,
        )

    def build_decision_context(self, observation):

        current_position = (
            self.env.positions.get(
                self.agent_name
            )
        )

        return {
            "observation": observation,
            "internal_state": self.get_decision_context(),
            "memory": self.get_memory(),

            "competition": {
                "available": True,
                "agent_position": current_position,
            },
        }
    
    # =========================================================
# Competition
# =========================================================

    def get_competition_state(
        self,
        resource_value=0.0,
        resource_location=None,
        agent_position=None,
        other_agents=None,
    ):
        """
        ارزیابی وضعیت رقابتی Agent نسبت به Resource.
        """

        if agent_position is None:
            agent_position = self.env.positions.get(
                self.agent_name
            )

        return self.competition.evaluate(
            resource_value=resource_value,
            resource_location=resource_location,
            agent_position=agent_position,
            other_agents=other_agents,
        )

    def get_competition_strategy(
        self,
        resource_value=0.0,
        resource_location=None,
        agent_position=None,
        other_agents=None,
    ):
        """
        تعیین رفتار رقابتی Agent نسبت به Resource.
        """

        state = self.get_competition_state(
            resource_value=resource_value,
            resource_location=resource_location,
            agent_position=agent_position,
            other_agents=other_agents,
        )

    # Risk tolerance فعلی Agent
        risk_tolerance = self.get_risk_tolerance(
            position=resource_location
        )

    # Strategy باید tolerance فعلی Agent را بداند.
        self.competition_strategy.risk_tolerance = (
            risk_tolerance
        )

        competition = float(
            state.get("competition", 0.0)
        )

        risk = float(
            state.get("risk", 0.0)
        )

        competitor_distance = (
            self._nearest_competitor_distance(state)
        )

    # اگر CompetitionSystem هیچ ریسک یا رقابتی
    # گزارش نکرده، tolerance همچنان باید روی
    # تصمیم Agent اثر بگذارد.
        if competition <= 0.0 and risk <= 0.0:
            if risk_tolerance >= 0.75:
                return "pursue"

            if risk_tolerance <= 0.25:
                return "avoid"

        return self.competition_strategy.choose(
            competition=competition,
            risk=risk,
            resource_value=resource_value,
            competitor_distance=competitor_distance,
            risk_tolerance=risk_tolerance,
        )
    
    def _nearest_competitor_distance(
        self,
        state,
    ):
        agents = state.get(
            "agents",
            [],
        )

        distances = []

        for agent in agents:
            distance = agent.get(
                "distance_to_resource"
            )

            if distance is not None:
                distances.append(
                    float(distance)
                )

        if not distances:
            return None

        return min(distances)

    def get_resource_competition(
        self,
        position,
    ):
        state = self.get_competition_state(
            resource_value=self.env.resources.get(
                position,
                0.0,
            ),
            resource_location=position,
        )

        return float(
            state.get(
                "competition",
                0.0,
            )
        ) 
