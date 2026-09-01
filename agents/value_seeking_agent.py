from agents.base_agent import BaseAgent
import numpy as np

class ValueSeekingAgent(BaseAgent):
    """
    Value Seeking Agent

    Decision factors:
        1. Resource Value
        2. Distance
        3. Motivation
        4. Location Memory
        5. Action Memory
        6. Risk Tolerance
        7. Recent Experience Learning
        8. Exploration / Exploitation

    Memory is preserved between episodes.
    """

    def __init__(
        self,
        env,
        agent_name,
        distance_weight=1.0,
        motivation_weight=1.0,
        memory_weight=0.5,
        risk_weight=1.0,
        action_memory_weight=0.1,
        learning_weight=0.35,
        recency_decay=0.995,
        exploration_weight=0.15,
        novelty_weight=0.10,
        internal_state=None,
    ):
        super().__init__(
            env=env,
            agent_name=agent_name,
            internal_state=internal_state,
        )

        self.distance_weight = float(
            distance_weight
        )

        self.motivation_weight = float(
            motivation_weight
        )

        self.memory_weight = float(
            memory_weight
        )
        
        # =========================================================
# Day 5 - Experience Memory
# =========================================================

        self.memory_max_size = 200

        self.experience_memory = []

        self.current_episode_experiences = []

        self.completed_episode_memory = []

        self.risk_weight = float(
            risk_weight
        )

        self.action_memory_weight = float(
            action_memory_weight
        )

        self.learning_weight = float(
            learning_weight
        )

        self.recency_decay = float(
            recency_decay
        )

        self.exploration_weight = float(
            exploration_weight
        )

        self.novelty_weight = float(
            novelty_weight
        )
 
                # =====================================================
        # Motivation State - Day 3
        # =====================================================

        self.motivation_state = {
            "lack": 0.0,
            "desire": 0.0,
            "satisfaction": 0.0,
            "urgency": 0.0,
        }

        # Last interaction state
        self.steps_since_collection = 0
        self.last_collected_value = 0.0

        # Motivation parameters
        self.motivation_decay = 0.95
        self.urgency_growth = 0.025
        self.satisfaction_decay = 0.90
        
        # =========================================================
        # Day 4 - Risk Model
        # =========================================================

        self.risk_distance_weight = 1.0  
        self.risk_competition_weight = 1.0

        # میزان  تحمل ریسک پایه
        self.base_risk_tolerance = 0.50

        # اثر Motivation روی Risk Tolerance
        self.lack_risk_tolerance_weight = 0.40
        self.satisfaction_risk_tolerance_weight = 0.40

        # شدت اثر Risk روی Resource Score
        self.risk_penalty_weight = 0.50

    # =========================================================
    # Distance
    # =========================================================

    def get_distance(self, position):

        current = self.env.positions[
            self.agent_name
        ]

        return (
            abs(
                int(current[0])
                - int(position[0])
            )
            +
            abs(
                int(current[1])
                - int(position[1])
            )
        )

    # =========================================================
    # Risk
    # =========================================================

    def get_risk_tolerance(self):

        motivation = (
            self.get_motivation_state()
        )

        lack = float(
            motivation.get(
                "lack",
                0.0,
            )
        )

        urgency = float(
            motivation.get(
                "urgency",
                0.0,
            )
        )

        satisfaction = float(
            motivation.get(
                "satisfaction",
                0.0,
            )
        )

        risk = (
            0.45 * lack
            + 0.45 * urgency
            - 0.35 * satisfaction
        )

        return max(
            0.0,
            min(1.0, risk),
        )

    # =========================================================
    # Memory
    # =========================================================

    def get_memory(self):

        return self.internal_state.get_memory()

    # =========================================================
    # Safe Reward
    # =========================================================

    def _get_reward(self, experience):

        if not isinstance(
            experience,
            dict,
        ):
            return 0.0

        try:
            return float(
                experience.get(
                    "reward",
                    0.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    # =========================================================
    # Position Extraction
    # =========================================================

    def _get_experience_position(
        self,
        experience,
    ):

        if not isinstance(
            experience,
            dict,
        ):
            return None

        position = experience.get(
            "position"
        )

        if position is None:

            info = experience.get(
                "info",
                {},
            )

            if isinstance(
                info,
                dict,
            ):
                position = info.get(
                    "position"
                )

        if position is None:
            return None

        try:

            return tuple(
                int(x)
                for x in position
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

    # =========================================================
    # Collected Resource
    # =========================================================

    def _get_collected_resource(
        self,
        experience,
    ):

        if not isinstance(
            experience,
            dict,
        ):
            return 0.0

        collected = experience.get(
            "collected_resource"
        )

        if collected is None:

            info = experience.get(
                "info",
                {},
            )

            if isinstance(
                info,
                dict,
            ):
                collected = info.get(
                    "collected_resource",
                    0.0,
                )

        try:
            return max(
                0.0,
                float(collected),
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    # =========================================================
    # Memory Reward
    # =========================================================

    def get_memory_reward(self):

        memory = self.get_memory()

        if not memory:
            return 0.0

        rewards = []

        for experience in memory:

            if not isinstance(
                experience,
                dict,
            ):
                continue

            rewards.append(
                self._get_reward(
                    experience
                )
            )

        if not rewards:
            return 0.0

        return (
            sum(rewards)
            / len(rewards)
        )

    # =========================================================
    # Recency Weight
    # =========================================================

    def get_recency_weight(
        self,
        index,
        total,
    ):
        """
        index:
            Index in the ORIGINAL memory.

        Newer experiences have larger weights.
        """

        if total <= 0:
            return 0.0

        age = (
            total - 1 - index
        )

        return (
            self.recency_decay ** age
        )

    # =========================================================
    # Weighted Reward
    # =========================================================

    def get_weighted_reward(
        self,
        experiences,
    ):

        if not experiences:
            return 0.0

        weighted_sum = 0.0
        weight_sum = 0.0

        total = len(experiences)

        for index, experience in enumerate(
            experiences
        ):

            if not isinstance(
                experience,
                dict,
            ):
                continue

            reward = self._get_reward(
                experience
            )

            weight = (
                self.get_recency_weight(
                    index,
                    total,
                )
            )

            weighted_sum += (
                reward * weight
            )

            weight_sum += weight

        if weight_sum <= 0.0:
            return 0.0

        return (
            weighted_sum
            / weight_sum
        )

    # =========================================================
    # Recent Reward
    # =========================================================

    def get_recent_reward(
        self,
        window=100,
    ):
        """
        Reward from the most recent experiences.

        This reacts faster to environmental changes
        than the full-memory average.
        """

        memory = self.get_memory()

        if not memory:
            return 0.0

        recent = memory[
            -max(1, int(window)):
        ]

        return self.get_weighted_reward(
            recent
        )

    # =========================================================
    # Action Memory
    # =========================================================

    def get_action_memory_value(
        self,
        action,
    ):
        """
        Learned reward for an action.

        Normalized to [-1, +1].
        """

        memory = self.get_memory()

        if not memory:
            return 0.0

        matching = []

        for experience in memory:

            if not isinstance(
                experience,
                dict,
            ):
                continue

            try:
                previous_action = int(
                    experience.get(
                        "action",
                        -999,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if previous_action == int(
                action
            ):
                matching.append(
                    experience
                )

        if not matching:
            return 0.0

        average_reward = (
            self.get_weighted_reward(
                matching
            )
        )

        # Smooth normalization instead of
        # a hard 50 reward threshold.
        normalized = (
            average_reward
            / (
                abs(average_reward)
                + 50.0
            )
        )

        return max(
            -1.0,
            min(1.0, normalized),
        )

    # =========================================================
    # Location Memory
    # =========================================================

    def get_location_memory_value(
        self,
        position,
    ):
        """
        Weighted historical resource value
        observed at this exact location.
        """

        memory = self.get_memory()

        if not memory:
            return 0.0

        target = tuple(
            int(x)
            for x in position
        )

        matching = []

        for experience in memory:

            previous_position = (
                self._get_experience_position(
                    experience
                )
            )

            if previous_position != target:
                continue

            collected = (
                self._get_collected_resource(
                    experience
                )
            )

            if collected > 0.0:
                matching.append(
                    experience
                )

        if not matching:
            return 0.0

        # IMPORTANT:
        # matching is already ordered according
        # to the original memory order.
        return self.get_weighted_reward_like_resource(
            matching
        )

    # =========================================================
    # Location Resource Average
    # =========================================================

    def get_weighted_reward_like_resource(
        self,
        experiences,
    ):

        if not experiences:
            return 0.0

        total = len(experiences)

        weighted_sum = 0.0
        weight_sum = 0.0

        for index, experience in enumerate(
            experiences
        ):

            collected = (
                self._get_collected_resource(
                    experience
                )
            )

            weight = (
                self.get_recency_weight(
                    index,
                    total,
                )
            )

            weighted_sum += (
                collected * weight
            )

            weight_sum += weight

        if weight_sum <= 0.0:
            return 0.0

        return (
            weighted_sum
            / weight_sum
        )

    # =========================================================
    # Location Visit Count
    # =========================================================

    def get_location_visit_count(
        self,
        position,
    ):

        memory = self.get_memory()

        if not memory:
            return 0

        target = tuple(
            int(x)
            for x in position
        )

        count = 0

        for experience in memory:

            previous_position = (
                self._get_experience_position(
                    experience
                )
            )

            if previous_position == target:
                count += 1

        return count

    # =========================================================
    # Location Success Rate
    # =========================================================

    def get_location_success_rate(
        self,
        position,
    ):

        memory = self.get_memory()

        if not memory:
            return 0.0

        target = tuple(
            int(x)
            for x in position
        )

        successes = 0
        visits = 0

        for experience in memory:

            previous_position = (
                self._get_experience_position(
                    experience
                )
            )

            if previous_position != target:
                continue

            visits += 1

            if (
                self._get_collected_resource(
                    experience
                )
                > 0.0
            ):
                successes += 1

        if visits == 0:
            return 0.0

        return (
            successes / visits
        )

    # =========================================================
    # Location Learning Signal
    # =========================================================

    def get_location_learning_signal(
        self,
        position,
        current_value,
    ):

        memory_value = (
            self.get_location_memory_value(
                position
            )
        )

        if memory_value <= 0.0:
            return 0.0

        current_value = max(
            1.0,
            float(current_value),
        )

        ratio = (
            memory_value
            / current_value
        )

        signal = (
            ratio - 1.0
        )

        return max(
            -1.0,
            min(1.0, signal),
        )

    # =========================================================
    # Novelty
    # =========================================================

    def get_location_novelty(
        self,
        position,
    ):
        """
        Unknown locations receive a small exploration bonus.

        The bonus decreases as the location is visited.
        """

        visits = (
            self.get_location_visit_count(
                position
            )
        )

        if visits <= 0:
            return 1.0

        return 1.0 / (
            1.0
            + visits
        )

    # =========================================================
    # Motivation
    # =========================================================

       # =========================================================
    # Motivation Multiplier
    # =========================================================

    def get_motivation_multiplier(self):

        motivation = (
            self.get_motivation_state()
        )

        lack = float(
            motivation.get("lack", 0.0)
        )

        desire = float(
            motivation.get("desire", 0.0)
        )

        urgency = float(
            motivation.get("urgency", 0.0)
        )

        satisfaction = float(
            motivation.get("satisfaction", 0.0)
        )

        signal = (
            0.30 * lack
            + 0.45 * desire
            + 0.35 * urgency
            - 0.25 * satisfaction
        )

        return max(
            0.50,
            min(
                2.00,
                1.0
                + self.motivation_weight
                * signal,
            ),
        )

    # =========================================================
    # Expected Reward
    # =========================================================

    def get_expected_location_value(
        self,
        position,
        current_value,
    ):
        """
        Combines current resource value with
        historical location value.

        Current observation remains dominant,
        memory modifies the estimate instead
        of completely replacing it.
        """

        current_value = max(
            0.0,
            float(current_value),
        )

        historical = (
            self.get_location_memory_value(
                position
            )
        )

        if historical <= 0.0:
            return current_value

        success_rate = (
            self.get_location_success_rate(
                position
            )
        )

        # Confidence increases with repeated visits.
        visits = (
            self.get_location_visit_count(
                position
            )
        )

        confidence = min(
            0.60,
            visits / 20.0,
        )

        confidence *= (
            0.50
            + 0.50 * success_rate
        )

        expected = (
            (1.0 - confidence)
            * current_value
            +
            confidence
            * historical
        )

        return max(
            0.0,
            expected,
        )

    # =========================================================
    # Resource Score
    # =========================================================

    def get_resource_score(
        self,
        position,
        value,
    ):
        """
        Final resource score.

        Combines:
            current value
            distance
            motivation
            location memory
            recent learning
            novelty
            risk
        """

        value = max(
            0.0,
            float(value),
        )

        distance = self.get_distance(
            position
        )

        # -----------------------------------------------------
        # Expected Value
        # -----------------------------------------------------

        expected_value = (
            self.get_expected_location_value(
                position,
                value,
            )
        )

        # -----------------------------------------------------
        # Distance
        # -----------------------------------------------------

        # -----------------------------------------------------
        # Day 4 Risk
        # -----------------------------------------------------

        risk = self.get_resource_risk(
        position
        )

        risk_adjustment = (
            self.get_risk_adjustment(
                risk
            )
        )

        distance_cost = (
            self.distance_weight
            +
            risk_adjustment
        )

        base_score = (
            expected_value
            / max(
                0.1,
                distance_cost,
            )
        )

        base_score = (
            expected_value
            / max(
                0.1,
                distance_cost,
            )
        )



# -----------------------------------------------------
# Day 3 Motivation
# -----------------------------------------------------

        motivation = (
            self.get_motivation_state()
        )

        lack = float(
            motivation.get("lack", 0.0)
        )

        desire = float(
            motivation.get("desire", 0.0)
        )

        urgency = float(
            motivation.get("urgency", 0.0)
        )

        satisfaction = float(
            motivation.get("satisfaction", 0.0)
        )

        value_ratio = (
            value / 50.0
        )

        score = base_score

        # Desire
        desire_bonus = (
            1.0
            + 0.80
            * desire
            * value_ratio
        )

        score *= desire_bonus

        # Urgency
        proximity = 1.0 / (
            1.0 + distance
        )

        urgency_bonus = (
            1.0
            + 1.20
            * urgency
            * proximity
            * value_ratio
        )

        score *= urgency_bonus

        # Lack
        lack_bonus = (
            1.0
            + 0.35 * lack
        )

        score *= lack_bonus

        # Satisfaction
        satisfaction_penalty = (
            1.0
            - 0.30
            * satisfaction
            * (1.0 - proximity)
        )

        score *= max(
            0.70,
            satisfaction_penalty,
        )

        memory_adjustment = (
            self.get_memory_adjustment(
                position=position,
                value=value,
            )
        )

        score *= memory_adjustment

        return float(score)
        # =========================================================
        # Target
        # =========================================================

    def select_target(self):

        if not self.env.resources:
            return None

        best_resource = None
        best_score = float("-inf")

        for position, value in (
            self.env.resources.items()
        ):

            score = (
                self.get_resource_score(
                    position,
                    value,
                )
            )

            if score > best_score:

                best_score = score
                best_resource = position

        return best_resource

    # =========================================================
    # Action Toward
    # =========================================================

    def action_toward(self, target):

        if target is None:
            return 4

        current = self.env.positions[
            self.agent_name
        ]

        current_x = int(
            current[0]
        )

        current_y = int(
            current[1]
        )

        target_x = int(
            target[0]
        )

        target_y = int(
            target[1]
        )

        if current_x < target_x:
            return 3

        if current_x > target_x:
            return 2

        if current_y < target_y:
            return 1

        if current_y > target_y:
            return 0

        return 4

    # =========================================================
    # Learned Action Adjustment
    # =========================================================

    def get_action_learning_bonus(
        self,
        action,
    ):

        value = (
            self.get_action_memory_value(
                action
            )
        )

        return (
            self.action_memory_weight
            * value
        )

    # =========================================================
    # Alternative Actions
    # =========================================================

    def get_action_candidates(
        self,
        target,
    ):
        """
        Candidate actions:
            0 = up
            1 = down
            2 = left
            3 = right
            4 = stay
        """

        preferred = (
            self.action_toward(
                target
            )
        )

        candidates = [
            preferred
        ]

        # Add all movement actions.
        for action in (
            0,
            1,
            2,
            3,
        ):
            if action not in candidates:
                candidates.append(
                    action
                )

        return candidates

    # =========================================================
    # Action Selection
    # =========================================================

    def select_action(self, target):

        preferred = (
            self.action_toward(target)
        )

        motivation = (
            self.get_motivation_state()
        )

        satisfaction = float(
            motivation.get(
                "satisfaction",
                0.0,
            )
        )

        urgency = float(
            motivation.get(
                "urgency",
                0.0,
            )
        )

        # -----------------------------------------------------
        # High satisfaction:
        # less aggressive movement
        # -----------------------------------------------------

        if (
            satisfaction > 0.70
            and urgency < 0.30
        ):

            if preferred != 4:

                # Occasionally stay instead of
                # immediately pursuing a resource.
                if np.random.random() < (
                    0.20 * satisfaction
                ):
                    return 4

        # -----------------------------------------------------
        # High urgency:
        # commit strongly to target
        # -----------------------------------------------------

        if urgency > 0.70:

            return preferred

        # -----------------------------------------------------
        # Existing action-memory logic
        # -----------------------------------------------------

        risk = (
            self.get_risk_tolerance()
        )

        preferred_memory = (
            self.get_action_memory_value(
                preferred
            )
        )

        if (
            preferred_memory < -0.50
            and risk < 0.30
        ):

            candidates = (
                self.get_action_candidates(
                    target
                )
            )

            best_action = 4
            best_score = float(
                "-inf"
            )

            for action in candidates:

                action_value = (
                    self.get_action_learning_bonus(
                        action
                    )
                )

                if action == preferred:
                    action_value += 0.05

                if action == 4:
                    action_value -= 0.02

                if action_value > best_score:

                    best_score = action_value
                    best_action = action

            return best_action

        return preferred
    
    # =========================================================
    # Diagnostics
    # =========================================================

    def get_learning_diagnostics(self):

        memory = self.get_memory()

        return {
            "memory_size": len(memory),
            "memory_reward": self.get_memory_reward(),
            "recent_reward": self.get_recent_reward(),
            "risk_tolerance": self.get_risk_tolerance(),

            "motivation":
                self.get_motivation_state(),

            "steps_since_collection":
                self.steps_since_collection,

            "last_collected_value":
                self.last_collected_value,
        }
    # =========================================================
    # Act
    # =========================================================

    def act(self, observation):

        target = (
            self.select_target()
        )

        action = (
            self.select_action(
                target
            )
        )

        self.record_action(
            action
        )

        return action

    # =========================================================
    # Motivation State - Day 3
    # =========================================================

    def get_motivation_state(self):
        """
        Return current internal motivation state.

        All values are normalized to [0, 1].
        """

        return self.motivation_state.copy()

    # =========================================================
    # Motivation Update
    # =========================================================

    def update_motivation(
        self,
        reward=0.0,
        collected_resource=0.0,
    ):
        """
        Update motivation after each environment step.

        Interpretation:

        Lack:
            increases when the agent fails to collect.

        Desire:
            follows lack and recent unsuccessful search.

        Satisfaction:
            increases after successful collection
            and decays gradually.

        Urgency:
            increases when the agent has not collected
            anything for several steps.
        """

        reward = max(
            0.0,
            float(reward),
        )

        collected_resource = max(
            0.0,
            float(collected_resource),
        )

        # -----------------------------------------------------
        # Collection event
        # -----------------------------------------------------

        if collected_resource > 0.0 or reward > 0.0:

            self.steps_since_collection = 0

            self.last_collected_value = max(
                collected_resource,
                reward,
            )

        else:

            self.steps_since_collection += 1

        # -----------------------------------------------------
        # Satisfaction
        # -----------------------------------------------------

        satisfaction_gain = min(
            1.0,
            self.last_collected_value / 50.0,
        )

        self.motivation_state["satisfaction"] = (
            self.satisfaction_decay
            * self.motivation_state["satisfaction"]
            + (1.0 - self.satisfaction_decay)
            * satisfaction_gain
        )

        # -----------------------------------------------------
        # Lack
        # -----------------------------------------------------

        if self.steps_since_collection == 0:

            self.motivation_state["lack"] *= (
                self.motivation_decay
            )

        else:

            lack_growth = min(
                0.05,
                0.01
                * self.steps_since_collection,
            )

            self.motivation_state["lack"] = min(
                1.0,
                self.motivation_state["lack"]
                + lack_growth,
            )

        # -----------------------------------------------------
        # Desire
        # -----------------------------------------------------

        lack = self.motivation_state["lack"]
        satisfaction = self.motivation_state[
            "satisfaction"
        ]

        desire = (
            0.70 * lack
            + 0.30 * (1.0 - satisfaction)
        )

        self.motivation_state["desire"] = max(
            0.0,
            min(1.0, desire),
        )

        # -----------------------------------------------------
        # Urgency
        # -----------------------------------------------------

        urgency = (
            self.steps_since_collection
            * self.urgency_growth
        )

        # Lack also contributes to urgency.

        urgency += (
            0.35
            * self.motivation_state["lack"]
        )

        # Satisfaction suppresses urgency.

        urgency *= (
            1.0
            - 0.60
            * satisfaction
        )

        self.motivation_state["urgency"] = max(
            0.0,
            min(1.0, urgency),
        )
        
        # =========================================================
    # Experience
    # =========================================================

    def record_experience(
        self,
        action,
        reward,
        info,
    ):
        """
        Record experience and update motivation.
        """

        reward = float(
            reward or 0.0
        )

        if not isinstance(info, dict):
            info = {}

        collected_resource = float(
            info.get(
                "collected_resource",
                0.0,
            )
            or 0.0
        )

        # -----------------------------------------------------
        # Update motivation FIRST
        # -----------------------------------------------------

        self.update_motivation(
            reward=reward,
            collected_resource=(
                collected_resource
            ),
        )

        # -----------------------------------------------------
        # Preserve existing BaseAgent memory
        # -----------------------------------------------------

        super().record_experience(
            action=action,
            reward=reward,
            info=info,
        )
        
        resource_location = info.get(
            "resource_location"
        )

        resource_value = info.get(
            "resource_value",
            0.0,
        )

        competition = info.get(
            "competition",
            0.0,
        )

        outcome = info.get(
            "outcome",
            "unknown",
        )

        if resource_location is not None:

            distance = self.get_distance(
                resource_location
            )

            self.remember_experience(
                resource_location=(
                    resource_location
                ),
                resource_value=(
                    resource_value
                ),
                distance=distance,
                action=action,
                reward=float(reward),
                competition=competition,
                outcome=outcome,
            )

    def get_risk_adjustment(self, *args, **kwargs):
        """
        Day 3 compatibility hook.

        Risk modeling هنوز در این مرحله فعال نشده است.
        بنابراین مقدار خنثی برمی‌گرداند تا منطق
        Motivation بتواند بدون تغییر رفتار پایه اجرا شود.
        """
        return 0.0
    
    def get_resource_competition(
    self,
    position,
    ):
        """
        Day 4 - Competition

        تخمین Competition برای یک Resource
        بر اساس فاصله سایر Agentها از Resource.

        Agent نزدیک‌تر به Resource
        -> Competition بیشتر
        """

        position = tuple(position)

        competition = 0.0

        for agent_name in self.env.agents:

            if agent_name == self.agent_name:
                continue

            if agent_name not in self.env.positions:
                continue

            other_position = tuple(
                self.env.positions[agent_name]
            )

        # Manhattan distance
            distance = (
                abs(
                    position[0]
                    - other_position[0]
                )
                +
                abs(
                    position[1]
                    - other_position[1]
                )
            )

            competition += (
                1.0
                / (1.0 + distance)
            )

        return float(competition)
    
    def get_resource_risk(
    self,
    position,
    ):
        """
        Day 4 Risk Model

        Risk = Distance + Competition
        """

        distance = float(
            self.get_distance(position)
        )

        competition = float(
            self.get_resource_competition(
                position
            )
        )

        risk = (
            self.risk_distance_weight
            * distance
            +
            self.risk_competition_weight
            * competition
        )

        return float(risk)
    
    def get_risk_tolerance(self):
        """
        Risk tolerance تحت تأثیر Motivation قرار می‌گیرد.

        Lack بالا
            -> تحمل ریسک بیشتر

        Satisfaction بالا
            -> تحمل ریسک کمتر
        """

        motivation = (
            self.get_motivation_state()
        )

        lack = float(
            motivation.get("lack", 0.0)
        )

        satisfaction = float(
            motivation.get(
                "satisfaction",
                0.0,
            )
        )

        tolerance = (
            self.base_risk_tolerance
            +
            self.lack_risk_tolerance_weight
            * lack
            -
            self.satisfaction_risk_tolerance_weight
            * satisfaction
        )

        return float(
            np.clip(
                tolerance,
                0.0,
                1.0,
            )
        )
        
    def get_risk_adjustment(
        self,
        risk,
    ):
        """
        تبدیل Risk به ضریب قابل استفاده در Score.

        Risk بالاتر از tolerance
            -> penalty بیشتر

        Risk پایین
            -> penalty کمتر
        """

        risk = max(
            0.0,
            float(risk),
        )

        tolerance = max(
            0.05,
            self.get_risk_tolerance(),
        )

        normalized_risk = (
            risk
            / tolerance
        )

        penalty = (
            1.0
            +
            self.risk_penalty_weight
            * normalized_risk
        )

        return float(
            max(
                1.0,
                penalty,
            )
        )
        
    def remember_experience(
        self,
        resource_location,
        resource_value,
        distance,
        action,
        reward,
        competition,
        outcome,
    ):
        """
        Day 5 - Store one compact experience.
        """

        experience = {
            "resource_location": tuple(
                resource_location
            ),
            "resource_value": float(
                resource_value
            ),
            "distance": float(
                distance
            ),
            "action": action,
            "reward": float(
                reward
            ),
            "competition": float(
                competition
            ),
            "outcome": str(
                outcome
            ),
        }

        self.experience_memory.append(
            experience
        )

        self.current_episode_experiences.append(
            experience
        )

    # محدود کردن حافظه
        if len(
            self.experience_memory
        ) > self.memory_max_size:

            self.experience_memory = (
                self.experience_memory[
                    -self.memory_max_size:
                ]
            )
            
    def summarize_episode_memory(self):
        """
        Day 5 - Create a compact summary
        of the current episode.
        """

        experiences = (
            self.current_episode_experiences
        )

        if not experiences:
            return {
                "steps": 0,
                "total_reward": 0.0,
                "mean_reward": 0.0,
                "successful_actions": 0,
                "success_rate": 0.0,
                "mean_distance": 0.0,
                "mean_competition": 0.0,
            }

        total_reward = sum(
            exp["reward"]
            for exp in experiences
        )

        successful_actions = sum(
            1
            for exp in experiences
            if exp["reward"] > 0
        )

        count = len(experiences)

        summary = {
            "steps": count,

            "total_reward": float(
                total_reward
            ),

            "mean_reward": float(
                total_reward / count
            ),

            "successful_actions":
                successful_actions,

            "success_rate": float(
                successful_actions / count
            ),

            "mean_distance": float(
                sum(
                    exp["distance"]
                    for exp in experiences
                )
                / count
            ),

            "mean_competition": float(
                sum(
                    exp["competition"]
                    for exp in experiences
                )
                / count
            ),
        }

        self.completed_episode_memory.append(
            summary
        )

        # Episode summaries هم محدود باشند
        if len(
            self.completed_episode_memory
        ) > 20:

            self.completed_episode_memory = (
                self.completed_episode_memory[-20:]
            )

        self.current_episode_experiences = []

        return summary
        
    def get_memory_adjustment(
        self,
        position,
        value,
        action=None,
    ):
        """
        Day 5 - Experience-based adjustment.

        تجربه‌های قبلی مشابه را بررسی می‌کند
        و در صورت موفقیت، Score را کمی تقویت می‌کند.
        """

        if not self.experience_memory:
            return 1.0

        position = tuple(position)

        relevant = []

        for experience in self.experience_memory:

            same_location = (
                experience[
                    "resource_location"
                ]
                == position
            )

            similar_value = (
                abs(
                    experience[
                        "resource_value"
                    ] - value
                )
                <= 10.0
            )

            if (
                same_location
                or similar_value
            ):
                relevant.append(
                    experience
                )

        if not relevant:
            return 1.0

        rewards = [
            exp["reward"]
            for exp in relevant
        ]

        mean_reward = (
            sum(rewards)
            / len(rewards)
        )

        # adjustment محدود نگه داشته می‌شود
        adjustment = (
            1.0
            + 0.20
            * np.tanh(
                mean_reward / 50.0
            )
        )

        return float(
            np.clip(
                adjustment,
                0.80,
                1.20,
            )
        )
        
    
