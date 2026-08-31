from agents.base_agent import BaseAgent


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

    def get_motivation_multiplier(self):

        motivation = (
            self.get_motivation_state()
        )

        desire = float(
            motivation.get(
                "desire",
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

        signal = (
            0.55 * desire
            + 0.35 * urgency
            - 0.10 * satisfaction
        )

        return max(
            0.0,
            1.0
            + self.motivation_weight
            * signal,
        )

    # =========================================================
    # Risk Adjustment
    # =========================================================

    def get_risk_adjustment(
        self,
        distance,
    ):

        risk = (
            self.get_risk_tolerance()
        )

        return (
            distance
            * (
                1.0
                - 0.70 * risk
            )
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

        distance_cost = (
            self.get_risk_adjustment(
                distance
            )
            + self.distance_weight
        )

        base_score = (
            expected_value
            / max(
                0.1,
                distance_cost,
            )
        )

        # -----------------------------------------------------
        # Motivation
        # -----------------------------------------------------

        motivation_multiplier = (
            self.get_motivation_multiplier()
        )

        score = (
            base_score
            * motivation_multiplier
        )

        # -----------------------------------------------------
        # Location Memory
        # -----------------------------------------------------

        location_memory = (
            self.get_location_memory_value(
                position
            )
        )

        if location_memory > 0.0:

            memory_ratio = (
                location_memory
                / max(
                    1.0,
                    value,
                )
            )

            memory_ratio = max(
                0.0,
                min(
                    2.0,
                    memory_ratio,
                ),
            )

            # Confidence based on number of observations.
            visits = (
                self.get_location_visit_count(
                    position
                )
            )

            confidence = min(
                1.0,
                visits / 10.0,
            )

            memory_bonus = (
                self.memory_weight
                * base_score
                * memory_ratio
                * confidence
            )

            score += memory_bonus

        # -----------------------------------------------------
        # Learning
        # -----------------------------------------------------

        learning_signal = (
            self.get_location_learning_signal(
                position,
                value,
            )
        )

        learning_bonus = (
            self.learning_weight
            * base_score
            * learning_signal
        )

        score += learning_bonus

        # -----------------------------------------------------
        # Novelty / Exploration
        # -----------------------------------------------------

        novelty = (
            self.get_location_novelty(
                position
            )
        )

        exploration_bonus = (
            self.exploration_weight
            * base_score
            * novelty
        )

        score += exploration_bonus

        # -----------------------------------------------------
        # Satisfaction
        # -----------------------------------------------------

        motivation = (
            self.get_motivation_state()
        )

        satisfaction = float(
            motivation.get(
                "satisfaction",
                0.0,
            )
        )

        if satisfaction > 0.0:

            score *= max(
                0.50,
                1.0
                - 0.20 * satisfaction,
            )

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
            self.action_toward(
                target
            )
        )

        risk = (
            self.get_risk_tolerance()
        )

        preferred_memory = (
            self.get_action_memory_value(
                preferred
            )
        )

        # A strongly negative historical action
        # should be avoided when risk tolerance is low.
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
            "motivation": self.get_motivation_state(),
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
