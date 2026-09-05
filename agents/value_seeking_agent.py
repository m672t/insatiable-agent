from agents.base_agent import BaseAgent
import numpy as np
from agents.competition_strategy import CompetitionStrategy

class ValueSeekingAgent(BaseAgent):
    """
    Value Seeking Agent

    هدف:
        انتخاب منابع بر اساس ارزش مورد انتظار، فاصله،
        انگیزش، حافظه، رقابت، ریسک و اکتشاف.

    معماری حافظه:
        فقط یک Memory System وجود دارد:

            BaseAgent
                |
                v
            InternalState
                |
                v
            memory

        ValueSeekingAgent مستقیماً حافظه جداگانه‌ای ندارد.

    Decision factors:
        1. Resource Value
        2. Distance
        3. Motivation
        4. Location Memory
        5. Action Memory
        6. Risk Tolerance
        7. Recent Experience Learning
        8. Exploration / Exploitation
        9. Competition
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
        motivation_model=None,
    ):
        super().__init__(
            env=env,
            agent_name=agent_name,
            internal_state=internal_state,
            motivation_model=motivation_model,
        )

        self.distance_weight = float(distance_weight)
        self.motivation_weight = float(motivation_weight)
        self.memory_weight = float(memory_weight)
        self.risk_weight = float(risk_weight)
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

        # -----------------------------------------------------
        # Motivation
        # -----------------------------------------------------

        self.steps_since_collection = 0
        self.last_collected_value = 0.0

        self.motivation_decay = 0.95
        self.urgency_growth = 0.025
        self.satisfaction_decay = 0.90

        # -----------------------------------------------------
        # Risk
        # -----------------------------------------------------

        self.risk_distance_weight = 1.0
        self.risk_competition_weight = 1.0

        self.base_risk_tolerance = 0.50

        self.lack_risk_tolerance_weight = 0.40
        self.satisfaction_risk_tolerance_weight = 0.40

        self.risk_penalty_weight = 0.50
        self.competition_strategy = CompetitionStrategy()

    # =========================================================
    # Episode Lifecycle
    # =========================================================

    def reset_episode(self):
        """
        Reset وضعیت کوتاه‌مدت Agent.

        Memory پاک نمی‌شود.

        بنابراین:

            action_history -> reset
            short-term state -> reset
            long-term memory -> preserved
            hidden motivation -> preserved according
            to MotivationModel policy
        """

        super().reset_episode()

        self.steps_since_collection = 0
        self.last_collected_value = 0.0

        if hasattr(
            self.motivation_model,
            "reset_episode",
        ):
            self.motivation_model.reset_episode()

    # =========================================================
    # Distance
    # =========================================================

    def get_distance(self, position):
        current = self.env.positions[self.agent_name]

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
    # Unified Memory Access
    # =========================================================

    def get_memory(self):
        """
        تنها منبع حافظه Agent.

        توجه:
            هیچ experience_memory جداگانه‌ای وجود ندارد.
        """

        return self.internal_state.get_memory()

    # =========================================================
    # Safe Reward
    # =========================================================

    def _get_reward(self, experience):
        if not isinstance(experience, dict):
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
        if not isinstance(experience, dict):
            return None

        position = experience.get(
            "position"
        )

        if position is None:
            info = experience.get(
                "info",
                {},
            )

            if isinstance(info, dict):
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
        if not isinstance(experience, dict):
            return 0.0

        collected = experience.get(
            "collected_resource"
        )

        if collected is None:
            info = experience.get(
                "info",
                {},
            )

            if isinstance(info, dict):
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
    # Resource Value Extraction
    # =========================================================

    def _get_resource_value(
        self,
        experience,
    ):
        if not isinstance(experience, dict):
            return 0.0

        value = experience.get(
            "resource_value",
            0.0,
        )

        try:
            return max(
                0.0,
                float(value),
            )
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    # =========================================================
    # Competition Extraction
    # =========================================================

    def _get_competition(
        self,
        experience,
    ):
        if not isinstance(experience, dict):
            return 0.0

        value = experience.get(
            "competition",
            0.0,
        )

        try:
            return max(
                0.0,
                float(value),
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

        rewards = [
            self._get_reward(exp)
            for exp in memory
            if isinstance(exp, dict)
        ]

        if not rewards:
            return 0.0

        return sum(rewards) / len(rewards)

    # =========================================================
    # Recency
    # =========================================================

    def get_recency_weight(
        self,
        index,
        total,
    ):
        """
        تجربه‌های جدیدتر وزن بیشتری دارند.
        """

        if total <= 0:
            return 0.0

        age = total - 1 - index

        return self.recency_decay ** age

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

            weight = self.get_recency_weight(
                index,
                total,
            )

            weighted_sum += reward * weight
            weight_sum += weight

        if weight_sum <= 0.0:
            return 0.0

        return weighted_sum / weight_sum

    # =========================================================
    # Recent Reward
    # =========================================================

    def get_recent_reward(
        self,
        window=100,
    ):
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
        ارزش یادگرفته‌شده Action.

        خروجی:
            [-1, +1]
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

            if previous_action == int(action):
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

        normalized = (
            average_reward
            /
            (
                abs(average_reward)
                + 50.0
            )
        )

        return float(
            np.clip(
                normalized,
                -1.0,
                1.0,
            )
        )

    # =========================================================
    # Location Memory
    # =========================================================

    def get_location_memory_value(
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

        return self.get_weighted_resource_value(
            matching
        )

    # =========================================================
    # Weighted Resource Value
    # =========================================================

    def get_weighted_resource_value(
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

            resource_value = (
                self._get_resource_value(
                    experience
                )
            )

            # اگر resource_value ذخیره نشده بود،
            # collected_resource fallback است.
            observed_value = max(
                collected,
                resource_value,
            )

            weight = self.get_recency_weight(
                index,
                total,
            )

            weighted_sum += (
                observed_value * weight
            )

            weight_sum += weight

        if weight_sum <= 0.0:
            return 0.0

        return weighted_sum / weight_sum

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

        return sum(
            1
            for experience in memory
            if self._get_experience_position(
                experience
            )
            == target
        )

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

        visits = 0
        successes = 0

        for experience in memory:
            if (
                self._get_experience_position(
                    experience
                )
                != target
            ):
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

        return successes / visits

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

        signal = (
            memory_value / current_value
        ) - 1.0

        return float(
            np.clip(
                signal,
                -1.0,
                1.0,
            )
        )

    # =========================================================
    # Novelty
    # =========================================================

    def get_location_novelty(
        self,
        position,
    ):
        visits = (
            self.get_location_visit_count(
                position
            )
        )

        if visits <= 0:
            return 1.0

        return 1.0 / (1.0 + visits)

    # =========================================================
    # Motivation
    # =========================================================

    def get_motivation_state(self):
        """
        Motivation از BaseAgent/MotivationModel می‌آید.

        ValueSeekingAgent دیگر یک motivation state
        مستقل ایجاد نمی‌کند.
        """

        return super().get_motivation_state()

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

        return float(
            np.clip(
                1.0
                + self.motivation_weight
                * signal,
                0.50,
                2.00,
            )
        )

    # =========================================================
    # Expected Location Value
    # =========================================================

    def get_expected_location_value(
        self,
        position,
        current_value,
    ):
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
    # Competition
    # =========================================================

    def get_resource_competition(
        self,
        position,
    ):
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

    # =========================================================
    # Risk
    # =========================================================

    def get_risk_tolerance(self, position=None):
        motivation = self.get_motivation_state()

        lack = float(
            motivation.get("lack", 0.0)
        ) 

        satisfaction = float(
            motivation.get("satisfaction", 0.0)
        )

        tolerance = (
            self.base_risk_tolerance
            + self.lack_risk_tolerance_weight * lack
            - self.satisfaction_risk_tolerance_weight * satisfaction
        )

        tolerance += self.get_experience_risk_modifier(
            position=position
        )

        tolerance += self.get_personality_risk_modifier()

        tolerance += self.get_social_risk_modifier()

        risk_tolerance = float(
            np.clip(
                tolerance,
                0.0,
                1.0,
            )
        )

        if hasattr(
            self,
            "competition_strategy",
        ):
            self.competition_strategy.risk_tolerance = (
                risk_tolerance
            )

        return risk_tolerance
    
    def get_resource_risk(
        self,
        position,
    ):
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

        return float(
            np.clip(
                risk / 10.0,
                0.0,
                1.0,
            )
        )

    def get_risk_adjustment(
        self,
        risk,
    ):
        risk = max(
            0.0,
            float(risk),
        )

        tolerance = max(
            0.05,
            self.get_risk_tolerance(),
        )

        normalized_risk = (
            risk / tolerance
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

    # =========================================================
    # Memory Adjustment
    # =========================================================

    def get_memory_adjustment(
        self,
        position,
        value,
        action=None,
    ):
        """
        Experience-based adjustment.

        از همان InternalState.memory استفاده می‌کند.
        """

        memory = self.get_memory()

        if not memory:
            return 1.0

        position = tuple(position)

        relevant = []

        for experience in memory:
            if not isinstance(
                experience,
                dict,
            ):
                continue

            same_location = (
                self._get_experience_position(
                    experience
                )
                == position
            )

            similar_value = (
                abs(
                    self._get_resource_value(
                        experience
                    )
                    - float(value)
                )
                <= 10.0
            )

            if same_location or similar_value:
                relevant.append(
                    experience
                )

        if not relevant:
            return 1.0

        mean_reward = (
            self.get_weighted_reward(
                relevant
            )
        )

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

    # =========================================================
    # Resource Score
    # =========================================================

        # =========================================================
    # Resource Score
    # =========================================================

    def get_resource_score(
        self,
        position,
        value,
    ):
        value = max(
            0.0,
            float(value),
        )

        distance = self.get_distance(
            position
        )

        expected_value = (
            self.get_expected_location_value(
                position,
                value,
            )
        )

        # -----------------------------------------------------
        # Risk
        # -----------------------------------------------------

        risk = self.get_resource_risk(
            position
        )

        competition = self.get_resource_competition(
            position
        )

        # -----------------------------------------------------
        # Competition Strategy
        # -----------------------------------------------------

        strategy = self.get_competition_strategy(
            resource_value=value,
            resource_location=position,
        )

        strategy_modifier = (
            self.competition_strategy.get_priority_modifier(
                strategy
            )
        )

        # -----------------------------------------------------
        # Risk Cost
        # -----------------------------------------------------

        risk_adjustment = (
            self.get_risk_adjustment(
                risk
            )
        )

        distance_cost = (
            self.distance_weight
            + risk_adjustment
        )

        base_score = (
            expected_value
            /
            max(
                0.1,
                distance_cost,
            )
        )

        # -----------------------------------------------------
        # Competition Penalty
        # -----------------------------------------------------
        #
        # Competition باید مستقیماً روی ترجیح Resource
        # اثر بگذارد.
        #
        # competition = 0.0
        #     => بدون جریمه
        #
        # competition = 1.0
        #     => جریمه محسوس
        #
        # این ضریب مستقل از Strategy است تا حتی اگر
        # Strategy = "approach" شد، وجود رقیب نادیده
        # گرفته نشود.
        #

        competition_modifier = (
            1.0
            /
            (
                1.0
                + 0.50
                * max(
                    0.0,
                    float(competition),
                )
            )
        )

        score = (
            base_score
            * strategy_modifier
            * competition_modifier
        )

        # -----------------------------------------------------
        # Motivation
        # -----------------------------------------------------

        motivation = (
            self.get_motivation_state()
        )

        lack = float(
            motivation.get(
                "lack",
                0.0,
            )
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

        value_ratio = (
            value / 50.0
        )

        # -----------------------------------------------------
        # Desire
        # -----------------------------------------------------

        score *= (
            1.0
            + 0.80
            * desire
            * value_ratio
        )

        # -----------------------------------------------------
        # Urgency
        # -----------------------------------------------------

        proximity = 1.0 / (
            1.0 + distance
        )

        score *= (
            1.0
            + 1.20
            * urgency
            * proximity
            * value_ratio
        )

        # -----------------------------------------------------
        # Lack
        # -----------------------------------------------------

        score *= (
            1.0
            + 0.35 * lack
        )

        # -----------------------------------------------------
        # Satisfaction
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Memory
        # -----------------------------------------------------

        memory_adjustment = (
            self.get_memory_adjustment(
                position=position,
                value=value,
            )
        )

        score *= memory_adjustment

        # -----------------------------------------------------
        # Location Learning
        # -----------------------------------------------------

        learning_signal = (
            self.get_location_learning_signal(
                position,
                value,
            )
        )

        score *= (
            1.0
            + self.learning_weight
            * learning_signal
        )

        # -----------------------------------------------------
        # Novelty
        # -----------------------------------------------------

        novelty = (
            self.get_location_novelty(
                position
            )
        )

        
        exploration_pressure = (
            self.get_exploration_pressure()
        )

        risk_exploration_factor = (
            self.get_risk_exploration_pressure(
                position
            )
        )

        effective_exploration_pressure = (
            exploration_pressure
            * risk_exploration_factor
        )

        score += (
            self.novelty_weight
            * effective_exploration_pressure
            * novelty
            * max(
                1.0,
                value,
            )
        )


        return float(
            max(
                0.0,
                score,
            )
        )
    
    # =========================================================
    # Target Selection
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

        current_x = int(current[0])
        current_y = int(current[1])

        target_x = int(target[0])
        target_y = int(target[1])

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
    # Action Memory
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
    # Action Candidates
    # =========================================================

    def get_action_candidates(
        self,
        target,
    ):
        preferred = (
            self.action_toward(target)
        )

        candidates = [preferred]

        for action in (
            0,
            1,
            2,
            3,
        ):
            if action not in candidates:
                candidates.append(action)

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

        # High satisfaction:
        # occasionally stop.
        if (
            satisfaction > 0.70
            and urgency < 0.30
            and preferred != 4
        ):
            if np.random.random() < (
                0.20 * satisfaction
            ):
                return 4

        # High urgency:
        # pursue target directly.
        if urgency > 0.70:
            return preferred

        # -----------------------------------------------------
        # Action memory
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

            best_action = preferred
            best_score = float("-inf")

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
    # Motivation / Experience
    # =========================================================

    def update_motivation(
        self,
        reward=0.0,
        collected_resource=0.0,
    ):
        """
        سازگاری با منطق قبلی.

        وضعیت اصلی انگیزش توسط InternalState/
        MotivationModel مدیریت می‌شود.
        """

        reward = max(
            0.0,
            float(reward),
        )

        collected_resource = max(
            0.0,
            float(collected_resource),
        )

        if (
            collected_resource > 0.0
            or reward > 0.0
        ):
            self.steps_since_collection = 0

            self.last_collected_value = max(
                collected_resource,
                reward,
            )
        else:
            self.steps_since_collection += 1

    # =========================================================
    # Experience Recording
    # =========================================================

    def record_experience(
        self,
        action,
        reward,
        info=None,
    ):
        """
        تجربه فقط یک بار و در Memory واحد ثبت می‌شود.

        جریان:

            ValueSeekingAgent
                    |
                    v
              BaseAgent.record_experience
                    |
                    v
              InternalState.memory

        هیچ حافظه دومی ساخته نمی‌شود.
        """

        if not isinstance(info, dict):
            info = {}

        reward = float(
            reward or 0.0
        )

        collected_resource = float(
            info.get(
                "collected_resource",
                0.0,
            )
            or 0.0
        )

        self.update_motivation(
            reward=reward,
            collected_resource=collected_resource,
        )

        # -----------------------------------------------------
        # Hidden motivation update
        # -----------------------------------------------------

        if hasattr(
            self.motivation_model,
            "update_from_experience",
        ):
            current_motivation = (
                self.get_motivation_state()
            )

            competition = float(
                info.get(
                    "competition",
                    0.0,
                )
                or 0.0
            )

            novelty = float(
                info.get(
                    "novelty",
                    0.0,
                )
                or 0.0
            )

            success = (
                1.0
                if (
                    collected_resource > 0.0
                    or reward > 0.0
                )
                else 0.0
            )

            self.motivation_model.update_from_experience(
                lack=current_motivation.get(
                    "lack",
                    0.0,
                ),
                satisfaction=current_motivation.get(
                    "satisfaction",
                    0.0,
                ),
                competition=np.clip(
                    competition,
                    0.0,
                    1.0,
                ),
                novelty=np.clip(
                    novelty,
                    0.0,
                    1.0,
                ),
                success=success,
            )

        # -----------------------------------------------------
        # SINGLE MEMORY WRITE
        # -----------------------------------------------------

        super().record_experience(
            action=action,
            reward=reward,
            info=info,
        )

    # =========================================================
    # Diagnostics
    # =========================================================

    def get_learning_diagnostics(self):
        memory = self.get_memory()

        return {
            "memory_size": len(memory),
            "memory_reward": (
                self.get_memory_reward()
            ),
            "recent_reward": (
                self.get_recent_reward()
            ),
            "risk_tolerance": (
                self.get_risk_tolerance()
            ),
            "motivation": (
                self.get_motivation_state()
            ),
            "steps_since_collection": (
                self.steps_since_collection
            ),
            "last_collected_value": (
                self.last_collected_value
            ),
        }

    # =========================================================
    # Act
    # =========================================================

    def act(self, observation):
        target = self.select_target()

        action = self.select_action(
            target
        )

        self.record_action(
            action
        )

        return action


# =========================================================
# Exploration Pressure
# =========================================================
    

# =========================================================
# Exploration Pressure
# =========================================================

    def get_exploration_pressure(self):
        """
        Dynamic Exploration / Exploitation balance.

        Motivation affects exploration:

            satisfaction ↑ -> exploration ↑
            lack ↑         -> exploitation ↑
            urgency ↑      -> exploitation ↑
            desire ↑       -> exploitation ↑

        Recent failures increase exploration pressure.

        Returns:
            0.0 -> strong exploitation
            1.0 -> strong exploration
        """

        base_pressure = float(
            self.exploration_weight
        )

        motivation = (
            self.get_motivation_state()
        )

        lack = float(
            motivation.get(
                "lack",
                0.0,
            )
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

    # -----------------------------------------------------
    # Motivation
    # -----------------------------------------------------

        pressure = base_pressure

    # Satisfaction gives the agent freedom to explore.
        pressure += (
            0.25
            * satisfaction
        )

    # Lack favors exploitation of known resources.
        pressure -= (
            0.20
            * lack
        )

    # Urgency favors reliable exploitation.
        pressure -= (
            0.25
            * urgency
        )

    # Desire favors goal-directed exploitation.
        pressure -= (
            0.10
            * desire
        )

    # -----------------------------------------------------
    # Recent failures
    # -----------------------------------------------------

        recent_failure_pressure = (
            self.get_recent_failure_pressure()
        )

        pressure += (
            0.25
            * recent_failure_pressure
        )

        return float(
            np.clip(
                pressure,
                0.0,
                1.0,
            )
        )

    
# =========================================================
# Recent Failure Pressure
# =========================================================

    def get_recent_failure_pressure(
        self,
        window=10,
    ):
        """
        Measures pressure to explore after recent failures.

        Recent unsuccessful experiences increase exploration.

        Returns:
            0.0 -> no recent failure pressure
            1.0 -> strong recent failure pressure
        """

        memory = self.get_memory()

        if not memory:
            return 0.0

        window = max(
            1,
            int(window),
        )

        recent = memory[-window:]

        if not recent:
            return 0.0

        valid_experiences = []

        for experience in recent:
            if not isinstance(
                experience,
                dict,
            ):
                continue

            valid_experiences.append(
                experience
            )

        if not valid_experiences:
            return 0.0

        failures = 0

        for experience in valid_experiences:
            reward = self._get_reward(
                experience
            )

            collected = (
                self._get_collected_resource(
                    experience
                )
            )

        # An experience is considered successful
        # when it produced either reward or resource.
            success = (
                reward > 0.0
                or collected > 0.0
            )

            if not success:
                failures += 1

        failure_rate = (
            failures
            /
            len(valid_experiences)
        )

        return float(
            np.clip(
                failure_rate,
                0.0,
                1.0,
            )
        )


# =========================================================
# Risk Exploration Pressure
# =========================================================

    def get_risk_exploration_pressure(
        self,
        position=None,
    ):
        """
        Risk-based adjustment for exploration.

        High risk makes the agent more cautious.
        Low risk allows more exploration.

        Returns:
            0.0 -> risk strongly suppresses exploration
            1.0 -> risk has little/no suppressive effect
        """

        if position is None:
            return 1.0

        risk = self.get_resource_risk(
            position
        ) 

        tolerance = max(
            0.05,
            self.get_risk_tolerance(),
        )

        relative_risk = (
            risk / tolerance
        )

    # As relative risk increases,
    # exploration pressure decreases.
        exploration_factor = (
            1.0
            /
            (
                1.0
                + relative_risk
            )
        )

        return float(
            np.clip(
                exploration_factor,
                0.0,
                1.0,
            )
        )
        
    def get_experience_risk_modifier(self, position=None):
        """
        تعیین میزان اثر تجربه بر ریسک‌پذیری.

        تجربه موفق در شرایط مشابه:
            افزایش تحمل ریسک

        تجربه ناموفق:
            کاهش تحمل ریسک

        بدون تجربه:
            بدون تغییر
        """

        memory = getattr(self.internal_state, "memory", None)

        if memory is None:
            return 0.0

        try:
            experiences = memory.get_all()
        except (AttributeError, TypeError):
            return 0.0

        if not experiences:
            return 0.0

        relevant = []

        for experience in experiences:
            if not isinstance(experience, dict):
                continue

            # اگر Position مشخص شده، تجربه‌های همان موقعیت
            # را مهم‌تر و دقیق‌تر در نظر می‌گیریم.
            if position is not None:
                experience_position = experience.get(
                    "position",
                    experience.get("resource_location"),
                )

                if experience_position is not None:
                    try:
                        if tuple(experience_position) != tuple(position):
                            continue
                    except TypeError:
                        continue

            relevant.append(experience)
  
        if not relevant:
            return 0.0

        successes = 0
        failures = 0

        for experience in relevant:
            collected = experience.get(
                "collected_resource",
                False,
            )

            reward = experience.get(
                "reward",
                0.0,
            )

            outcome = experience.get(
                "outcome",
                None,
            )

            try:
                reward = float(reward)
            except (TypeError, ValueError):
                reward = 0.0

            success = (
                bool(collected)
                or outcome == "success"
                or reward > 0.0
             )
   
            failure = (
                outcome == "failure"
                or reward < 0.0
                or (
                    not collected
                    and reward == 0.0
                )
            )

            if success:
                successes += 1
            elif failure:
                failures += 1

        total = successes + failures

        if total == 0:
            return 0.0

        success_rate = successes / total

    # بازه تقریبی:
    # -0.20 تا +0.20
        modifier = (
            success_rate - 0.50
        ) * 0.40

        return float(
            np.clip(
                modifier,
                -0.20,
                0.20,
            )
        )
        
    def get_experience_risk_modifier(self, position=None):
        """
        اثر تجربه قبلی بر Risk Tolerance.

        تجربه موفق  -> افزایش ریسک‌پذیری
        تجربه ناموفق -> کاهش ریسک‌پذیری
        بدون تجربه -> بدون اثر

        اگر position داده شود، فقط تجربیات مرتبط با همان
        Resource/Location بررسی می‌شوند.
        """

        memory = getattr(
            self.internal_state,
            "memory",
            None,
        )

        if memory is None:
            return 0.0

        experiences = memory.get_all()

        if not experiences:
            return 0.0

        relevant = []

        for experience in experiences:
            if not isinstance(experience, dict):
                continue

            if position is not None:
                experience_position = experience.get(
                    "resource_location"
                )

                if experience_position is None:
                    experience_position = experience.get(
                        "position"
                    )

                if experience_position is None:
                    continue

                try:
                    if tuple(experience_position) != tuple(position):
                        continue
                except (TypeError, ValueError):
                    continue

            relevant.append(experience)

        if not relevant:
            return 0.0

        successes = 0
        failures = 0

        for experience in relevant:
            try:
                reward = float(
                    experience.get("reward", 0.0)
                )
            except (TypeError, ValueError):
                reward = 0.0

            collected = experience.get(
                "collected_resource",
                0.0,
            )

            try:
                collected = float(collected)
            except (TypeError, ValueError):
                collected = 0.0

            outcome = experience.get(
                "outcome",
                "unknown",
            )

            if (
                collected > 0.0
                or reward > 0.0
                or outcome == "success"
            ):
                successes += 1
 
            elif (
                reward < 0.0
                or outcome == "failure"
                or (
                    collected <= 0.0
                    and reward == 0.0
                )
            ):
                failures += 1

        total = successes + failures

        if total == 0:
            return 0.0

        success_rate = successes / total

        modifier = (
            success_rate - 0.5
        ) * 0.40

        return float(
            np.clip(
                modifier,
                -0.20,
                0.20,
            )
        )
            
    def get_personality_risk_modifier(self):
        personality = getattr(self, "personality", "neutral")

        modifiers = {
            "risk_seeking": 0.20,
            "neutral": 0.0,
            "conservative": -0.20,
        }

        return modifiers.get(personality, 0.0)
    
    def get_social_risk_modifier(self):
        social_state = getattr(
            self,
            "social_state",
            {},
        )

        support = float(
            social_state.get("support", 0.0)
        )

        isolation = float(
            social_state.get("isolation", 0.0)
        )

        return (
            0.10 * support
            - 0.10 * isolation
        )
