class LearningSystem:
    """
    Learning system for the agent.

    MemorySystem stores experiences.
    LearningSystem derives reusable knowledge from them.

    Supported capabilities:
        - reward learning
        - action-value learning
        - recent experience
        - recency weighting
        - location learning
        - learning signal
        - failure learning
        - generalization
        - strategy adaptation
        - environment adaptation
    """

    def __init__(
        self,
        memory,
        learning_rate=0.20,
        recency_weight=0.995,
    ):
        self.memory = memory

        self.learning_rate = max(
            0.0,
            min(1.0, float(learning_rate)),
        )

        self.recency_weight = max(
            0.0,
            min(1.0, float(recency_weight)),
        )

        # -----------------------------------------------------
        # Learned action values
        # -----------------------------------------------------

        self.action_values = {}

        # -----------------------------------------------------
        # Learned location values
        # -----------------------------------------------------

        self.location_values = {}

        # -----------------------------------------------------
        # Strategy adaptation
        # -----------------------------------------------------

        self.strategy_bias = {}

        # -----------------------------------------------------
        # Environment knowledge
        # -----------------------------------------------------

        self.environment_model = {
            "average_reward": 0.0,
            "success_rate": 0.0,
            "failure_rate": 0.0,
        }

    # =========================================================
    # Utility
    # =========================================================

    @staticmethod
    def _clip(value):
        return max(
            0.0,
            min(1.0, float(value)),
        )

    # =========================================================
    # Reward Learning
    # =========================================================

    def get_reward_learning(self):
        """
        میانگین وزنی Reward از تجربیات.

        تجربیات جدیدتر وزن بیشتری دارند.
        """

        experiences = self.memory.get_all()

        if not experiences:
            return 0.0

        weighted_sum = 0.0
        weight_sum = 0.0

        total = len(experiences)

        for index, experience in enumerate(
            experiences
        ):
            reward = float(
                experience.get(
                    "reward",
                    0.0,
                )
            )

            age = (
                total
                - 1
                - index
            )

            weight = (
                self.recency_weight
                ** age
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
    # Action Value
    # =========================================================

    def update_action_value(
        self,
        action,
        reward,
    ):
        """
        یادگیری تدریجی ارزش یک Action.

        Q_new =
            Q_old + alpha * (reward - Q_old)
        """

        try:
            action = int(action)
        except (TypeError, ValueError):
            return 0.0

        try:
            reward = float(reward)
        except (TypeError, ValueError):
            reward = 0.0

        old_value = self.action_values.get(
            action,
            0.0,
        )

        new_value = (
            old_value
            + self.learning_rate
            * (reward - old_value)
        )

        self.action_values[action] = (
            new_value
        )

        return new_value

    def get_action_value(self, action):
        try:
            action = int(action)
        except (TypeError, ValueError):
            return 0.0

        # حافظه منبع اصلی دانش است.
        memory_value = (
            self.memory.get_action_value(
                action
            )
        )

        learned_value = (
            self.action_values.get(
                action,
                0.0,
            )
        )

        if action not in self.action_values:
            return memory_value

        return (
            0.5 * memory_value
            + 0.5 * learned_value
        )

    # =========================================================
    # Recent Experience
    # =========================================================

    def get_recent_experience(
        self,
        size=5,
    ):
        """
        جدیدترین تجربیات.
        """

        return self.memory.get_short_term(
            size=size
        )

    # =========================================================
    # Recency Weight
    # =========================================================

    def get_recency_weight(
        self,
        index,
    ):
        """
        وزن تجربه بر اساس تازگی.
        """

        experiences = self.memory.get_all()

        total = len(experiences)

        if total <= 0:
            return 0.0

        age = (
            total
            - 1
            - int(index)
        )

        return (
            self.recency_weight
            ** age
        )

    # =========================================================
    # Location Learning
    # =========================================================

    def update_location_value(
        self,
        position,
        reward,
    ):
        """
        یادگیری ارزش یک مکان.
        """

        if position is None:
            return 0.0

        try:
            position = tuple(
                int(x)
                for x in position
            )
        except (TypeError, ValueError):
            return 0.0

        try:
            reward = float(reward)
        except (TypeError, ValueError):
            reward = 0.0

        old_value = (
            self.location_values.get(
                position,
                0.0,
            )
        )

        new_value = (
            old_value
            + self.learning_rate
            * (reward - old_value)
        )

        self.location_values[position] = (
            new_value
        )

        return new_value

    def get_location_value(
        self,
        position,
    ):
        if position is None:
            return 0.0

        try:
            position = tuple(
                int(x)
                for x in position
            )
        except (TypeError, ValueError):
            return 0.0

        memory_value = (
            self.memory.get_location_value(
                position
            )
        )

        learned_value = (
            self.location_values.get(
                position,
                0.0,
            )
        )

        if position not in self.location_values:
            return memory_value

        return (
            0.5 * memory_value
            + 0.5 * learned_value
        )

    # =========================================================
    # Learning Signal
    # =========================================================

    def get_learning_signal(
        self,
        reward,
    ):
        """
        تبدیل Reward به سیگنال یادگیری.

        Reward مثبت:
            تقویت رفتار

        Reward منفی:
            اصلاح رفتار
        """

        try:
            reward = float(reward)
        except (TypeError, ValueError):
            reward = 0.0

        return reward * self.learning_rate

    # =========================================================
    # Failure Learning
    # =========================================================

    def get_failure_learning(
        self,
    ):
        """
        میزان یادگیری از شکست.

        شکست به‌عنوان اطلاعات منفی
        در نظر گرفته می‌شود، نه صرفاً فقدان موفقیت.
        """

        experiences = self.memory.get_all()

        if not experiences:
            return 0.0

        failures = []

        for experience in experiences:
            reward = float(
                experience.get(
                    "reward",
                    0.0,
                )
            )

            collected = float(
                experience.get(
                    "collected_resource",
                    0.0,
                )
            )

            if reward < 0.0 or (
                reward <= 0.0
                and collected <= 0.0
            ):
                failures.append(
                    experience
                )

        if not failures:
            return 0.0

        return len(failures) / len(
            experiences
        )

    # =========================================================
    # Generalization
    # =========================================================

    def get_generalized_action_value(
        self,
        action,
    ):
        """
        Generalization ساده:

        اگر Action تجربه مستقیم نداشته باشد،
        از میانگین دانش Actionها استفاده می‌شود.
        """

        direct = self.get_action_value(
            action
        )

        if action in self.action_values:
            return direct

        if not self.action_values:
            return direct

        values = list(
            self.action_values.values()
        )

        average = (
            sum(values)
            / len(values)
        )

        return (
            0.7 * direct
            + 0.3 * average
        )

    # =========================================================
    # Strategy Adaptation
    # =========================================================

    def update_strategy(
        self,
        action,
        reward,
    ):
        """
        Strategy بر اساس نتیجه Action تغییر می‌کند.
        """

        value = self.update_action_value(
            action,
            reward,
        )

        try:
            action = int(action)
        except (TypeError, ValueError):
            return value

        self.strategy_bias[action] = (
            value
        )

        return value

    def get_strategy_bias(self):
        return self.strategy_bias.copy()

    # =========================================================
    # Environment Adaptation
    # =========================================================

    def update_environment_model(self):
        """
        Agent به تدریج الگوی محیط را از تجربه
        استخراج می‌کند.
        """

        experiences = self.memory.get_all()

        if not experiences:
            return self.environment_model.copy()

        rewards = [
            float(
                experience.get(
                    "reward",
                    0.0,
                )
            )
            for experience in experiences
        ]

        average_reward = (
            sum(rewards)
            / len(rewards)
        )

        success_rate = (
            self.memory.get_success_rate(
                experiences
            )
        )

        failure_rate = (
            self.memory.get_failure_rate(
                experiences
            )
        )

        self.environment_model = {
            "average_reward": average_reward,
            "success_rate": success_rate,
            "failure_rate": failure_rate,
        }

        return self.environment_model.copy()

    # =========================================================
    # Learn From Experience
    # =========================================================

    def learn_from_experience(
        self,
        experience,
    ):
        """
        نقطه ورود اصلی سیستم Learning.

        Memory قبلاً تجربه را ذخیره کرده است.
        اینجا از آن تجربه دانش استخراج می‌شود.
        """

        if not isinstance(
            experience,
            dict,
        ):
            return None

        action = experience.get(
            "action"
        )

        reward = experience.get(
            "reward",
            0.0,
        )

        position = experience.get(
            "position"
        )

        self.update_action_value(
            action,
            reward,
        )

        self.update_location_value(
            position,
            reward,
        )

        self.update_strategy(
            action,
            reward,
        )

        self.update_environment_model()

        return self.get_state()

    # =========================================================
    # Full Learning State
    # =========================================================

    def get_state(self):
        """
        وضعیت کامل Learning.
        """

        experiences = (
            self.memory.get_all()
        )

        return {
            "reward_learning": (
                self.get_reward_learning()
            ),

            "action_values": (
                self.action_values.copy()
            ),

            "recent_experience": (
                self.get_recent_experience()
            ),

            "recency_weight": (
                self.recency_weight
            ),

            "location_values": (
                self.location_values.copy()
            ),

            "learning_signal": (
                self.get_reward_learning()
                * self.learning_rate
            ),

            "failure_learning": (
                self.get_failure_learning()
            ),

            "strategy_bias": (
                self.strategy_bias.copy()
            ),

            "environment_model": (
                self.environment_model.copy()
            ),

            "experience_count": len(
                experiences
            ),
        }

    # =========================================================
    # Episode
    # =========================================================

    def reset_episode(self):
        """
        Learning بین Episodeها حفظ می‌شود.

        چون یادگیری بخشی از شخصیت/دانش Agent است.
        """

        # هیچ دانش بلندمدتی پاک نمی‌شود.
        self.update_environment_model()
