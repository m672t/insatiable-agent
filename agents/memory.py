class MemorySystem:
    """
    Unified memory system for agents.

    Stores one experience stream and provides:
        - short-term memory
        - long-term memory
        - recency weighting
        - success/failure learning
        - location -> outcome association
        - action -> outcome association
        - forgetting
    """

    def __init__(
        self,
        max_size=1000,
        short_term_size=50,
        recency_decay=0.995,
    ):
        self.max_size = max(
            1,
            int(max_size),
        )

        self.short_term_size = max(
            1,
            int(short_term_size),
        )

        self.recency_decay = float(
            recency_decay
        )

        self.experiences = []

    # =========================================================
    # Store
    # =========================================================

    def record(
        self,
        action,
        reward,
        info=None,
        episode=None,
        step=None,
    ):
        if not isinstance(info, dict):
            info = {}

        try:
            action = int(action)
        except (TypeError, ValueError):
            action = 4

        try:
            reward = float(reward)
        except (TypeError, ValueError):
            reward = 0.0

        position = info.get("position")

        if position is not None:
            try:
                position = tuple(
                    int(x)
                    for x in position
                )
            except (TypeError, ValueError):
                position = None

        resource_location = info.get(
            "resource_location"
        )

        if resource_location is not None:
            try:
                resource_location = tuple(
                    int(x)
                    for x in resource_location
                )
            except (TypeError, ValueError):
                resource_location = None

        try:
            resource_value = float(
                info.get(
                    "resource_value",
                    0.0,
                ) or 0.0
            )
        except (TypeError, ValueError):
            resource_value = 0.0

        try:
            collected_resource = float(
                info.get(
                    "collected_resource",
                    0.0,
                ) or 0.0
            )
        except (TypeError, ValueError):
            collected_resource = 0.0

        try:
            competition = float(
                info.get(
                    "competition",
                    0.0,
                ) or 0.0
            )
        except (TypeError, ValueError):
            competition = 0.0

        outcome = str(
            info.get(
                "outcome",
                "unknown",
            )
        )

        experience = {
            "action": action,
            "reward": reward,
            "position": position,
            "resource_location": resource_location,
            "resource_value": resource_value,
            "collected_resource": max(
                0.0,
                collected_resource,
            ),
            "competition": max(
                0.0,
                competition,
            ),
            "outcome": outcome,
            "episode": episode,
            "step": step,
            "info": info.copy(),
        }

        self.experiences.append(
            experience
        )

        self._forget_old_experiences()

        return experience.copy()

    # =========================================================
    # Forgetting
    # =========================================================

    def _forget_old_experiences(self):
        overflow = (
            len(self.experiences)
            - self.max_size
        )

        if overflow > 0:
            del self.experiences[
                :overflow
            ]

    # =========================================================
    # Access
    # =========================================================

    def get_all(self):
        return [
            self._copy_experience(exp)
            for exp in self.experiences
        ]

    def get_short_term(
        self,
        size=None,
    ):
        if size is None:
            size = self.short_term_size

        size = max(
            1,
            int(size),
        )

        return [
            self._copy_experience(exp)
            for exp in self.experiences[-size:]
        ]

    def get_long_term(self):
        cutoff = max(
            0,
            len(self.experiences)
            - self.short_term_size,
        )

        return [
            self._copy_experience(exp)
            for exp in self.experiences[:cutoff]
        ]

    def __len__(self):
        return len(self.experiences)

    # =========================================================
    # Recency
    # =========================================================

    def get_recency_weight(self, index):
        total = len(self.experiences)

        if total <= 0:
            return 0.0

        age = (
            total
            - 1
            - int(index)
        )

        return (
            self.recency_decay ** age
        )

    def weighted_average_reward(
        self,
        experiences=None,
    ):
        if experiences is None:
            experiences = self.experiences

        if not experiences:
            return 0.0

        weighted_sum = 0.0
        weight_sum = 0.0

        total = len(experiences)

        for index, experience in enumerate(
            experiences
        ):
            try:
                reward = float(
                    experience.get(
                        "reward",
                        0.0,
                    )
                )
            except (TypeError, ValueError):
                reward = 0.0

            age = (
                total
                - 1
                - index
            )

            weight = (
                self.recency_decay ** age
            )

            weighted_sum += (
                reward * weight
            )

            weight_sum += weight

        if weight_sum <= 0:
            return 0.0

        return (
            weighted_sum
            / weight_sum
        )

    # =========================================================
    # Action -> Outcome
    # =========================================================

    def get_action_value(self, action):
        matches = [
            exp
            for exp in self.experiences
            if exp["action"] == int(action)
        ]

        if not matches:
            return 0.0

        return self._weighted_average(
            matches
        )

    # =========================================================
    # Location -> Outcome
    # =========================================================

    def get_location_value(self, position):
        try:
            position = tuple(
                int(x)
                for x in position
            )
        except (TypeError, ValueError):
            return 0.0

        matches = []

        for exp in self.experiences:
            if (
                exp.get("position")
                == position
                or
                exp.get("resource_location")
                == position
            ):
                matches.append(exp)

        if not matches:
            return 0.0

        return self._weighted_average(
            matches
        )

    # =========================================================
    # Success / Failure
    # =========================================================

    def get_success_rate(
        self,
        experiences=None,
    ):
        if experiences is None:
            experiences = self.experiences

        if not experiences:
            return 0.0

        successes = sum(
            1
            for exp in experiences
            if (
                exp.get(
                    "collected_resource",
                    0.0,
                ) > 0.0
                or
                exp.get(
                    "reward",
                    0.0,
                ) > 0.0
            )
        )

        return successes / len(
            experiences
        )

    def get_failure_rate(
        self,
        experiences=None,
    ):
        return 1.0 - self.get_success_rate(
            experiences
        )

    # =========================================================
    # Helpers
    # =========================================================

    def _weighted_average(
        self,
        experiences,
    ):
        if not experiences:
            return 0.0

        weighted_sum = 0.0
        weight_sum = 0.0

        total = len(experiences)

        for index, exp in enumerate(
            experiences
        ):
            reward = float(
                exp.get(
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
                self.recency_decay ** age
            )

            weighted_sum += (
                reward * weight
            )

            weight_sum += weight

        if weight_sum <= 0:
            return 0.0

        return (
            weighted_sum
            / weight_sum
        )

    @staticmethod
    def _copy_experience(
        experience
    ):
        copied = experience.copy()

        if isinstance(
            copied.get("info"),
            dict,
        ):
            copied["info"] = (
                copied["info"].copy()
            )

        return copied
    
    def get_memory(self):
        """
        سازگاری با API قبلی InternalState.

        تمام تجربیات ذخیره‌شده را برمی‌گرداند.
        """
        return self.get_all()
    
    def append(self, experience):
        """
        Compatibility API.

        Allows tests and legacy callers to append an already
        constructed experience directly.
        """
        if not isinstance(experience, dict):
            return

        self.experiences.append(
            self._normalize_experience(experience)
        )

        self._forget_old_experiences()


    def clear(self):
        """
        Clear all stored experiences.
        """
        self.experiences.clear()
        
    def _normalize_experience(self, experience):
        exp = experience.copy()

        if "action" in exp:
            try:
                exp["action"] = int(exp["action"])
            except (TypeError, ValueError):
                exp["action"] = 4

        try:
            exp["reward"] = float(
                exp.get("reward", 0.0)
            )
        except (TypeError, ValueError):
            exp["reward"] = 0.0

        if "collected_resource" in exp:
            try:
                exp["collected_resource"] = max(
                    0.0,
                    float(
                        exp.get(
                            "collected_resource",
                            0.0,
                        )
                    ),
                )
            except (TypeError, ValueError):
                exp["collected_resource"] = 0.0

        if "resource_value" in exp:
            try:
                exp["resource_value"] = max(
                    0.0,
                    float(
                        exp.get(
                            "resource_value",
                            0.0,
                        )
                    ),
                )
            except (TypeError, ValueError):
                exp["resource_value"] = 0.0

        return exp
    
    def clear(self):
        self.experiences.clear()


    def append(self, experience):
        if not isinstance(experience, dict):
            raise TypeError(
                "MemorySystem only accepts dictionary experiences"
            )

        self.experiences.append(
            experience.copy()
        )

        self._forget_old_experiences()


    def __iter__(self):
        return iter(self.experiences)


    def __getitem__(self, index):
        return self.experiences[index]
