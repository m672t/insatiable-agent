class InternalState:
    """
    وضعیت داخلی پویا برای Agent.

    شامل:
    - lack
    - desire
    - satisfaction
    - memory

    Memory بین Episodeها حفظ می‌شود.
    """

    def __init__(
        self,
        lack_decay=0.02,
        desire_decay=0.01,
        satisfaction_decay=0.05,
        max_memory=1000,
    ):
        self.lack_decay = float(lack_decay)
        self.desire_decay = float(desire_decay)
        self.satisfaction_decay = float(
            satisfaction_decay
        )

        self.max_memory = max(
            1,
            int(max_memory),
        )

        self.memory = []

        self.reset()

    # =========================================================
    # Episode State
    # =========================================================

    def reset(self):
        """
        فقط وضعیت کوتاه‌مدت را Reset می‌کند.

        Memory پاک نمی‌شود.
        """

        self.lack = 0.0
        self.desire = 0.0
        self.satisfaction = 0.0

    # =========================================================
    # Internal Dynamics
    # =========================================================

    def update(
        self,
        reward=0.0,
        collected_resource=0.0,
    ):
        try:
            reward = float(reward)
        except (TypeError, ValueError):
            reward = 0.0

        try:
            collected_resource = float(
                collected_resource
            )
        except (TypeError, ValueError):
            collected_resource = 0.0

        reward = max(0.0, reward)
        collected_resource = max(
            0.0,
            collected_resource,
        )

        # -----------------------------------------------------
        # Lack
        # -----------------------------------------------------

        self.lack += self.lack_decay

        if collected_resource > 0.0:
            self.lack -= (
                collected_resource / 100.0
            )

        self.lack = max(
            0.0,
            min(1.0, self.lack),
        )

        # -----------------------------------------------------
        # Satisfaction
        # -----------------------------------------------------

        self.satisfaction -= (
            self.satisfaction_decay
        )

        if collected_resource > 0.0:
            self.satisfaction += (
                collected_resource / 100.0
            )

        self.satisfaction = max(
            0.0,
            min(1.0, self.satisfaction),
        )

        # -----------------------------------------------------
        # Desire
        # -----------------------------------------------------

        self.desire -= self.desire_decay

        self.desire += (
            self.lack * 0.08
        )

        self.desire -= (
            self.satisfaction * 0.04
        )

        self.desire = max(
            0.0,
            min(1.0, self.desire),
        )

    # =========================================================
    # Memory
    # =========================================================

    def record_experience(
        self,
        action,
        reward,
        info=None,
    ):
        """ثبت یک تجربه بلندمدت."""

        try:
            action = int(action)
        except (TypeError, ValueError):
            action = 4

        try:
            reward = float(reward)
        except (TypeError, ValueError):
            reward = 0.0

        raw_info = (
            info.copy()
            if isinstance(info, dict)
            else {}
        )

        # -----------------------------------------------------
        # Position
        # -----------------------------------------------------

        position = raw_info.get("position")

        if position is not None:
            try:
                position = tuple(
                    int(x)
                    for x in position
                )
            except (
                TypeError,
                ValueError,
            ):
                position = None

        # -----------------------------------------------------
        # Collected Resource
        # -----------------------------------------------------

        collected_resource = raw_info.get(
            "collected_resource",
            0.0,
        )

        try:
            collected_resource = float(
                collected_resource
            )
        except (
            TypeError,
            ValueError,
        ):
            collected_resource = 0.0

        collected_resource = max(
            0.0,
            collected_resource,
        )

        # -----------------------------------------------------
        # Experience
        # -----------------------------------------------------

        experience = {
            "action": action,
            "reward": reward,
            "position": position,
            "collected_resource": collected_resource,
            "info": raw_info,
        }

        self.memory.append(experience)

        # -----------------------------------------------------
        # Limit
        # -----------------------------------------------------

        if len(self.memory) > self.max_memory:
            self.memory = self.memory[
                -self.max_memory:
            ]

    # =========================================================
    # Memory Queries
    # =========================================================

    def get_memory(self):
        """دریافت Copy از Memory."""

        result = []

        for experience in self.memory:
            copied = experience.copy()

            if isinstance(
                copied.get("info"),
                dict,
            ):
                copied["info"] = (
                    copied["info"].copy()
                )

            result.append(copied)

        return result

    def clear_memory(self):
        """پاک کردن کامل Memory."""

        self.memory = []

    # =========================================================
    # State
    # =========================================================

    def get_state(self):
        return {
            "lack": float(self.lack),
            "desire": float(self.desire),
            "satisfaction": float(
                self.satisfaction
            ),
            "memory": self.get_memory(),
        }
