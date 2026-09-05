from agents.memory import MemorySystem


class InternalState:
    """
    Dynamic internal state.

    Memory is injected from BaseAgent and is shared
    with the unified MemorySystem.
    """

    def __init__(
        self,
        lack_decay=0.02,
        desire_decay=0.01,
        satisfaction_decay=0.05,
        max_memory=1000,
        memory=None,
    ):
        self.lack_decay = float(
            lack_decay
        )

        self.desire_decay = float(
            desire_decay
        )

        self.satisfaction_decay = float(
            satisfaction_decay
        )

        # Unified Memory
        self.memory = (
            memory
            if memory is not None
            else MemorySystem(
                max_memory=max_memory
            )
        )

        self.reset()

    # =========================================================
    # Memory Injection
    # =========================================================

    def set_memory(self, memory):
        if memory is None:
            raise ValueError(
                "memory cannot be None"
            )

        self.memory = memory

    # =========================================================
    # Episode State
    # =========================================================

    def reset(self):
        """
        فقط وضعیت کوتاه‌مدت reset می‌شود.

        Memory حفظ می‌شود.
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

        collected_resource = max(
            0.0,
            collected_resource,
        )

        self.lack += self.lack_decay

        if collected_resource > 0.0:
            self.lack -= (
                collected_resource / 100.0
            )

        self.lack = max(
            0.0,
            min(1.0, self.lack),
        )

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
    # Memory Access
    # =========================================================

    def get_memory(self):
        return self.memory.get_memory()

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
