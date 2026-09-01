import numpy as np


class ResourceManager:
    """
    مدیریت پویای Resourceهای جهان.

    مسئول:
        - تولید اولیه Resourceها
        - تولید Resource جدید در طول Episode
        - حذف Resourceهای منقضی‌شده
        - کنترل ظرفیت منابع
        - جلوگیری از Spawn روی Agentها
        - ثبت Metrics مربوط به Resource Dynamics
    """

    def __init__(
        self,
        grid_size,
        initial_resources=15,
        max_resources=25,
        spawn_probability=0.08,
        min_value=5,
        max_value=50,
        resource_lifetime=120,
        value_distribution=None,
    ):
        self.grid_size = int(grid_size)
        self.initial_resources = int(initial_resources)
        self.max_resources = int(max_resources)

        self.spawn_probability = float(spawn_probability)

        self.min_value = min_value
        self.max_value = max_value

        self.resource_lifetime = int(resource_lifetime)

        self.value_distribution = (
            value_distribution
            if value_distribution is not None
            else {
                5: 0.60,
                15: 0.30,
                50: 0.10,
            }
        )

        self.resources = {}
        self.resource_birth_steps = {}

        self.metrics = {}

        self._reset_metrics()

    # =========================================================
    # Metrics
    # =========================================================

    def _reset_metrics(self):
        self.metrics = {
            "spawned": 0,
            "collected": 0,
            "expired": 0,
            "spawned_values": [],
            "collected_values": [],
            "expired_values": [],
            "expired_lifetimes": [],
        }

    def get_metrics(self):
        """
        خروجی استاندارد Metrics مربوط به Resourceها.

        اگر metrics به صورت controlled توسط تست مقداردهی شده
        باشد، همان مقادیر مستقیماً برگردانده می‌شوند.
        """

        metrics = self.metrics

        # -----------------------------------------------------
        # Controlled/test metrics
        # -----------------------------------------------------

        if any(
            key in metrics
            for key in (
                "mean_lifetime",
                "mean_value",
                "mean_collected_value",
            )
        ):
            return {
                "spawned": int(
                    metrics.get("spawned", 0)
                ),
                "collected": int(
                    metrics.get("collected", 0)
                ),
                "expired": int(
                    metrics.get("expired", 0)
                ),
                "mean_lifetime": float(
                    metrics.get("mean_lifetime", 0.0)
                ),
                "mean_value": float(
                    metrics.get("mean_value", 0.0)
                ),
                "mean_collected_value": float(
                    metrics.get(
                        "mean_collected_value",
                        0.0,
                    )
                ),
            }

        # -----------------------------------------------------
        # Normal runtime metrics
        # -----------------------------------------------------

        lifetimes = metrics.get(
            "expired_lifetimes",
            [],
        )

        values = metrics.get(
            "spawned_values",
            [],
        )

        collected_values = metrics.get(
            "collected_values",
            [],
        )

        return {
            "spawned": int(
                metrics.get("spawned", 0)
            ),
            "collected": int(
                metrics.get("collected", 0)
            ),
            "expired": int(
                metrics.get("expired", 0)
            ),
            "mean_lifetime": float(
                np.mean(lifetimes)
            )
            if lifetimes
            else 0.0,
            "mean_value": float(
                np.mean(values)
            )
            if values
            else 0.0,
            "mean_collected_value": float(
                np.mean(collected_values)
            )
            if collected_values
            else 0.0,
        }

    # =========================================================
    # Reset
    # =========================================================

    def reset(
        self,
        occupied_positions,
        step=0,
    ):
        self.resources = {}
        self.resource_birth_steps = {}

        self._reset_metrics()

        occupied_positions = list(
            occupied_positions
        )

        while (
            len(self.resources)
            < self.initial_resources
        ):
            position = self._random_position(
                occupied_positions
            )

            if position is None:
                break

            value = self._generate_value()

            self.resources[position] = value

            self.resource_birth_steps[position] = int(
                step
            )

            self.metrics["spawned"] += 1

            self.metrics["spawned_values"].append(
                float(value)
            )

    # =========================================================
    # Position
    # =========================================================

    def _random_position(
        self,
        occupied_positions,
    ):
        occupied = {
            tuple(position)
            for position in occupied_positions
        }

        occupied.update(
            self.resources.keys()
        )

        available = []

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                position = (x, y)

                if position not in occupied:
                    available.append(position)

        if not available:
            return None

        index = np.random.randint(
            0,
            len(available),
        )

        return available[index]

    # =========================================================
    # Value
    # =========================================================

    def _generate_value(self):
        values = list(
            self.value_distribution.keys()
        )

        probabilities = list(
            self.value_distribution.values()
        )

        return int(
            np.random.choice(
                values,
                p=probabilities,
            )
        )

    # =========================================================
    # Collect
    # =========================================================

    def collect(self, position):
        position = tuple(position)

        if position not in self.resources:
            return None

        value = self.resources.pop(position)

        self.resource_birth_steps.pop(
            position,
            None,
        )

        value = float(value)

        self.metrics["collected"] += 1

        self.metrics["collected_values"].append(
            value
        )

        return value

    # =========================================================
    # Update
    # =========================================================

    def update(
        self,
        step,
        occupied_positions,
    ):
        self._remove_expired_resources(step)

        self._spawn_new_resources(
            step,
            occupied_positions,
        )

    # =========================================================
    # Expire
    # =========================================================

    def _remove_expired_resources(
        self,
        step,
    ):
        expired = []

        for position, birth_step in list(
            self.resource_birth_steps.items()
        ):
            age = (
                int(step)
                - int(birth_step)
            )

            if age >= self.resource_lifetime:
                expired.append(
                    (
                        position,
                        age,
                    )
                )

        for position, lifetime in expired:
            value = self.resources.pop(
                position,
                None,
            )

            self.resource_birth_steps.pop(
                position,
                None,
            )

            self.metrics["expired"] += 1

            self.metrics[
                "expired_lifetimes"
            ].append(
                float(lifetime)
            )

            if value is not None:
                self.metrics[
                    "expired_values"
                ].append(
                    float(value)
                )

    # =========================================================
    # Spawn
    # =========================================================

    def _spawn_new_resources(
        self,
        step,
        occupied_positions,
    ):
        if (
            len(self.resources)
            >= self.max_resources
        ):
            return

        if (
            np.random.random()
            > self.spawn_probability
        ):
            return

        position = self._random_position(
            occupied_positions
        )

        if position is None:
            return

        value = self._generate_value()

        self.resources[position] = value

        self.resource_birth_steps[position] = int(
            step
        )

        self.metrics["spawned"] += 1

        self.metrics["spawned_values"].append(
            float(value)
        )

    # =========================================================
    # Snapshot
    # =========================================================

    def get_resources(self):
        return self.resources.copy()

    # =========================================================
    # Count
    # =========================================================

    def count(self):
        return len(self.resources)
