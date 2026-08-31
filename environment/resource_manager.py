
import numpy as np


class ResourceManager:
    """
    مدیریت پویا‌ی Resourceهای جهان.

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
        
        self.grid_size = grid_size

        self.initial_resources = initial_resources
        self.max_resources = max_resources

        self.spawn_probability = float(
            spawn_probability
        )

        self.min_value = min_value
        self.max_value = max_value

        self.resource_lifetime = int(
            resource_lifetime
        )

        self.resources = {}
        self.resource_birth_steps = {}

        # =====================================================
        # Metrics
        # =====================================================

        self.metrics = {}

        self._reset_metrics()

        self.value_distribution = (
            value_distribution
            if value_distribution is not None
            else {
                5: 0.60,
                15: 0.30,
                50: 0.10,
            }
        )
        
    # =========================================================
    # Metrics
    # =========================================================

    def _reset_metrics(self):
        """
        Reset تمام Metrics مربوط به Resourceها.
        """

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
        دریافت Summary Metrics مربوط به Resourceها.
        """

        expired_lifetimes = (
            self.metrics["expired_lifetimes"]
        )

        spawned_values = (
            self.metrics["spawned_values"]
        )

        collected_values = (
            self.metrics["collected_values"]
        )

        return {
            "spawned": int(
                self.metrics["spawned"]
            ),

            "collected": int(
                self.metrics["collected"]
            ),

            "expired": int(
                self.metrics["expired"]
            ),

            "mean_lifetime": (
                float(
                    np.mean(
                        expired_lifetimes
                    )
                )
                if expired_lifetimes
                else 0.0
            ),

            "mean_value": (
                float(
                    np.mean(
                        spawned_values
                    )
                )
                if spawned_values
                else 0.0
            ),

            "mean_collected_value": (
                float(
                    np.mean(
                        collected_values
                    )
                )
                if collected_values
                else 0.0
            ),
        }

    # =========================================================
    # Reset
    # =========================================================

    def reset(
        self,
        occupied_positions,
        step=0,
    ):
        """
        ایجاد منابع اولیه و Reset کامل وضعیت ResourceManager.
        """

        self.resources = {}
        self.resource_birth_steps = {}

        self._reset_metrics()

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

            self.resource_birth_steps[
                position
            ] = step

            # ثبت Spawn
            self.metrics[
                "spawned"
            ] += 1

            self.metrics[
                "spawned_values"
            ].append(
                value
            )

    # =========================================================
    # Position
    # =========================================================

    def _random_position(
        self,
        occupied_positions,
    ):
        """
        پیدا کردن یک موقعیت آزاد تصادفی.
        """

        occupied = set(
            tuple(position)
            for position in occupied_positions
        )

        occupied.update(
            self.resources.keys()
        )

        available = []

        for x in range(
            self.grid_size
        ):

            for y in range(
                self.grid_size
            ):

                position = (x, y)

                if position not in occupied:
                    available.append(
                        position
                    )

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
        """
        جمع‌آوری Resource.

        Resource از جهان حذف می‌شود و Metrics
        مربوط به Collection به‌روزرسانی می‌شود.
        """

        position = tuple(position)

        if position not in self.resources:
            return None

        value = self.resources.pop(
            position
        )

        self.resource_birth_steps.pop(
            position,
            None,
        )

        value = float(value)

        self.metrics[
            "collected"
        ] += 1

        self.metrics[
            "collected_values"
        ].append(
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
        """
        به‌روزرسانی Resourceها در هر Step.

        شامل:
        - حذف Resourceهای قدیمی
        - تولید Resourceهای جدید
        """

        self._remove_expired_resources(
            step
        )

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
        """
        حذف Resourceهایی که بیش از Lifetime
        در جهان باقی مانده‌اند.
        """

        expired = []

        for (
            position,
            birth_step,
        ) in (
            self.resource_birth_steps.items()
        ):

            age = (
                step - birth_step
            )

            if (
                age
                >= self.resource_lifetime
            ):
                expired.append(
                    (
                        position,
                        age,
                    )
                )

        for (
            position,
            lifetime,
        ) in expired:

            value = self.resources.pop(
                position,
                None,
            )

            self.resource_birth_steps.pop(
                position,
                None,
            )

            self.metrics[
                "expired"
            ] += 1

            self.metrics[
                "expired_lifetimes"
            ].append(
                lifetime
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
        """
        تولید Resource جدید با احتمال مشخص.
        """

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

        self.resource_birth_steps[
            position
        ] = step

        self.metrics[
            "spawned"
        ] += 1

        self.metrics[
            "spawned_values"
        ].append(
            value
        )

    # =========================================================
    # Snapshot
    # =========================================================

    def get_resources(self):
        """
        دریافت Snapshot منابع.
        """

        return self.resources.copy()

    # =========================================================
    # Count
    # =========================================================

    def count(self):
        """
        تعداد فعلی Resourceها.
        """

        return len(
            self.resources
        )
