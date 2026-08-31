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
    ):
        self.grid_size = grid_size

        self.initial_resources = initial_resources
        self.max_resources = max_resources

        self.spawn_probability = spawn_probability

        self.min_value = min_value
        self.max_value = max_value

        self.resource_lifetime = resource_lifetime

        self.resources = {}
        self.resource_birth_steps = {}

    def reset(self, occupied_positions, step=0):
        """
        ایجاد منابع اولیه.
        """

        self.resources = {}
        self.resource_birth_steps = {}

        while len(self.resources) < self.initial_resources:

            position = self._random_position(
                occupied_positions
            )

            if position is None:
                break

            value = self._generate_value()

            self.resources[position] = value
            self.resource_birth_steps[position] = step

    def _random_position(self, occupied_positions):
        """
        پیدا کردن یک موقعیت آزاد تصادفی.
        """

        occupied = set(
            tuple(position)
            for position in occupied_positions
        )

        occupied.update(self.resources.keys())

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

    def _generate_value(self):
        """
        تولید Value برای Resource جدید.
        """

        return int(
            np.random.choice(
                [5, 15, 50],
                p=[0.6, 0.3, 0.1],
            )
        )

    def collect(self, position):
        """
        جمع‌آوری Resource.

        Resource از جهان حذف می‌شود و اطلاعات
        آن برای تولید مجدد در اختیار Environment قرار می‌گیرد.
        """

        position = tuple(position)

        if position not in self.resources:
            return None

        value = self.resources.pop(position)

        self.resource_birth_steps.pop(
            position,
            None,
        )

        return value

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

    def _remove_expired_resources(self, step):
        """
        حذف Resourceهایی که بیش از lifetime
        در جهان باقی مانده‌اند.
        """

        expired = []

        for position, birth_step in (
            self.resource_birth_steps.items()
        ):

            age = step - birth_step

            if age >= self.resource_lifetime:
                expired.append(position)

        for position in expired:

            self.resources.pop(
                position,
                None,
            )

            self.resource_birth_steps.pop(
                position,
                None,
            )

    def _spawn_new_resources(
        self,
        step,
        occupied_positions,
    ):
        """
        تولید Resource جدید با احتمال مشخص.
        """

        if len(self.resources) >= self.max_resources:
            return

        if np.random.random() > self.spawn_probability:
            return

        position = self._random_position(
            occupied_positions
        )

        if position is None:
            return

        value = self._generate_value()

        self.resources[position] = value

        self.resource_birth_steps[position] = step

    def get_resources(self):
        """
        دریافت Snapshot منابع.
        """

        return self.resources.copy()

    def count(self):
        """
        تعداد فعلی Resourceها.
        """

        return len(self.resources)
