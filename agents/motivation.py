class MotivationModel:
    """
    مدل انگیزش Agent.

    Motivation:
    - Lack
    - Desire
    - Satisfaction
    - Urgency
    """

    def __init__(
        self,
        lack_weight=1.0,
        satisfaction_weight=0.5,
        urgency_lack_weight=0.5,
        urgency_desire_weight=0.5,
    ):
        self.lack_weight = float(
            lack_weight
        )

        self.satisfaction_weight = float(
            satisfaction_weight
        )

        self.urgency_lack_weight = float(
            urgency_lack_weight
        )

        self.urgency_desire_weight = float(
            urgency_desire_weight
        )

    @staticmethod
    def _clip(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0

        return max(
            0.0,
            min(1.0, value),
        )

    # =========================================================
    # Desire
    # =========================================================

    def calculate_desire(
        self,
        lack,
        satisfaction,
    ):
        lack = self._clip(lack)
        satisfaction = self._clip(
            satisfaction
        )

        desire = (
            self.lack_weight * lack
            -
            self.satisfaction_weight
            * satisfaction
        )

        return self._clip(desire)

    # =========================================================
    # Urgency
    # =========================================================

    def calculate_urgency(
        self,
        lack,
        desire,
    ):
        lack = self._clip(lack)
        desire = self._clip(desire)

        urgency = (
            self.urgency_lack_weight
            * lack
            +
            self.urgency_desire_weight
            * desire
        )

        return self._clip(urgency)

    # =========================================================
    # State
    # =========================================================

    def get_state(
        self,
        lack,
        satisfaction,
    ):
        lack = self._clip(lack)
        satisfaction = self._clip(
            satisfaction
        )

        desire = self.calculate_desire(
            lack,
            satisfaction,
        )

        urgency = self.calculate_urgency(
            lack,
            desire,
        )

        return {
            "lack": lack,
            "desire": desire,
            "satisfaction": satisfaction,
            "urgency": urgency,
        }
