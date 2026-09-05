class CompetitionStrategy:
    """
    تصمیم‌گیری استراتژیک بر اساس Competition و Risk.

    Strategyها:

        neutral
        approach
        race
        alternative
        avoid
        pursue
    """

    def __init__(
        self,
        race_competition_threshold=0.60,
        pursue_competition_threshold=0.70,
        alternative_competition_threshold=0.35,
        avoid_risk_threshold=0.80,
        pursue_risk_threshold=0.60,
    ):
        self.race_competition_threshold = float(
            race_competition_threshold
        )

        self.pursue_competition_threshold = float(
            pursue_competition_threshold
        )

        self.alternative_competition_threshold = float(
            alternative_competition_threshold
        )

        self.avoid_risk_threshold = float(
            avoid_risk_threshold
        )

        self.pursue_risk_threshold = float(
            pursue_risk_threshold
        )
        
        self.risk_tolerance = 0.5

    def choose(
        self,
        competition=0.0,
        risk=0.0,
        resource_value=0.0,
        competitor_distance=None,
        risk_tolerance=None,
    ):
        competition = self._clip(competition)
        risk = self._clip(risk)

    # اگر tolerance از بیرون داده نشده،
    # از مقدار ذخیره‌شده روی Strategy استفاده کن.
        if risk_tolerance is None:
            risk_tolerance = getattr(
                self,
                "risk_tolerance",
                0.5,
            )

        risk_tolerance = self._clip(risk_tolerance)

    # ---------------------------------------------------------
    # Relative Risk
    # ---------------------------------------------------------
    #
    # tolerance بالا => risk مؤثر کمتر
    # tolerance پایین => risk مؤثر بیشتر
    #
        effective_risk = (
            risk
            * (
                1.5
                - risk_tolerance
            )
        )

        effective_risk = self._clip(
            effective_risk
        )

        # Risk always first
        if effective_risk >= self.avoid_risk_threshold:
            return "avoid"

        if competition >= self.pursue_competition_threshold:
            if effective_risk <= self.pursue_risk_threshold:
                return "pursue"
            return "race"

        if competition >= self.race_competition_threshold:
            return "race"

        if competition >= self.alternative_competition_threshold:
            return "alternative"

        return "approach"
    
    def get_priority_modifier(
        self,
        strategy,
    ):
        """
        ضریب تأثیر Strategy روی Resource score.

        مقدار بزرگ‌تر:
            ترجیح بیشتر

        مقدار کوچک‌تر:
            ترجیح کمتر
        """

        # نکته مهم:
        # حتی pursue نیز به دلیل هزینه Competition
        # نباید امتیاز Resource را بالاتر از حالت
        # بدون Competition ببرد.
        modifiers = {
            "approach": 1.00,
            "race": 0.90,
            "pursue": 0.85,
            "alternative": 0.75,
            "avoid": 0.10,
            "neutral": 1.00,
        }

        return float(
            modifiers.get(
                strategy,
                1.00,
            )
        )

    def get_strategy_details(
        self,
        competition=0.0,
        risk=0.0,
        resource_value=0.0,
        competitor_distance=None,
    ):
        """
        اطلاعات کامل Strategy.

        برای جاهایی که علاوه بر نام Strategy،
        modifier و وضعیت Competition لازم است.
        """

        competition = self._clip(competition)
        risk = self._clip(risk)

        strategy = self.choose(
            competition=competition,
            risk=risk,
            resource_value=resource_value,
            competitor_distance=competitor_distance,
        )

        return {
            "strategy": strategy,
            "priority_modifier": self.get_priority_modifier(
                strategy
            ),
            "competition": competition,
            "risk": risk,
            "resource_value": float(
                resource_value or 0.0
            ),
            "competitor_distance": competitor_distance,
        }

    @staticmethod
    def _clip(value):
        try:
            value = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        return max(
            0.0,
            min(
                1.0,
                value,
            )
        )
