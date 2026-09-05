class CompetitionSystem:
    """
    Competitive behavior system.

    فعلاً رفتارهای زیر را پشتیبانی می‌کند:
        - تشخیص رقبا
        - فاصله رقیب تا Resource
        - محاسبه Competition
        - تخمین Risk
        - انتخاب استراتژی رقابتی پایه

    رفتارهای تهاجمی، دفاعی، تعقیب و سرقت عمداً
    در این مرحله فعال نشده‌اند.
    """

    def __init__(
        self,
        competition_weight=1.0,
        risk_threshold=0.5,
        pursue_threshold=0.7,
        avoid_threshold=0.8,
    ):
        self.competition_weight = float(
            competition_weight
        )

        self.risk_threshold = float(
            risk_threshold
        )

        self.pursue_threshold = float(
            pursue_threshold
        )

        self.avoid_threshold = float(
            avoid_threshold
        )

        self.last_state = {
            "agents": [],
            "competition": 0.0,
            "risk": 0.0,
            "strategy": "neutral",
        }

    # =========================================================
    # Public API
    # =========================================================

    def evaluate(
        self,
        resource_value=0.0,
        resource_location=None,
        agent_position=None,
        other_agents=None,
    ):
        resource_value = self._safe_float(
            resource_value
        )

        agent_position = self._normalize_position(
            agent_position
        )

        resource_location = self._normalize_position(
            resource_location
        )

        competitors = self.detect_agents(
            other_agents
        )

        competitor_data = []

        for competitor in competitors:
            position = self._extract_position(
                competitor
            )

            distance_to_resource = (
                self.distance(
                    position,
                    resource_location,
                )
            )

            distance_to_agent = (
                self.distance(
                    position,
                    agent_position,
                )
            )

            competitor_data.append(
                {
                    "agent": competitor,
                    "position": position,
                    "distance_to_resource": (
                        distance_to_resource
                    ),
                    "distance_to_agent": (
                        distance_to_agent
                    ),
                }
            )

        competition = self.compute_competition(
            resource_value=resource_value,
            competitors=competitor_data,
        )

        risk = self.compute_risk(
            competition=competition,
            resource_value=resource_value,
            competitors=competitor_data,
        )

        strategy = self.choose_strategy(
            competition=competition,
            risk=risk,
            resource_value=resource_value,
            competitors=competitor_data,
        )

        self.last_state = {
            "agents": competitor_data,
            "competition": competition,
            "risk": risk,
            "strategy": strategy,
            "resource_value": resource_value,
            "resource_location": resource_location,
        }

        return self.get_state()

    # =========================================================
    # Agent Detection
    # =========================================================

    def detect_agents(self, other_agents):
        """
        تشخیص Agentهای دیگر.

        ورودی می‌تواند:
            - list
            - tuple
            - None

        باشد.
        """

        if other_agents is None:
            return []

        if not isinstance(
            other_agents,
            (list, tuple),
        ):
            other_agents = [other_agents]

        result = []

        for agent in other_agents:
            if agent is None:
                continue

            result.append(agent)

        return result

    # =========================================================
    # Distance
    # =========================================================

    @staticmethod
    def distance(
        position_a,
        position_b,
    ):
        if (
            position_a is None
            or position_b is None
        ):
            return None

        try:
            return sum(
                abs(
                    float(a) - float(b)
                )
                for a, b in zip(
                    position_a,
                    position_b,
                )
            )
        except (TypeError, ValueError):
            return None

    # =========================================================
    # Competition
    # =========================================================

    def compute_competition(
        self,
        resource_value,
        competitors,
    ):
        if not competitors:
            return 0.0

        best_score = 0.0

        for competitor in competitors:
            distance = competitor.get(
                "distance_to_resource"
            )

            if distance is None:
                continue

            proximity = 1.0 / (
                1.0 + max(
                    0.0,
                    float(distance),
                )
            )

            value_factor = min(
                1.0,
                max(
                    0.0,
                    resource_value
                    / (
                        resource_value
                        + 10.0
                    )
                    if resource_value > 0
                    else 0.0,
                ),
            )

            score = (
                proximity
                * value_factor
                * self.competition_weight
            )

            best_score = max(
                best_score,
                score,
            )

        return min(
            1.0,
            max(
                0.0,
                best_score,
            ),
        )

    # =========================================================
    # Risk
    # =========================================================

    def compute_risk(
        self,
        competition,
        resource_value,
        competitors,
    ):
        """
        Competition به تنهایی Risk نیست.

        Risk وقتی بالا می‌رود که:
            - Competition بالا باشد
            - Resource ارزشمند باشد
            - رقیب نزدیک Resource باشد
        """

        if not competitors:
            return 0.0

        value_factor = min(
            1.0,
            max(
                0.0,
                resource_value / 100.0,
            ),
        )

        risk = (
            competition
            * (
                0.5
                + 0.5 * value_factor
            )
        )

        return min(
            1.0,
            max(
                0.0,
                risk,
            ),
        )

    # =========================================================
    # Strategy
    # =========================================================

    def choose_strategy(
        self,
        competition,
        risk,
        resource_value,
        competitors,
    ):
        if not competitors:
            return "neutral"

        if (
            competition
            >= self.pursue_threshold
            and risk
            < self.risk_threshold
        ):
            return "pursue"

        if risk >= self.avoid_threshold:
            return "avoid"

        if competition >= 0.4:
            return "alternative"

        return "neutral"

    # =========================================================
    # State
    # =========================================================

    def get_state(self):
        return {
            key: value
            for key, value in self.last_state.items()
        }

    # =========================================================
    # Helpers
    # =========================================================

    @staticmethod
    def _safe_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _normalize_position(position):
        if position is None:
            return None

        try:
            return tuple(
                int(x)
                for x in position
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_position(agent):
        if isinstance(agent, dict):
            return CompetitionSystem._normalize_position(
                agent.get("position")
            )

        if hasattr(agent, "position"):
            return CompetitionSystem._normalize_position(
                agent.position
            )

        if hasattr(agent, "get_position"):
            try:
                return CompetitionSystem._normalize_position(
                    agent.get_position()
                )
            except Exception:
                return None

        return None
    
    def evaluate_resource(self, agent_position, resource, competitors):
        """
        ارزیابی یک Resource با توجه به رقابت.

        خروجی شامل:
            - distance
            - competition
            - risk
            - priority
            - strategy
        """

        resource_position = resource.get("position")

        if resource_position is None:
            return {
                "distance": float("inf"),
                "competition": 0.0,
                "risk": 0.0,
                "priority": 0.0,
                "strategy": "ignore",
            }

        distance = self._distance(
            agent_position,
            resource_position,
        )

        competition = 0.0
        nearest_competitor_distance = float("inf")

        for competitor in competitors or []:
            competitor_position = competitor.get("position")

            if competitor_position is None:
                continue

            competitor_distance = self._distance(
                competitor_position,
                resource_position,
            )

            nearest_competitor_distance = min(
                nearest_competitor_distance,
                competitor_distance,
            )

        # هرچه رقیب به Resource نزدیک‌تر باشد،
        # Competition بیشتر است.
            if competitor_distance <= 0:
                competition += 1.0
            else:
                competition += 1.0 / (
                    1.0 + competitor_distance
                )

        competition = min(
            1.0,
            competition,
        )

    # Resource ارزشمندتر، حساسیت رقابتی بیشتری دارد.
        try:
            resource_value = float(
                resource.get(
                    "value",
                    resource.get(
                        "resource_value",
                        0.0,
                    ),
                )
                or 0.0
            )
        except (TypeError, ValueError):
            resource_value = 0.0

        value_factor = min(
            1.0,
            max(
                0.0,
                resource_value / 10.0,
            ),
        )

    # Risk ترکیبی از Competition و ارزش Resource است.
        risk = min(
            1.0,
            competition * (
                0.5 + 0.5 * value_factor
            ),
        )

    # ارزش Resource در کنار فاصله و Competition
    # برای تصمیم‌گیری استفاده می‌شود.
        distance_factor = 1.0 / (
            1.0 + distance
        )

        priority = (
            value_factor
            * distance_factor
            * (1.0 - 0.5 * competition)
        )

        if competition >= 0.75:
            strategy = "avoid"
        elif competition >= 0.35:
            strategy = "race"
        else:
            strategy = "approach"

        return {
            "distance": distance,
            "competition": competition,
            "nearest_competitor_distance": (
                nearest_competitor_distance
            ),
            "resource_value": resource_value,
            "risk": risk,
            "priority": priority,
            "strategy": strategy,
        }


    def choose_resource(
        self,
        agent_position,
        resources,
        competitors,
    ):
        """
        انتخاب بهترین Resource با درنظرگرفتن Competition.

        Resourceهای بسیار رقابتی حذف می‌شوند،
        مگر اینکه گزینه مناسب دیگری وجود نداشته باشد.
        """

        if not resources:
            return None

        evaluations = []

        for resource in resources:
            evaluation = self.evaluate_resource(
                agent_position,
                resource,
                competitors,
            )

            evaluations.append(
                (
                    resource,
                    evaluation,
                )
            )

    # ابتدا Resourceهای قابل‌تعقیب را بررسی می‌کنیم.
        preferred = [
            item
            for item in evaluations
            if item[1]["strategy"] != "avoid"
        ]

        candidates = (
            preferred
            if preferred
            else evaluations
        )

        candidates.sort(
            key=lambda item: item[1]["priority"],
            reverse=True,
        )

        resource, evaluation = candidates[0]

        return {
            "resource": resource,
            "evaluation": evaluation,
        }


    def get_strategy(
        self,
        agent_position,
        resource,
        competitors,
    ):
        """
        تعیین استراتژی رفتاری نسبت به یک Resource.
        """

        evaluation = self.evaluate_resource(
            agent_position,
            resource,
            competitors,
        )

        return evaluation["strategy"]
    
    @staticmethod
    def _distance(a, b):
        try:
            return sum(
                abs(int(x) - int(y))
                for x, y in zip(a, b)
            )
        except (TypeError, ValueError):
            return float("inf")
