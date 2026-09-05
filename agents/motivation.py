class MotivationModel:
    """
    مدل انگیزش چندگانه برای Agent.

    انگیزه‌های آشکار:
        - resource_need
        - curiosity
        - safety
        - competition
        - exploration
        - avoidance

    انگیزه‌های پنهان:
        - status
        - novelty_need
        - control
        - social_comparison

    Conflict:
        انگیزه‌های متضاد همزمان فعال می‌شوند.
        خروجی نهایی فقط یک انگیزه نیست؛
        بلکه یک motivation profile است.
    """

    def __init__(
        self,
        conflict_strength=0.50,
        hidden_influence=0.20,
        curiosity_decay=0.01,
        exploration_decay=0.01,
    ):
        self.conflict_strength = float(
            conflict_strength
        )

        self.hidden_influence = float(
            hidden_influence
        )

        self.curiosity_decay = float(
            curiosity_decay
        )

        self.exploration_decay = float(
            exploration_decay
        )

        # -----------------------------------------------------
        # Hidden motivational state
        # -----------------------------------------------------

        self.hidden_state = {
            "status": 0.0,
            "novelty_need": 0.0,
            "control": 0.0,
            "social_comparison": 0.0,
        }
        
        self.lack = 0.0
        self.desire = 0.0
        self.satisfaction = 0.0
        self.urgency = 0.0

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
    # Hidden Motives
    # =========================================================

    def update_hidden_state(
        self,
        lack=0.0,
        satisfaction=0.0,
        competition=0.0,
        novelty=0.0,
        success=0.0,
    ):
        """
        به‌روزرسانی انگیزه‌های پنهان.

        این انگیزه‌ها مستقیماً تصمیم نمی‌گیرند.
        ابتدا accumulated می‌شوند و بعد
        به انگیزه‌های آشکار اثر می‌گذارند.
        """

        lack = self._clip(lack)
        satisfaction = self._clip(
            satisfaction
        )
        competition = self._clip(
            competition
        )
        novelty = self._clip(novelty)
        success = self._clip(success)

        # -----------------------------------------------------
        # Status
        # -----------------------------------------------------

        status = self.hidden_state["status"]

        status += (
            0.03 * success
            + 0.02 * competition
            - 0.01 * satisfaction
        )

        self.hidden_state["status"] = (
            self._clip(status)
        )

        # -----------------------------------------------------
        # Novelty Need
        # -----------------------------------------------------

        novelty_need = (
            self.hidden_state["novelty_need"]
        )

        novelty_need += (
            0.03 * novelty
            + 0.01 * lack
        )

        novelty_need -= (
            self.curiosity_decay
            * satisfaction
        )

        self.hidden_state["novelty_need"] = (
            self._clip(novelty_need)
        )

        # -----------------------------------------------------
        # Control
        # -----------------------------------------------------

        control = self.hidden_state["control"]

        control += (
            0.02 * lack
            + 0.02 * competition
            - 0.015 * satisfaction
        )

        self.hidden_state["control"] = (
            self._clip(control)
        )

        # -----------------------------------------------------
        # Social Comparison
        # -----------------------------------------------------

        comparison = (
            self.hidden_state[
                "social_comparison"
            ]
        )

        comparison += (
            0.04 * competition
        )

        comparison -= (
            0.01 * satisfaction
        )

        self.hidden_state[
            "social_comparison"
        ] = self._clip(comparison)

    # =========================================================
    # Multiple Motives
    # =========================================================

    def calculate_motives(
        self,
        lack=0.0,
        satisfaction=0.0,
        urgency=0.0,
    ):
        """
        تولید انگیزه‌های آشکار.

        توجه:
        اینجا هنوز محیط را نمی‌شناسیم.
        فقط وضعیت درونی Agent بررسی می‌شود.
        """

        lack = self._clip(lack)
        satisfaction = self._clip(
            satisfaction
        )
        urgency = self._clip(urgency)

        hidden = self.hidden_state

        # -----------------------------------------------------
        # Resource Need
        # -----------------------------------------------------

        resource_need = (
            0.55 * lack
            + 0.35 * urgency
            + 0.10 * hidden["control"]
        )

        # -----------------------------------------------------
        # Curiosity
        # -----------------------------------------------------

        curiosity = (
            0.45 * (1.0 - satisfaction)
            + 0.35 * hidden["novelty_need"]
            + 0.20 * hidden["control"]
        )

        # -----------------------------------------------------
        # Safety
        # -----------------------------------------------------

        safety = (
            0.60 * satisfaction
            + 0.20 * (1.0 - lack)
            + 0.20 * (1.0 - urgency)
        )

        # -----------------------------------------------------
        # Competition
        # -----------------------------------------------------

        competition = (
            0.60 * hidden["social_comparison"]
            + 0.40 * hidden["status"]
        )

        # -----------------------------------------------------
        # Exploration
        # -----------------------------------------------------

        exploration = (
            0.50 * curiosity
            + 0.30 * hidden["novelty_need"]
            + 0.20 * (1.0 - satisfaction)
        )

        # -----------------------------------------------------
        # Avoidance
        # -----------------------------------------------------

        avoidance = (
            0.50 * safety
            + 0.30 * satisfaction
            + 0.20 * hidden["control"]
        )

        motives = {
            "resource_need": self._clip(
                resource_need
            ),
            "curiosity": self._clip(
                curiosity
            ),
            "safety": self._clip(
                safety
            ),
            "competition": self._clip(
                competition
            ),
            "exploration": self._clip(
                exploration
            ),
            "avoidance": self._clip(
                avoidance
            ),
        }

        return motives

    # =========================================================
    # Motivation Conflict
    # =========================================================

    def calculate_conflicts(
        self,
        motives,
    ):
        """
        تعارض بین انگیزه‌ها.

        نمونه:

            resource_need ↔ safety
            curiosity    ↔ safety
            exploration  ↔ avoidance
            competition  ↔ avoidance

        مقدار مثبت یعنی تعارض بیشتر.
        """

        resource = motives.get(
            "resource_need",
            0.0,
        )

        curiosity = motives.get(
            "curiosity",
            0.0,
        )

        safety = motives.get(
            "safety",
            0.0,
        )

        competition = motives.get(
            "competition",
            0.0,
        )

        exploration = motives.get(
            "exploration",
            0.0,
        )

        avoidance = motives.get(
            "avoidance",
            0.0,
        )

        conflicts = {
            "resource_vs_safety": (
                resource * safety
            ),

            "curiosity_vs_safety": (
                curiosity * safety
            ),

            "exploration_vs_avoidance": (
                exploration * avoidance
            ),

            "competition_vs_avoidance": (
                competition * avoidance
            ),
        }

        return {
            key: self._clip(value)
            for key, value
            in conflicts.items()
        }

    # =========================================================
    # Conflict Pressure
    # =========================================================

    def get_conflict_pressure(
        self,
        conflicts,
    ):
        """
        شدت کلی تعارض درونی.

        برای رفتار انسانی مهم است چون Agent
        همیشه نباید یک تصمیم کاملاً قطعی داشته باشد.
        """

        if not conflicts:
            return 0.0

        total = sum(
            conflicts.values()
        )

        average = (
            total / len(conflicts)
        )

        return self._clip(
            average
            * self.conflict_strength
        )

    # =========================================================
    # Final Motivation State
    # =========================================================

    def get_state(
        self,
        lack=None,
        satisfaction=None,
        urgency=None,
    ):
        if lack is None:
            lack = self.lack

        if satisfaction is None:
            satisfaction = self.satisfaction

        if urgency is None:
            urgency = self.urgency
    

        motives = self.calculate_motives(
            lack=lack,
            satisfaction=satisfaction,
            urgency=urgency,
        )

        conflicts = (
            self.calculate_conflicts(
                motives
            )
        )

        conflict_pressure = (
            self.get_conflict_pressure(
                conflicts
            )
        )
        
        dynamics = self.get_motivation_dynamics(
            motives,
            conflicts,
        )

        # -----------------------------------------------------
        # Desire
        # -----------------------------------------------------

        desire = (
            0.45
            * motives["resource_need"]
            + 0.20
            * motives["curiosity"]
            + 0.15
            * motives["competition"]
            + 0.20
            * motives["exploration"]
        )

        # تعارض زیاد، Desire را حذف نمی‌کند؛
        # بلکه تصمیم را ناپایدارتر می‌کند.
        desire *= (
            1.0
            + 0.20
            * conflict_pressure
        )

        return {
            "lack": self._clip(lack),

            "desire": self._clip(
                desire
            ),

            "satisfaction": self._clip(
                satisfaction
            ),

            "urgency": self._clip(
                urgency
            ),

            "motives": motives,

            "conflicts": conflicts,

            "conflict_pressure": (
                conflict_pressure
            ),

            "motivation_dynamics": (
                dynamics
            ),

            "hidden_motives": (
                self.hidden_state.copy()
            ),
        }
        
    # =========================================================
    # Experience Update
    # =========================================================

    def update_from_experience(
        self,
        lack=0.0,
        satisfaction=0.0,
        competition=0.0,
        novelty=0.0,
        success=0.0,
    ):
        """
        این متد باید بعد از تجربه Agent فراخوانی شود.
        """

        self.update_hidden_state(
            lack=lack,
            satisfaction=satisfaction,
            competition=competition,
            novelty=novelty,
            success=success,
        )

    # =========================================================
    # Reset
    # =========================================================

    def reset_episode(self):
        """
        Hidden motives بین Episodeها پاک نمی‌شوند.

        فقط کمی کاهش می‌یابند تا شخصیت Agent
        به‌طور کامل reset نشود ولی تغییرپذیر باقی بماند.
        """

        for key in self.hidden_state:
            self.hidden_state[key] *= 0.98
            
            
        # =========================================================
    # Motivation Conflict Analysis
    # =========================================================

    def get_motivation_dynamics(
        self,
        motives,
        conflicts,
    ):
        """
        تحلیل وضعیت انگیزشی Agent.

        خروجی مشخص می‌کند:
            - انگیزه غالب چیست
            - انگیزه دوم چیست
            - آیا تعارض جدی وجود دارد
            - تعارض اصلی بین کدام انگیزه‌هاست
        """

        if not motives:
            return {
                "dominant_motive": None,
                "secondary_motive": None,
                "main_conflict": None,
                "conflict_pressure": 0.0,
            }

        ordered = sorted(
            motives.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        dominant_motive = ordered[0][0]

        secondary_motive = (
            ordered[1][0]
            if len(ordered) > 1
            else None
        )

        main_conflict = None
        main_conflict_value = 0.0

        if conflicts:

            main_conflict, main_conflict_value = max(
                conflicts.items(),
                key=lambda item: item[1],
            )

        pressure = self.get_conflict_pressure(
            conflicts
        )

        return {
            "dominant_motive": dominant_motive,

            "secondary_motive": secondary_motive,

            "main_conflict": main_conflict,

            "main_conflict_value": (
                self._clip(
                    main_conflict_value
                )
            ),

            "conflict_pressure": pressure,
        }
