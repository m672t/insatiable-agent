"""
SubjectState — Subject Architecture

Current conceptual model:

    Need
      ↓
    Demand ───────────────→ Other
      ↓                       ↓
    Lack ←────────────────────┘
      ↓
    Desire
      ↓
    Object / Object-Cause
      ↓
    Action

Persistent subjective influences:

    Memory
    Other
    Conflict
    Repetition
    Identification

Important principles:

1. Need and Desire are not identical.
2. Demand is not merely a stronger Need; it is directed toward an Other.
3. Lack is not erased by satisfaction.
4. Desire is not erased by satisfaction.
5. Repetition survives satisfaction.
6. Desire may attach to an object/signifier.
7. An object can remain desirable even after its immediate reward
   has been obtained.
8. Other and identification can alter desire independently of
   biological need.
9. Conflict is represented explicitly and can influence action tendency.
10. All numeric pressures remain bounded.

This module remains deliberately lightweight. It does not attempt to
implement psychoanalysis as a clinical or literal simulation.
It provides computational structures inspired by the conceptual
architecture being explored in this project.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Optional


class SubjectState:
    """Persistent, bounded subject state for a single agent."""

    CORE_KEYS = (
        "need",
        "demand",
        "lack",
        "desire",
        "satisfaction",
        "urgency",
    )

    def __init__(
        self,
        *,
        need: float = 0.0,
        demand: float = 0.0,
        lack: float = 0.0,
        desire: float = 0.0,
        satisfaction: float = 0.0,
        urgency: float = 0.0,
        memory_decay: float = 0.985,
        desire_persistence: float = 0.08,
        lack_persistence: float = 0.18,
        satisfaction_decay: float = 0.92,
    ) -> None:
        self._state: Dict[str, Any] = {
            "need": self._clip(need),
            "demand": self._clip(demand),
            "lack": self._clip(lack),
            "desire": self._clip(desire),
            "satisfaction": self._clip(satisfaction),
            "urgency": self._clip(urgency),

            # ----------------------------------------------------------
            # Subjective / relational structures
            # ----------------------------------------------------------

            "memory_influence": {},
            "other_influence": {},
            "conflict": {},
            "repetition": {},
            "action_tendencies": {},

            # ----------------------------------------------------------
            # New symbolic / desire structures
            # ----------------------------------------------------------

            # Per-object desire state.
            #
            # Example:
            #
            # "recognition": {
            #     "desire": 0.72,
            #     "lack": 0.61,
            #     "investment": 0.83,
            #     "satisfaction": 0.20,
            #     "persistence": 0.08,
            #     "object_cause": 0.75,
            # }
            #
            "desire_objects": {},

            # Object-cause trace.
            #
            # This deliberately represents an object as something that
            # continues to organize desire rather than as a simple
            # reward target.
            "object_cause": {},

            # Demand as an addressed relation.
            "demands": {},

            # Signifiers associated with objects/experiences.
            "signifiers": {},
        }

        self.memory_decay = self._clip(memory_decay, 0.0, 1.0)
        self.desire_persistence = self._clip(
            desire_persistence,
            0.0,
            1.0,
        )
        self.lack_persistence = self._clip(
            lack_persistence,
            0.0,
            1.0,
        )
        self.satisfaction_decay = self._clip(
            satisfaction_decay,
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Persistent traces
        # --------------------------------------------------------------

        self._memory_trace = defaultdict(float)
        self._other_trace = defaultdict(float)
        self._repetition_trace = defaultdict(float)

        # Per-object traces.
        self._object_desire_trace = defaultdict(float)
        self._object_memory_trace = defaultdict(float)
        self._object_other_trace = defaultdict(float)
        self._object_repetition_trace = defaultdict(float)
        self._object_lack_trace = defaultdict(float)
        self._object_satisfaction_trace = defaultdict(float)
        self._object_cause_trace = defaultdict(float)

    # ==================================================================
    # Numeric helpers
    # ==================================================================

    @staticmethod
    def _clip(
        value: Any,
        lo: float = 0.0,
        hi: float = 1.0,
    ) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = lo

        return max(lo, min(hi, value))

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    # ==================================================================
    # Public state API
    # ==================================================================

    def get_state(self) -> Dict[str, Any]:
        """Return a safe snapshot of the complete subject state."""
        return deepcopy(self._state)

    def set(self, name: str, value: Any) -> None:
        if name not in self.CORE_KEYS:
            raise KeyError(
                f"Unknown core subject variable: {name}"
            )

        self._state[name] = self._clip(value)

    def get(
        self,
        name: str,
        default: Any = 0.0,
    ) -> Any:
        return deepcopy(
            self._state.get(name, default)
        )

    # ------------------------------------------------------------------
    # Compatibility setters
    # ------------------------------------------------------------------

    def set_need(self, *args: Any) -> None:
        self.set("need", args[-1])

    def set_demand(self, *args: Any) -> None:
        self.set("demand", args[-1])

    def set_lack(self, *args: Any) -> None:
        self.set("lack", args[-1])

    def set_desire(self, *args: Any) -> None:
        self.set("desire", args[-1])

    def set_satisfaction(self, *args: Any) -> None:
        self.set("satisfaction", args[-1])

    def set_urgency(self, *args: Any) -> None:
        self.set("urgency", args[-1])

    # ------------------------------------------------------------------
    # Compatibility getters
    # ------------------------------------------------------------------

    def get_need(self, *args: Any) -> float:
        return self.get("need")

    def get_demand(self, *args: Any) -> float:
        return self.get("demand")

    def get_lack(self, *args: Any) -> float:
        return self.get("lack")

    def get_desire(self, *args: Any) -> float:
        return self.get("desire")

    def get_satisfaction(self, *args: Any) -> float:
        return self.get("satisfaction")

    def get_urgency(self, *args: Any) -> float:
        return self.get("urgency")

    # ==================================================================
    # Core dynamics
    # ==================================================================

    def update(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Update subject dynamics.

        Explicitly supplied core values are preserved.

        Otherwise the subject layer derives soft relationships between:

            Need
            Demand
            Lack
            Desire
            Satisfaction
            Urgency

        Desire additionally receives pressure from:

            Memory
            Other
            Repetition
            Object-cause
            Per-object investment

        Satisfaction reduces immediate pressure but does not erase
        subjective structures.
        """

        supplied: Dict[str, Any] = {}

        if args and isinstance(args[0], dict):
            supplied.update(args[0])

        supplied.update(kwargs)

        # --------------------------------------------------------------
        # Explicit values always win.
        # --------------------------------------------------------------

        for key in self.CORE_KEYS:
            if key in supplied:
                self._state[key] = self._clip(
                    supplied[key]
                )

        need = self._state["need"]
        demand = self._state["demand"]
        lack = self._state["lack"]
        desire = self._state["desire"]
        satisfaction = self._state["satisfaction"]
        urgency = self._state["urgency"]

        # --------------------------------------------------------------
        # Demand
        # --------------------------------------------------------------

        if "demand" not in supplied:
            demand = self._clip(
                0.65 * demand
                + 0.35 * need
            )

        # --------------------------------------------------------------
        # Lack
        #
        # Satisfaction reduces immediate lack, but persistence prevents
        # it from collapsing to zero merely because something was
        # obtained.
        # --------------------------------------------------------------

        if "lack" not in supplied:
            immediate_lack = (
                0.45 * max(need, demand)
                + self.lack_persistence * lack
                - 0.18 * satisfaction
            )

            lack = self._clip(
                0.55 * lack
                + 0.45 * immediate_lack
            )

        # --------------------------------------------------------------
        # Global pressures
        # --------------------------------------------------------------

        memory_pressure = self._memory_pressure()
        other_pressure = self._other_pressure()
        repetition_pressure = self._repetition_pressure()
        object_pressure = self._object_desire_pressure()
        cause_pressure = self._object_cause_pressure()

        # --------------------------------------------------------------
        # Desire
        # --------------------------------------------------------------

        if "desire" not in supplied:

            target_desire = (
                0.34 * lack
                + 0.12 * demand
                + 0.12 * memory_pressure
                + 0.09 * other_pressure
                + 0.08 * repetition_pressure
                + 0.10 * object_pressure
                + 0.08 * cause_pressure
                + self.desire_persistence * desire
                - 0.08 * satisfaction
            )

            desire = self._clip(
                0.65 * desire
                + 0.35 * target_desire
            )

        # --------------------------------------------------------------
        # Urgency
        # --------------------------------------------------------------

        if "urgency" not in supplied:

            urgency_target = (
                0.55 * lack
                + 0.45 * max(
                    desire,
                    need,
                )
                - 0.35 * satisfaction
            )

            # Repetition can create pressure without requiring fresh
            # biological need.
            urgency_target += (
                0.08 * repetition_pressure
            )

            # Object-cause can keep something pressing even when
            # immediate satisfaction is high.
            urgency_target += (
                0.06 * cause_pressure
            )

            urgency = self._clip(
                0.55 * urgency
                + 0.45 * urgency_target
            )

        # --------------------------------------------------------------
        # Save derived state
        # --------------------------------------------------------------

        self._state["demand"] = self._clip(demand)
        self._state["lack"] = self._clip(lack)
        self._state["desire"] = self._clip(desire)
        self._state["urgency"] = self._clip(urgency)

        # --------------------------------------------------------------
        # Per-object desire dynamics
        # --------------------------------------------------------------

        self._update_object_desires(
            global_lack=lack,
            global_desire=desire,
            satisfaction=satisfaction,
        )

        # --------------------------------------------------------------
        # Satisfaction itself slowly decays.
        #
        # This means satisfaction is a temporary subjective condition,
        # not a permanent reset.
        #
        # Explicitly supplied satisfaction is preserved for this update.
        # --------------------------------------------------------------

        if "satisfaction" not in supplied:
            self._state["satisfaction"] = self._clip(
                satisfaction * self.satisfaction_decay
            )

        self._decay_traces()

        return self.get_state()

    # ==================================================================
    # Memory
    # ==================================================================

    def register_memory_influence(
        self,
        *,
        object_id: Any = "unknown",
        reward: float = 0.0,
        success: bool = False,
        strength: float = 1.0,
        signifier: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Register memory influence.

        Memory is not copied directly into desire. Instead it leaves
        a persistent trace which can later increase desire pressure.
        """

        key = str(object_id)

        strength_value = self._clip(strength)

        reward_signal = self._clip(
            self._safe_float(reward) / 50.0,
            -1.0,
            1.0,
        )

        success_signal = (
            1.0 if success else 0.0
        )

        contribution = (
            strength_value
            * (
                0.7 * reward_signal
                + 0.3 * success_signal
            )
        )

        self._memory_trace[key] += contribution

        self._object_memory_trace[key] += max(
            0.0,
            contribution,
        )

        if success:
            self.register_repetition(
                object_id=key,
                strength=strength_value,
            )

        self._state["memory_influence"][key] = {
            "reward": self._safe_float(reward),
            "success": bool(success),
            "strength": strength_value,
            "influence": self._clip(
                contribution,
                -1.0,
                1.0,
            ),
        }

        # --------------------------------------------------------------
        # Optional signifier association
        # --------------------------------------------------------------

        if signifier is not None:
            self.register_signifier(
                signifier=signifier,
                object_id=key,
                strength=strength_value,
            )

    # Compatibility alias.
    register_memory = register_memory_influence

    def _memory_pressure(self) -> float:
        if not self._memory_trace:
            return 0.0

        positive = [
            max(0.0, value)
            for value in self._memory_trace.values()
        ]

        return self._clip(
            sum(positive)
            / max(1, len(positive))
        )

    # ==================================================================
    # Other
    # ==================================================================

    def register_other_influence(
        self,
        *,
        other_id: Any = "unknown",
        object_id: Any = "unknown",
        observed_desire: float = 0.0,
        observed_success: float = 0.0,
        identification: float = 0.0,
        strength: float = 1.0,
        signifier: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Register influence from the Other.

        The Other can alter desire independently of biological need.

        Identification controls how strongly the observed desire of the
        Other becomes subjectively relevant.
        """

        other_key = str(other_id)
        object_key = str(object_id)
        key = f"{other_key}:{object_key}"

        strength_value = self._clip(strength)
        identification_value = self._clip(
            identification
        )

        observed_desire_value = self._clip(
            observed_desire
        )

        observed_success_value = self._clip(
            observed_success
        )

        contribution = (
            strength_value
            * identification_value
            * (
                0.65 * observed_desire_value
                + 0.35 * observed_success_value
            )
        )

        self._other_trace[key] += contribution
        self._object_other_trace[object_key] += contribution

        self._state["other_influence"][key] = {
            "other_id": other_key,
            "object_id": object_key,
            "observed_desire": observed_desire_value,
            "observed_success": observed_success_value,
            "identification": identification_value,
            "strength": strength_value,
            "influence": self._clip(
                contribution
            ),
        }

        if signifier is not None:
            self.register_signifier(
                signifier=signifier,
                object_id=object_key,
                strength=strength_value
                * identification_value,
            )

    register_other = register_other_influence

    def _other_pressure(self) -> float:
        if not self._other_trace:
            return 0.0

        return self._clip(
            sum(
                max(0.0, value)
                for value in self._other_trace.values()
            )
            / max(1, len(self._other_trace))
        )

    # ==================================================================
    # Demand / address to Other
    # ==================================================================

    def register_demand(
        self,
        *,
        object_id: Any = "unknown",
        other_id: Any = "unknown",
        intensity: float = 0.0,
        recognition: float = 0.0,
        dependency: float = 0.0,
        signifier: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Register a demand as something addressed to an Other.

        This separates Demand from raw Need.

        Example:

            Need:
                0.40

            Demand:
                "I want recognition from Other A"

        The computational state keeps the relational structure rather
        than reducing everything to one scalar.
        """

        object_key = str(object_id)
        other_key = str(other_id)
        demand_key = f"{other_key}:{object_key}"

        self._state["demands"][demand_key] = {
            "object_id": object_key,
            "other_id": other_key,
            "intensity": self._clip(intensity),
            "recognition": self._clip(recognition),
            "dependency": self._clip(dependency),
        }

        if signifier is not None:
            self.register_signifier(
                signifier=signifier,
                object_id=object_key,
                strength=self._clip(intensity),
            )

    # ==================================================================
    # Signifiers
    # ==================================================================

    def register_signifier(
        self,
        *,
        signifier: Any,
        object_id: Any = "unknown",
        strength: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Associate a signifier with an object.

        A signifier is intentionally represented as an association,
        not as a fixed semantic dictionary.

        Multiple objects may acquire investment in the same signifier,
        and one object may carry multiple signifiers.
        """

        signifier_key = str(signifier)
        object_key = str(object_id)

        strength_value = self._clip(strength)

        if signifier_key not in self._state["signifiers"]:
            self._state["signifiers"][signifier_key] = {
                "objects": {},
                "strength": 0.0,
            }

        entry = self._state["signifiers"][signifier_key]

        previous = entry["objects"].get(
            object_key,
            0.0,
        )

        entry["objects"][object_key] = self._clip(
            previous + strength_value
        )

        entry["strength"] = self._clip(
            max(
                entry["objects"].values()
            )
            if entry["objects"]
            else 0.0
        )

    # ==================================================================
    # Conflict
    # ==================================================================

    def register_conflict(
        self,
        *,
        approach: Any = None,
        avoidance: Any = None,
        approach_strength: float = 0.0,
        avoidance_strength: float = 0.0,
        object_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """
        Register an internal conflict.

        Conflict is represented explicitly and can later influence
        action selection.
        """

        conflict_id = (
            f"{approach}|{avoidance}"
        )

        a = self._clip(
            approach_strength
        )

        b = self._clip(
            avoidance_strength
        )

        ambivalence = self._clip(
            min(a, b) * 2.0
        )

        net_tendency = a - b

        self._state["conflict"][conflict_id] = {
            "approach": approach,
            "avoidance": avoidance,
            "approach_strength": a,
            "avoidance_strength": b,
            "ambivalence": ambivalence,
            "net_tendency": net_tendency,
            "object_id": (
                None
                if object_id is None
                else str(object_id)
            ),
        }

    # ==================================================================
    # Action tendency
    # ==================================================================

    def register_action(
        self,
        *,
        action: Any,
        tendency: float = 0.0,
        object_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """
        Register an action tendency.

        Negative values represent inhibition/avoidance.
        Positive values represent approach tendency.
        """

        key = str(action)

        self._state["action_tendencies"][key] = {
            "tendency": self._clip(
                tendency,
                -1.0,
                1.0,
            ),
            "object_id": (
                None
                if object_id is None
                else str(object_id)
            ),
        }

    # ==================================================================
    # Repetition
    # ==================================================================

    def register_repetition(
        self,
        *,
        object_id: Any = "unknown",
        strength: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Register repetition.

        Count is persistent and is NOT erased by satisfaction.

        Strength represents current recurrence pressure.
        Count represents historical recurrence.
        These are deliberately separate.
        """

        key = str(object_id)

        strength_value = self._clip(
            strength
        )

        self._repetition_trace[key] += (
            strength_value
        )

        previous_count = int(
            self._state["repetition"]
            .get(key, {})
            .get("count", 0)
        )

        new_count = previous_count + 1

        self._state["repetition"][key] = {
            "count": new_count,
            "strength": self._clip(
                self._repetition_trace[key]
            ),
        }

        self._object_repetition_trace[key] += (
            strength_value
        )

    def _repetition_pressure(self) -> float:
        if not self._repetition_trace:
            return 0.0

        return self._clip(
            sum(
                self._repetition_trace.values()
            )
            / max(
                1,
                len(self._repetition_trace),
            )
        )

    # ==================================================================
    # Object / Object-Cause
    # ==================================================================

    def register_object_desire(
        self,
        *,
        object_id: Any,
        desire: float = 0.0,
        lack: Optional[float] = None,
        satisfaction: float = 0.0,
        object_cause: float = 0.5,
        investment: float = 0.0,
        persistence: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Register subjective investment in a particular object.

        This is intentionally different from reward.

        An object may have:

            high satisfaction
            high reward
            high historical investment

        while still retaining desire pressure.

        `object_cause` represents how strongly the object organizes
        desire as an absent/unfinished cause rather than simply being
        a reward target.
        """

        key = str(object_id)

        desired = self._clip(desire)

        object_lack = (
            self._clip(lack)
            if lack is not None
            else self._clip(
                self._object_lack_trace[key]
            )
        )

        object_satisfaction = self._clip(
            satisfaction
        )

        cause = self._clip(
            object_cause
        )

        investment_value = self._clip(
            investment
        )

        persistence_value = (
            self.desire_persistence
            if persistence is None
            else self._clip(persistence)
        )

        self._object_desire_trace[key] = (
            0.7
            * self._object_desire_trace[key]
            + 0.3
            * desired
        )

        self._object_lack_trace[key] = (
            0.7
            * self._object_lack_trace[key]
            + 0.3
            * object_lack
        )

        self._object_satisfaction_trace[key] = (
            0.7
            * self._object_satisfaction_trace[key]
            + 0.3
            * object_satisfaction
        )

        self._object_cause_trace[key] = (
            0.7
            * self._object_cause_trace[key]
            + 0.3
            * cause
        )

        self._state["object_cause"][key] = {
            "strength": self._clip(
                self._object_cause_trace[key]
            ),
            "satisfaction": object_satisfaction,
            "investment": investment_value,
            "persistence": persistence_value,
        }

        self._state["desire_objects"][key] = {
            "desire": self._clip(
                self._object_desire_trace[key]
            ),
            "lack": self._clip(
                self._object_lack_trace[key]
            ),
            "satisfaction": self._clip(
                self._object_satisfaction_trace[key]
            ),
            "object_cause": self._clip(
                self._object_cause_trace[key]
            ),
            "investment": investment_value,
            "persistence": persistence_value,
        }

    # Compatibility alias.
    register_object = register_object_desire

    def get_object_desire(
        self,
        object_id: Any,
        default: float = 0.0,
    ) -> float:
        key = str(object_id)

        return self._clip(
            self._state["desire_objects"]
            .get(key, {})
            .get("desire", default)
        )

    def get_object_state(
        self,
        object_id: Any,
    ) -> Dict[str, Any]:
        key = str(object_id)

        return deepcopy(
            self._state["desire_objects"].get(
                key,
                {},
            )
        )

    def register_object_satisfaction(
        self,
        *,
        object_id: Any,
        satisfaction: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Register satisfaction for an object.

        Crucially, this does NOT delete:

            repetition
            object-cause
            memory
            investment
            desire
            lack
        """

        key = str(object_id)

        value = self._clip(
            satisfaction
        )

        self._object_satisfaction_trace[key] = (
            self._clip(
                0.7
                * self._object_satisfaction_trace[key]
                + 0.3 * value
            )
        )

        current = self._state[
            "desire_objects"
        ].get(
            key,
            {
                "desire": 0.0,
                "lack": 0.0,
                "investment": 0.0,
                "persistence": self.desire_persistence,
                "object_cause": self._object_cause_trace[
                    key
                ],
            },
        )

        current["satisfaction"] = (
            self._object_satisfaction_trace[key]
        )

        # Satisfaction reduces desire but does not erase it.
        current["desire"] = self._clip(
            current.get("desire", 0.0)
            * (
                1.0
                - 0.35 * value
            )
            + current.get(
                "object_cause",
                0.0,
            )
            * 0.08
            + current.get(
                "persistence",
                self.desire_persistence,
            )
            * 0.05
        )

        self._object_desire_trace[key] = (
            current["desire"]
        )

        self._state[
            "desire_objects"
        ][key] = current

    # ==================================================================
    # Object pressure helpers
    # ==================================================================

    def _object_desire_pressure(self) -> float:
        if not self._object_desire_trace:
            return 0.0

        values = [
            max(0.0, value)
            for value
            in self._object_desire_trace.values()
        ]

        if not values:
            return 0.0

        return self._clip(
            sum(values)
            / max(1, len(values))
        )

    def _object_cause_pressure(self) -> float:
        if not self._object_cause_trace:
            return 0.0

        values = [
            max(0.0, value)
            for value
            in self._object_cause_trace.values()
        ]

        if not values:
            return 0.0

        return self._clip(
            sum(values)
            / max(1, len(values))
        )

    # ==================================================================
    # Per-object dynamics
    # ==================================================================

    def _update_object_desires(
        self,
        *,
        global_lack: float,
        global_desire: float,
        satisfaction: float,
    ) -> None:
        """
        Update object-specific desire without collapsing it into the
        global scalar Desire.

        This is where symbolic investment begins to affect the subject.
        """

        object_keys = set(
            self._object_desire_trace.keys()
        )

        object_keys.update(
            self._object_cause_trace.keys()
        )

        object_keys.update(
            self._object_memory_trace.keys()
        )

        object_keys.update(
            self._object_other_trace.keys()
        )

        object_keys.update(
            self._object_repetition_trace.keys()
        )

        for key in object_keys:

            current = self._object_desire_trace[
                key
            ]

            memory = self._clip(
                self._object_memory_trace[key]
            )

            other = self._clip(
                self._object_other_trace[key]
            )

            repetition = self._clip(
                self._object_repetition_trace[key]
            )

            object_lack = self._clip(
                self._object_lack_trace[key]
            )

            cause = self._clip(
                self._object_cause_trace[key]
            )

            object_satisfaction = self._clip(
                self._object_satisfaction_trace[key]
            )

            target = (
                0.20 * global_desire
                + 0.18 * global_lack
                + 0.16 * object_lack
                + 0.12 * memory
                + 0.10 * other
                + 0.10 * repetition
                + 0.14 * cause
                - 0.10 * object_satisfaction
            )

            updated = self._clip(
                0.70 * current
                + 0.30 * target
            )

            self._object_desire_trace[key] = (
                updated
            )

            previous_state = self._state[
                "desire_objects"
            ].get(
                key,
                {},
            )

            self._state[
                "desire_objects"
            ][key] = {
                "desire": updated,
                "lack": object_lack,
                "satisfaction": object_satisfaction,
                "investment": self._clip(
                    previous_state.get(
                        "investment",
                        0.0,
                    )
                ),
                "persistence": self._clip(
                    previous_state.get(
                        "persistence",
                        self.desire_persistence,
                    )
                ),
                "object_cause": cause,
            }

            self._state[
                "object_cause"
            ][key] = {
                "strength": cause,
                "satisfaction": object_satisfaction,
                "investment": self._clip(
                    previous_state.get(
                        "investment",
                        0.0,
                    )
                ),
                "persistence": self._clip(
                    previous_state.get(
                        "persistence",
                        self.desire_persistence,
                    )
                ),
            }

    # ==================================================================
    # Trace decay
    # ==================================================================

    def _decay_traces(self) -> None:
        """
        Decay temporary influence traces.

        Historical structures such as repetition count and signifier
        associations are intentionally NOT deleted here.
        """

        for trace in (
            self._memory_trace,
            self._other_trace,
            self._repetition_trace,
            self._object_memory_trace,
            self._object_other_trace,
            self._object_repetition_trace,
        ):
            for key in list(trace):
                trace[key] *= self.memory_decay

                if abs(trace[key]) < 1e-6:
                    del trace[key]

        # Object desire/lack/cause are more persistent than raw pressure.
        for trace in (
            self._object_desire_trace,
            self._object_lack_trace,
            self._object_cause_trace,
        ):
            for key in list(trace):
                trace[key] *= (
                    0.995
                )

                if abs(trace[key]) < 1e-6:
                    del trace[key]

    # ==================================================================
    # Convenience / diagnostics
    # ==================================================================

    def get_desire_profile(self) -> Dict[str, Any]:
        """
        Return a compact representation of the current desire field.
        """

        return {
            "global_desire": self._state[
                "desire"
            ],
            "global_lack": self._state[
                "lack"
            ],
            "satisfaction": self._state[
                "satisfaction"
            ],
            "objects": deepcopy(
                self._state[
                    "desire_objects"
                ]
            ),
            "object_causes": deepcopy(
                self._state[
                    "object_cause"
                ]
            ),
        }

    def get_conflict_pressure(self) -> float:
        """
        Return aggregate ambivalence pressure.

        This is not yet an action-selection algorithm; it is simply
        a bounded measure that downstream systems can consume.
        """

        if not self._state["conflict"]:
            return 0.0

        values = [
            self._clip(
                item.get(
                    "ambivalence",
                    0.0,
                )
            )
            for item
            in self._state["conflict"].values()
        ]

        if not values:
            return 0.0

        return self._clip(
            sum(values)
            / len(values)
        )

    def get_action_pressure(
        self,
        action: Any,
    ) -> float:
        """
        Return the currently registered tendency for an action.
        """

        key = str(action)

        value = self._state[
            "action_tendencies"
        ].get(
            key,
            0.0,
        )

        if isinstance(value, dict):
            return self._clip(
                value.get(
                    "tendency",
                    0.0,
                ),
                -1.0,
                1.0,
            )

        return self._clip(
            value,
            -1.0,
            1.0,
        )


__all__ = ["SubjectState"]
