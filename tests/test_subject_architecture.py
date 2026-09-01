"""
Stage 6.5 — Subject Architecture Contract Test
===============================================

Purpose
-------
This test defines the minimum behavioral contract required before entering
Stage 7–10 experiments.

It intentionally tests the SUBJECT architecture, not reward optimization.

Required conceptual chain:
    Need -> Demand -> Lack -> Desire -> Object

and:
    Satisfaction does not automatically erase Lack/Desire.
    Memory can modify Desire.
    Other can modify Desire.
    Conflict can modify action selection.
    Repetition can persist after satisfaction.

This file is written as a contract test.  If the new SubjectState layer does
not exist yet, the test FAILS explicitly instead of silently falling back to
the old motivation dictionary.

Expected public API
-------------------
The project should expose:

    from agents.subject_state import SubjectState

The class should support:

    SubjectState()
    state.get_state()
    state.set_need(name, value)
    state.set_demand(name, value)
    state.set_desire(name, value)
    state.set_satisfaction(name, value)
    state.set_lack(name, value)
    state.set_urgency(name, value)
    state.update(...)
    state.register_memory_influence(...)
    state.register_other_influence(...)
    state.register_conflict(...)
    state.register_action(...)
    state.get_desire(name)
    state.get_lack(name)
    state.get_satisfaction(name)
    state.get_need(name)
    state.get_demand(name)

If your implementation uses different method names, change ONLY the adapter
section below.  The behavioral assertions should remain unchanged.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

IMPORT_ERROR = None

try:
    from agents.subject_state import SubjectState
except Exception as exc:  # pragma: no cover - intentional contract failure
    SubjectState = None
    IMPORT_ERROR = exc


def fail(message: str) -> None:
    raise AssertionError(message)


def make_state() -> Any:
    if SubjectState is None:
        fail(
            "SUBJECT ARCHITECTURE MISSING: "
            "could not import agents.subject_state.SubjectState. "
            f"Import error: {IMPORT_ERROR!r}"
        )
    return SubjectState()


def get_value(state: Any, name: str, default: float = 0.0) -> float:
    method = getattr(state, f"get_{name}", None)
    if callable(method):
        try:
            return float(method(name))
        except TypeError:
            return float(method())

    data = state.get_state()
    value = data.get(name, default)

    if isinstance(value, dict):
        value = value.get(name, default)

    return float(value)


def set_value(state: Any, name: str, value: float) -> None:
    method = getattr(state, f"set_{name}", None)
    if callable(method):
        try:
            method(name, value)
        except TypeError:
            method(value)
        return

    method = getattr(state, "set", None)
    if callable(method):
        method(name, value)
        return

    fail(
        f"SubjectState has no public setter for '{name}'. "
        "Add one or update the adapter."
    )


def update_state(state: Any, **kwargs: Any) -> None:
    method = getattr(state, "update", None)
    if not callable(method):
        fail("SubjectState must expose update(...).")

    try:
        method(**kwargs)
    except TypeError:
        method(kwargs)


def register(state: Any, kind: str, **kwargs: Any) -> None:
    method = getattr(state, f"register_{kind}_influence", None)

    if callable(method):
        try:
            method(**kwargs)
        except TypeError:
            method(kwargs)
        return

    method = getattr(state, f"register_{kind}", None)

    if callable(method):
        try:
            method(**kwargs)
        except TypeError:
            method(kwargs)
        return

    fail(
        f"SubjectState has no public '{kind}' influence API. "
        "This behavior must be represented explicitly."
    )


def snapshot(state: Any) -> Dict[str, Any]:
    data = state.get_state()
    if not isinstance(data, dict):
        fail("SubjectState.get_state() must return a dict.")
    return data


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def assertion(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(
            f"{name} FAILED"
            + (f": {detail}" if detail else "")
        )
    print(f"PASS  {name}")


# ---------------------------------------------------------------------------
# TEST 0 — SubjectState exists and is structured
# ---------------------------------------------------------------------------

def test_subject_state_exists() -> None:
    banner("TEST 0 — SUBJECT STATE EXISTS")

    state = make_state()
    data = snapshot(state)

    required = {
        "need",
        "demand",
        "lack",
        "desire",
        "satisfaction",
        "urgency",
    }

    missing = required - set(data.keys())

    assertion(
        "Subject state contains core drives",
        not missing,
        f"missing keys: {sorted(missing)}",
    )


# ---------------------------------------------------------------------------
# TEST 1 — Need changes behavior/state
# ---------------------------------------------------------------------------

def test_need_changes_state() -> None:
    banner("TEST 1 — NEED CHANGES STATE")

    state = make_state()

    set_value(state, "need", 0.0)
    update_state(state)
    baseline = snapshot(state)

    set_value(state, "need", 0.8)
    update_state(state)
    high = snapshot(state)

    assertion(
        "Need is stateful",
        baseline != high,
        "changing Need must change SubjectState",
    )

    assertion(
        "Need remains bounded",
        0.0 <= get_value(state, "need") <= 1.0,
        f"need={get_value(state, 'need')}",
    )


# ---------------------------------------------------------------------------
# TEST 2 — Need -> Demand -> Lack -> Desire
# ---------------------------------------------------------------------------

def test_need_demand_lack_desire_chain() -> None:
    banner("TEST 2 — NEED → DEMAND → LACK → DESIRE")

    state = make_state()

    set_value(state, "need", 0.9)
    set_value(state, "demand", 0.8)
    set_value(state, "lack", 0.7)
    set_value(state, "desire", 0.6)
    update_state(state)

    need = get_value(state, "need")
    demand = get_value(state, "demand")
    lack = get_value(state, "lack")
    desire = get_value(state, "desire")

    assertion("Need present", need > 0.0, f"need={need}")
    assertion("Demand present", demand > 0.0, f"demand={demand}")
    assertion("Lack present", lack > 0.0, f"lack={lack}")
    assertion("Desire present", desire > 0.0, f"desire={desire}")


# ---------------------------------------------------------------------------
# TEST 3 — Desire can diverge from Need
# ---------------------------------------------------------------------------

def test_desire_can_diverge_from_need() -> None:
    banner("TEST 3 — DESIRE CAN DIVERGE FROM NEED")

    state = make_state()

    # Low immediate need but persistent desire.
    set_value(state, "need", 0.1)
    set_value(state, "desire", 0.8)
    update_state(state)

    need = get_value(state, "need")
    desire = get_value(state, "desire")

    assertion(
        "Desire is not hard-coded to Need",
        desire > need,
        f"need={need}, desire={desire}",
    )


# ---------------------------------------------------------------------------
# TEST 4 — Satisfaction reduces pressure but does NOT erase Lack/Desire
# ---------------------------------------------------------------------------

def test_satisfaction_does_not_erase_lack_or_desire() -> None:
    banner("TEST 4 — SATISFACTION ≠ ERASURE OF LACK/DESIRE")

    state = make_state()

    set_value(state, "need", 0.8)
    set_value(state, "lack", 0.7)
    set_value(state, "desire", 0.9)
    set_value(state, "satisfaction", 0.0)
    update_state(state)

    before_lack = get_value(state, "lack")
    before_desire = get_value(state, "desire")

    set_value(state, "satisfaction", 0.9)
    update_state(state)

    after_lack = get_value(state, "lack")
    after_desire = get_value(state, "desire")

    assertion(
        "Satisfaction increases",
        get_value(state, "satisfaction") > 0.0,
    )

    assertion(
        "Lack survives satisfaction",
        after_lack > 0.0,
        f"before={before_lack}, after={after_lack}",
    )

    assertion(
        "Desire survives satisfaction",
        after_desire > 0.0,
        f"before={before_desire}, after={after_desire}",
    )


# ---------------------------------------------------------------------------
# TEST 5 — Memory influences Desire
# ---------------------------------------------------------------------------

def test_memory_influences_desire() -> None:
    banner("TEST 5 — MEMORY INFLUENCES DESIRE")

    state = make_state()

    set_value(state, "need", 0.5)
    set_value(state, "desire", 0.4)
    update_state(state)

    before = get_value(state, "desire")

    register(
        state,
        "memory",
        object_id="resource_B",
        reward=50.0,
        success=True,
        strength=1.0,
    )

    update_state(state)

    after = get_value(state, "desire")

    assertion(
        "Memory changes Desire",
        after != before,
        f"before={before}, after={after}",
    )


# ---------------------------------------------------------------------------
# TEST 6 — Other influences Desire
# ---------------------------------------------------------------------------

def test_other_influences_desire() -> None:
    banner("TEST 6 — OTHER INFLUENCES DESIRE")

    state = make_state()

    set_value(state, "need", 0.3)
    set_value(state, "desire", 0.3)
    update_state(state)

    before = get_value(state, "desire")

    register(
        state,
        "other",
        other_id="agent_1",
        object_id="resource_X",
        observed_desire=0.9,
        observed_success=1.0,
        identification=0.8,
        strength=1.0,
    )

    update_state(state)

    after = get_value(state, "desire")

    assertion(
        "Other changes Desire",
        after != before,
        f"before={before}, after={after}",
    )


# ---------------------------------------------------------------------------
# TEST 7 — Conflict changes action tendency
# ---------------------------------------------------------------------------

def test_conflict_changes_action() -> None:
    banner("TEST 7 — INTERNAL CONFLICT CHANGES ACTION")

    state = make_state()

    set_value(state, "need", 0.7)
    set_value(state, "lack", 0.7)
    set_value(state, "desire", 0.8)
    set_value(state, "satisfaction", 0.2)
    set_value(state, "urgency", 0.7)
    update_state(state)

    register(
        state,
        "conflict",
        approach="resource_X",
        avoidance="resource_X",
        approach_strength=0.9,
        avoidance_strength=0.8,
    )

    register(
        state,
        "action",
        action="approach_resource_X",
        tendency=0.5,
    )

    data = snapshot(state)

    # We deliberately do not require a particular numeric conflict formula.
    # We require the state to expose a non-empty conflict representation.
    conflict = data.get("conflict")

    assertion(
        "Conflict is represented explicitly",
        conflict not in (None, {}, [], 0, 0.0),
        f"conflict={conflict!r}",
    )


# ---------------------------------------------------------------------------
# TEST 8 — Repetition persists after satisfaction
# ---------------------------------------------------------------------------

def test_repetition_persists_after_satisfaction() -> None:
    banner("TEST 8 — REPETITION PERSISTS AFTER SATISFACTION")

    state = make_state()

    set_value(state, "need", 0.8)
    set_value(state, "desire", 0.9)
    set_value(state, "satisfaction", 0.0)
    update_state(state)

    # Register repeated successful relation to the same object.
    for _ in range(5):
        register(
            state,
            "memory",
            object_id="resource_A",
            reward=30.0,
            success=True,
            strength=1.0,
        )

    update_state(state)

    before = get_value(state, "desire")

    # Fulfilment occurs.
    set_value(state, "satisfaction", 1.0)
    update_state(state)

    after = get_value(state, "desire")
    data = snapshot(state)

    repetition = (
        data.get("repetition")
        or data.get("repetition_tendency")
        or data.get("habits")
    )

    assertion(
        "Repetition is explicitly represented",
        repetition not in (None, {}, [], 0, 0.0),
        f"repetition={repetition!r}",
    )

    assertion(
        "Desire remains after satisfaction",
        after > 0.0,
        f"before={before}, after={after}",
    )


# ---------------------------------------------------------------------------
# TEST 9 — Full subject architecture integration
# ---------------------------------------------------------------------------

def test_full_subject_architecture() -> None:
    banner("TEST 9 — FULL SUBJECT ARCHITECTURE INTEGRATION")

    state = make_state()

    # A compact synthetic subject history.
    set_value(state, "need", 0.8)
    set_value(state, "demand", 0.7)
    set_value(state, "lack", 0.75)
    set_value(state, "desire", 0.85)
    set_value(state, "satisfaction", 0.15)
    set_value(state, "urgency", 0.65)
    update_state(state)

    register(
        state,
        "memory",
        object_id="resource_A",
        reward=50.0,
        success=True,
        strength=1.0,
    )

    register(
        state,
        "other",
        other_id="agent_1",
        object_id="resource_B",
        observed_desire=0.9,
        observed_success=1.0,
        identification=0.7,
        strength=1.0,
    )

    register(
        state,
        "conflict",
        approach="resource_A",
        avoidance="resource_B",
        approach_strength=0.8,
        avoidance_strength=0.6,
    )

    update_state(state)

    data = snapshot(state)

    required = {
        "need",
        "demand",
        "lack",
        "desire",
        "satisfaction",
        "urgency",
    }

    missing = required - set(data.keys())

    assertion(
        "All core subject variables coexist",
        not missing,
        f"missing={sorted(missing)}",
    )

    for key in required:
        value = get_value(state, key)
        assertion(
            f"{key} bounded",
            0.0 <= value <= 1.0,
            f"{key}={value}",
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    test_subject_state_exists,
    test_need_changes_state,
    test_need_demand_lack_desire_chain,
    test_desire_can_diverge_from_need,
    test_satisfaction_does_not_erase_lack_or_desire,
    test_memory_influences_desire,
    test_other_influences_desire,
    test_conflict_changes_action,
    test_repetition_persists_after_satisfaction,
    test_full_subject_architecture,
]


def main() -> int:
    banner("STAGE 6.5 — SUBJECT ARCHITECTURE CONTRACT")
    print("This is a PRE-STAGE-7 gate.")
    print("It tests subjectivity, not reward maximization.")

    passed = 0
    failed = []

    for test in TESTS:
        try:
            test()
            passed += 1
        except Exception as exc:
            failed.append((test.__name__, exc))
            print(f"FAIL  {test.__name__}: {exc}")

    banner("FINAL RESULT")
    print(f"Passed: {passed}/{len(TESTS)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print()
        print("SUBJECT ARCHITECTURE GATE: NOT READY")
        print()
        print("Failures:")
        for name, exc in failed:
            print(f"  - {name}: {exc}")
        return 1

    print()
    print("SUBJECT ARCHITECTURE GATE: PASS")
    print("READY FOR STAGE 7–10 EXPERIMENTS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
