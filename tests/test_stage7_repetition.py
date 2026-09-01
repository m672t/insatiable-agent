from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    print("=" * 72)
    print("STAGE 7 — REPETITION / COMPULSION EXPERIMENT")
    print("=" * 72)
    print()
    print("This experiment tests whether repetition persists")
    print("after satisfaction and continues to influence desire.")
    print()

    from agents.subject_state import SubjectState

    print("PASS — SubjectState imported")
    print()

    object_id = "resource_A"

    # ============================================================
    # TEST 1 — INITIAL SUBJECT
    # ============================================================
    print("-" * 72)
    print("TEST 1 — INITIAL SUBJECT STATE")
    print("-" * 72)

    subject = SubjectState(
        need=0.8,
        demand=0.7,
        lack=0.75,
        desire=0.85,
        satisfaction=0.0,
        urgency=0.6,
    )

    state = subject.get_state()

    for key in (
        "need",
        "demand",
        "lack",
        "desire",
        "satisfaction",
        "urgency",
    ):
        assert key in state

    print(f"Need:         {state['need']}")
    print(f"Demand:       {state['demand']}")
    print(f"Lack:         {state['lack']}")
    print(f"Desire:       {state['desire']}")
    print(f"Satisfaction: {state['satisfaction']}")
    print(f"Urgency:      {state['urgency']}")

    print()
    print("PASS — Core subject state exists")
    print()

    # ============================================================
    # TEST 2 — SATISFACTION
    # ============================================================
    print("-" * 72)
    print("TEST 2 — SATISFACTION WITHOUT ERASURE")
    print("-" * 72)

    subject.set_satisfaction(0.9)
    subject.update()

    state = subject.get_state()

    satisfaction_after_first = state["satisfaction"]
    lack_after_first = state["lack"]
    desire_after_first = state["desire"]

    print(f"Satisfaction: {satisfaction_after_first}")
    print(f"Lack:         {lack_after_first}")
    print(f"Desire:       {desire_after_first}")

    assert satisfaction_after_first > 0
    assert lack_after_first > 0
    assert desire_after_first > 0

    print()
    print("PASS — Satisfaction occurs")
    print("PASS — Lack survives satisfaction")
    print("PASS — Desire survives satisfaction")
    print()

    # ============================================================
    # TEST 3 — FIRST REPETITION
    # ============================================================
    print("-" * 72)
    print("TEST 3 — FIRST REPETITION")
    print("-" * 72)

    subject.register_repetition(
        object_id=object_id,
        strength=1.0,
    )

    state = subject.get_state()

    first_rep = state["repetition"].get(object_id)

    print(f"Repetition: {state['repetition']}")

    assert first_rep is not None
    assert first_rep["count"] == 1
    assert first_rep["strength"] > 0

    print()
    print("PASS — First repetition recorded")
    print()

    # ============================================================
    # TEST 4 — REPEATED ENCOUNTERS
    # ============================================================
    print("-" * 72)
    print("TEST 4 — REPEATED ENCOUNTERS")
    print("-" * 72)

    counts = []

    # We already have repetition #1.
    for i in range(4):
        subject.register_repetition(
            object_id=object_id,
            strength=1.0,
        )

        state = subject.get_state()
        current_count = state["repetition"][object_id]["count"]
        counts.append(current_count)

        print(f"Repetition #{current_count}")

    final_before_satisfaction = subject.get_state()
    repetition_before_satisfaction = (
        final_before_satisfaction["repetition"][object_id]
    )

    print()
    print(f"Counts observed: {counts}")
    print(f"Final repetition: {repetition_before_satisfaction}")

    assert counts == [2, 3, 4, 5]
    assert repetition_before_satisfaction["count"] == 5

    print()
    print("PASS — Repetition count accumulates 1 → 5")
    print()

    # ============================================================
    # TEST 5 — REPETITION INFLUENCES DESIRE
    # ============================================================
    print("-" * 72)
    print("TEST 5 — REPETITION → DESIRE")
    print("-" * 72)

    # Use a clean comparison so satisfaction does not confound
    # the repetition effect.
    baseline = SubjectState(
        need=0.3,
        demand=0.3,
        lack=0.3,
        desire=0.2,
        satisfaction=0.0,
        urgency=0.2,
    )

    repeated = SubjectState(
        need=0.3,
        demand=0.3,
        lack=0.3,
        desire=0.2,
        satisfaction=0.0,
        urgency=0.2,
    )

    baseline.update()

    repeated.register_repetition(
        object_id=object_id,
        strength=1.0,
    )
    repeated.update()

    baseline_desire = baseline.get_desire()
    repeated_desire = repeated.get_desire()

    print(f"Baseline desire:  {baseline_desire:.6f}")
    print(f"Repeated desire: {repeated_desire:.6f}")

    assert repeated_desire > baseline_desire

    print()
    print("PASS — Repetition increases Desire pressure")
    print()

    # ============================================================
    # TEST 6 — SATISFACTION AFTER REPETITION
    # ============================================================
    print("-" * 72)
    print("TEST 6 — SATISFACTION AFTER REPETITION")
    print("-" * 72)

    state_before_satisfaction = subject.get_state()

    count_before = state_before_satisfaction["repetition"][
        object_id
    ]["count"]

    desire_before_satisfaction = state_before_satisfaction["desire"]
    lack_before_satisfaction = state_before_satisfaction["lack"]

    print(f"Count before satisfaction:   {count_before}")
    print(f"Desire before satisfaction:  {desire_before_satisfaction}")
    print(f"Lack before satisfaction:    {lack_before_satisfaction}")

    assert count_before == 5

    # Satisfaction happens AFTER the repeated relation exists.
    subject.set_satisfaction(1.0)
    subject.update()

    state_after_satisfaction = subject.get_state()

    count_after = state_after_satisfaction["repetition"][
        object_id
    ]["count"]

    desire_after_satisfaction = state_after_satisfaction["desire"]
    lack_after_satisfaction = state_after_satisfaction["lack"]

    print()
    print(f"Count after satisfaction:    {count_after}")
    print(f"Desire after satisfaction:   {desire_after_satisfaction}")
    print(f"Lack after satisfaction:     {lack_after_satisfaction}")
    print(f"Satisfaction:                {state_after_satisfaction['satisfaction']}")

    assert state_after_satisfaction["satisfaction"] > 0

    # This is the critical persistence condition.
    assert count_after == count_before

    print()
    print("PASS — Satisfaction does not erase repetition")
    print()

    # ============================================================
    # TEST 7 — DESIRE PERSISTS
    # ============================================================
    print("-" * 72)
    print("TEST 7 — DESIRE PERSISTS AFTER SATISFACTION")
    print("-" * 72)

    assert desire_after_satisfaction > 0
    assert lack_after_satisfaction > 0

    print(f"Desire after satisfaction: {desire_after_satisfaction}")
    print(f"Lack after satisfaction:   {lack_after_satisfaction}")

    print()
    print("PASS — Desire remains after satisfaction")
    print("PASS — Lack remains after satisfaction")
    print()

    # ============================================================
    # TEST 8 — REPETITION STILL EXISTS IN COMPLETE STATE
    # ============================================================
    print("-" * 72)
    print("TEST 8 — COMPLETE SUBJECT STATE")
    print("-" * 72)

    final_state = subject.get_state()

    required_structures = (
        "memory_influence",
        "other_influence",
        "conflict",
        "repetition",
        "action_tendencies",
    )

    for key in required_structures:
        assert key in final_state

    assert object_id in final_state["repetition"]
    assert final_state["repetition"][object_id]["count"] == 5

    print("Core variables:")
    print(f"  need:         {final_state['need']}")
    print(f"  demand:       {final_state['demand']}")
    print(f"  lack:         {final_state['lack']}")
    print(f"  desire:       {final_state['desire']}")
    print(f"  satisfaction: {final_state['satisfaction']}")
    print(f"  urgency:      {final_state['urgency']}")

    print()
    print("Repetition:")
    print(f"  {final_state['repetition']}")

    print()
    print("PASS — Repetition remains part of complete subject state")
    print()

    # ============================================================
    # FINAL GATE
    # ============================================================
    print("=" * 72)
    print("STAGE 7 — REPETITION / COMPULSION GATE")
    print("=" * 72)
    print()
    print("PASS — Subject state exists")
    print("PASS — Satisfaction does not erase Lack")
    print("PASS — Satisfaction does not erase Desire")
    print("PASS — Repetition is explicitly represented")
    print("PASS — Repeated encounters accumulate")
    print("PASS — Repetition influences Desire")
    print("PASS — Repetition survives Satisfaction")
    print("PASS — Desire survives Satisfaction")
    print("PASS — Complete subject state remains coherent")
    print()
    print("STAGE 7 GATE: PASS")
    print()
    print("=" * 72)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
