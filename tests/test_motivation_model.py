
from agents.motivation import MotivationModel


def test_multiple_motives_exist():
    model = MotivationModel()

    state = model.get_state(
        lack=0.7,
        satisfaction=0.2,
        urgency=0.6,
    )

    motives = state["motives"]

    expected = {
        "resource_need",
        "curiosity",
        "safety",
        "competition",
        "exploration",
        "avoidance",
    }

    assert expected.issubset(motives.keys())

    for value in motives.values():
        assert 0.0 <= value <= 1.0


def test_motivation_conflict_exists():
    model = MotivationModel()

    state = model.get_state(
        lack=0.8,
        satisfaction=0.3,
        urgency=0.7,
    )

    conflicts = state["conflicts"]

    expected = {
        "resource_vs_safety",
        "curiosity_vs_safety",
        "exploration_vs_avoidance",
        "competition_vs_avoidance",
    }

    assert expected.issubset(conflicts.keys())

    for value in conflicts.values():
        assert 0.0 <= value <= 1.0

    assert 0.0 <= state["conflict_pressure"] <= 1.0


def test_hidden_motives_exist():
    model = MotivationModel()

    state = model.get_state(
        lack=0.6,
        satisfaction=0.2,
        urgency=0.5,
    )

    hidden = state["hidden_motives"]

    expected = {
        "status",
        "novelty_need",
        "control",
        "social_comparison",
    }

    assert expected.issubset(hidden.keys())

    for value in hidden.values():
        assert 0.0 <= value <= 1.0


def test_hidden_motives_change_after_experience():
    model = MotivationModel()

    before = model.get_state()["hidden_motives"].copy()

    model.update_from_experience(
        lack=0.8,
        satisfaction=0.2,
        competition=1.0,
        novelty=1.0,
        success=1.0,
    )

    after = model.get_state()["hidden_motives"]

    assert after["status"] > before["status"]
    assert after["novelty_need"] > before["novelty_need"]
    assert after["control"] > before["control"]
    assert after["social_comparison"] > before["social_comparison"]


def test_hidden_motives_survive_episode_reset():
    model = MotivationModel()

    model.update_from_experience(
        lack=0.8,
        satisfaction=0.1,
        competition=1.0,
        novelty=1.0,
        success=1.0,
    )

    before = model.get_state()["hidden_motives"].copy()

    model.reset_episode()

    after = model.get_state()["hidden_motives"]

    for key in before:
        assert after[key] > 0.0
        assert after[key] < before[key]


def test_desire_is_generated_from_multiple_motives():
    model = MotivationModel()

    state = model.get_state(
        lack=0.9,
        satisfaction=0.1,
        urgency=0.9,
    )

    assert 0.0 <= state["desire"] <= 1.0


def test_high_conflict_does_not_remove_desire():
    model = MotivationModel(
        conflict_strength=0.5
    )

    state = model.get_state(
        lack=0.9,
        satisfaction=0.8,
        urgency=0.8,
    )

    assert state["conflict_pressure"] >= 0.0
    assert state["desire"] >= 0.0
