from environment.competitive_world import CompetitiveWorld
from agents.value_seeking_agent import ValueSeekingAgent


def print_scores(agent, resources, title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    for position, value in resources.items():
        score = agent.get_resource_score(
            position,
            value,
        )

        distance = agent.get_distance(position)

        print(
            f"Resource={position}, "
            f"Value={value}, "
            f"Distance={distance:.2f}, "
            f"Score={score:.4f}"
        )


def main():

    print("=" * 60)
    print("DAY 3 - MOTIVATION SENSITIVITY TEST")
    print("=" * 60)

    env = CompetitiveWorld(
        render_mode=None
    )

    env.reset(seed=42)

    agent = ValueSeekingAgent(
        env=env,
        agent_name="agent_0",
    )

    # ---------------------------------------------------------
    # کنترل منابع
    # ---------------------------------------------------------

    # همه Resourceها ارزش یکسان دارند.
    # بنابراین اثر Distance و Motivation
    # راحت‌تر قابل مشاهده است.

    resources = {
        (2, 1): 15,
        (3, 1): 15,
        (6, 1): 15,
    }

    env.resources = resources.copy()

    # ---------------------------------------------------------
    # BASELINE
    # ---------------------------------------------------------

    agent.motivation_state = {
        "lack": 0.0,
        "desire": 0.0,
        "satisfaction": 0.0,
        "urgency": 0.0,
    }

    print_scores(
        agent,
        resources,
        "BASELINE",
    )

    baseline_scores = {
        position: agent.get_resource_score(
            position,
            value,
        )
        for position, value in resources.items()
    }

    # ---------------------------------------------------------
    # HIGH DESIRE
    # ---------------------------------------------------------

    agent.motivation_state = {
        "lack": 0.0,
        "desire": 1.0,
        "satisfaction": 0.0,
        "urgency": 0.0,
    }

    print_scores(
        agent,
        resources,
        "HIGH DESIRE",
    )

    desire_scores = {
        position: agent.get_resource_score(
            position,
            value,
        )
        for position, value in resources.items()
    }

    # ---------------------------------------------------------
    # HIGH URGENCY
    # ---------------------------------------------------------

    agent.motivation_state = {
        "lack": 0.0,
        "desire": 0.0,
        "satisfaction": 0.0,
        "urgency": 1.0,
    }

    print_scores(
        agent,
        resources,
        "HIGH URGENCY",
    )

    urgency_scores = {
        position: agent.get_resource_score(
            position,
            value,
        )
        for position, value in resources.items()
    }

    # ---------------------------------------------------------
    # HIGH SATISFACTION
    # ---------------------------------------------------------

    agent.motivation_state = {
        "lack": 0.0,
        "desire": 0.0,
        "satisfaction": 1.0,
        "urgency": 0.0,
    }

    print_scores(
        agent,
        resources,
        "HIGH SATISFACTION",
    )

    satisfaction_scores = {
        position: agent.get_resource_score(
            position,
            value,
        )
        for position, value in resources.items()
    }

    # ---------------------------------------------------------
    # HIGH LACK
    # ---------------------------------------------------------

    agent.motivation_state = {
        "lack": 1.0,
        "desire": 0.0,
        "satisfaction": 0.0,
        "urgency": 0.0,
    }

    print_scores(
        agent,
        resources,
        "HIGH LACK",
    )

    lack_scores = {
        position: agent.get_resource_score(
            position,
            value,
        )
        for position, value in resources.items()
    }

    # ---------------------------------------------------------
    # TARGETS
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("TARGET COMPARISON")
    print("=" * 60)

    motivation_cases = {
        "Baseline": {
            "lack": 0.0,
            "desire": 0.0,
            "satisfaction": 0.0,
            "urgency": 0.0,
        },

        "High Desire": {
            "lack": 0.0,
            "desire": 1.0,
            "satisfaction": 0.0,
            "urgency": 0.0,
        },

        "High Urgency": {
            "lack": 0.0,
            "desire": 0.0,
            "satisfaction": 0.0,
            "urgency": 1.0,
        },

        "High Satisfaction": {
            "lack": 0.0,
            "desire": 0.0,
            "satisfaction": 1.0,
            "urgency": 0.0,
        },

        "High Lack": {
            "lack": 1.0,
            "desire": 0.0,
            "satisfaction": 0.0,
            "urgency": 0.0,
        },
    }

    targets = {}

    for name, motivation in motivation_cases.items():

        agent.motivation_state = motivation

        targets[name] = agent.select_target()

        print(
            f"{name:20s}: "
            f"{targets[name]}"
        )

    # ---------------------------------------------------------
    # EFFECT SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("MOTIVATION EFFECT SUMMARY")
    print("=" * 60)

    near = (2, 1)
    middle = (3, 1)
    far = (6, 1)

    print()
    print("Urgency:")
    print(
        f"Near : "
        f"{baseline_scores[near]:.4f} -> "
        f"{urgency_scores[near]:.4f}"
    )
    print(
        f"Far  : "
        f"{baseline_scores[far]:.4f} -> "
        f"{urgency_scores[far]:.4f}"
    )

    print()
    print("Satisfaction:")
    print(
        f"Near : "
        f"{baseline_scores[near]:.4f} -> "
        f"{satisfaction_scores[near]:.4f}"
    )
    print(
        f"Far  : "
        f"{baseline_scores[far]:.4f} -> "
        f"{satisfaction_scores[far]:.4f}"
    )

    print()
    print("Desire:")
    print(
        f"Near : "
        f"{baseline_scores[near]:.4f} -> "
        f"{desire_scores[near]:.4f}"
    )
    print(
        f"Far  : "
        f"{baseline_scores[far]:.4f} -> "
        f"{desire_scores[far]:.4f}"
    )

    print()
    print("Lack:")
    print(
        f"Near : "
        f"{baseline_scores[near]:.4f} -> "
        f"{lack_scores[near]:.4f}"
    )
    print(
        f"Far  : "
        f"{baseline_scores[far]:.4f} -> "
        f"{lack_scores[far]:.4f}"
    )

    # ---------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------

    # Desire باید Score را افزایش دهد.
    assert desire_scores[far] > baseline_scores[far]

    # Lack باید Score را افزایش دهد.
    assert lack_scores[far] > baseline_scores[far]

    # Satisfaction باید Score را کاهش دهد.
    assert satisfaction_scores[far] < baseline_scores[far]

    # Urgency باید روی Resource نزدیک اثر مثبت داشته باشد.
    assert urgency_scores[near] > baseline_scores[near]

    # Urgency نباید Resource نزدیک را نسبت به baseline
    # بدتر کند.
    assert urgency_scores[near] >= baseline_scores[near]

    print()
    print("=" * 60)
    print("ASSERTIONS: PASS")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
