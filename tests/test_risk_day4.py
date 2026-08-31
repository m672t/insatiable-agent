from environment.competitive_world import CompetitiveWorld
from agents.value_seeking_agent import ValueSeekingAgent


def print_risk_table(
    agent,
    resources,
    title,
):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    for position, value in resources.items():

        distance = agent.get_distance(
            position
        )

        competition = (
            agent.get_resource_competition(
                position
            )
        )

        risk = agent.get_resource_risk(
            position
        )

        score = agent.get_resource_score(
            position,
            value,
        )

        print(
            f"Resource={position}, "
            f"Value={value}, "
            f"Distance={distance:.2f}, "
            f"Competition={competition:.3f}, "
            f"Risk={risk:.3f}, "
            f"Score={score:.4f}"
        )


def main():

    print("=" * 60)
    print("DAY 4 - RISK TEST")
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
    # Controlled world
    # ---------------------------------------------------------

    # Agent 0 = (1, 1)
    #
    # Near resource:
    #   (2, 1)
    #
    # Far resource:
    #   (6, 1)

    resources = {
        (2, 1): 15,
        (6, 1): 50,
    }

    env.resources = resources.copy()

    # سایر Agentها را طوری قرار می‌دهیم
    # که Resource دور Competition بیشتری داشته باشد.

    env.positions["agent_1"] = (
        env.positions["agent_0"]
        .copy()
    )

    env.positions["agent_2"] = (
        env.positions["agent_0"]
        .copy()
    )

    env.positions["agent_3"] = (
        env.positions["agent_0"]
        .copy()
    )

    # ---------------------------------------------------------
    # BASELINE
    # ---------------------------------------------------------

    agent.motivation_state = {
        "lack": 0.0,
        "desire": 0.0,
        "satisfaction": 0.0,
        "urgency": 0.0,
    }

    print()
    print(
        "BASELINE RISK TOLERANCE:",
        agent.get_risk_tolerance()
    )

    print_risk_table(
        agent,
        resources,
        "BASELINE",
    )

    # ---------------------------------------------------------
    # HIGH LACK
    # ---------------------------------------------------------

    agent.motivation_state = {
        "lack": 1.0,
        "desire": 0.0,
        "satisfaction": 0.0,
        "urgency": 0.0,
    }

    high_lack_tolerance = (
        agent.get_risk_tolerance()
    )

    print()
    print(
        "HIGH LACK RISK TOLERANCE:",
        high_lack_tolerance
    )

    print_risk_table(
        agent,
        resources,
        "HIGH LACK",
    )

    # ---------------------------------------------------------
    # HIGH SATISFACTION
    # ---------------------------------------------------------

    agent.motivation_state = {
        "lack": 0.0,
        "desire": 0.0,
        "satisfaction": 1.0,
        "urgency": 0.0,
    }

    high_satisfaction_tolerance = (
        agent.get_risk_tolerance()
    )

    print()
    print(
        "HIGH SATISFACTION "
        "RISK TOLERANCE:",
        high_satisfaction_tolerance
    )

    print_risk_table(
        agent,
        resources,
        "HIGH SATISFACTION",
    )

    # ---------------------------------------------------------
    # Explicit Risk comparison
    # ---------------------------------------------------------

    near = (2, 1)
    far = (6, 1)

    near_risk = agent.get_resource_risk(
        near
    )

    far_risk = agent.get_resource_risk(
        far
    )

    print()
    print("=" * 60)
    print("RISK COMPARISON")
    print("=" * 60)

    print(
        f"Near Resource Risk: "
        f"{near_risk:.4f}"
    )

    print(
        f"Far Resource Risk: "
        f"{far_risk:.4f}"
    )

    # ---------------------------------------------------------
    # Tolerance assertions
    # ---------------------------------------------------------

    assert (
        high_lack_tolerance
        > 0.50
    )

    assert (
        high_satisfaction_tolerance
        < 0.50
    )

    assert (
        far_risk
        > near_risk
    )

    print()
    print("=" * 60)
    print("RISK TOLERANCE ASSERTIONS: PASS")
    print("=" * 60)

    # ---------------------------------------------------------
    # Motivation-driven risk test
    # ---------------------------------------------------------

    agent.motivation_state = {
        "lack": 0.0,
        "desire": 1.0,
        "satisfaction": 0.0,
        "urgency": 0.0,
    }

    baseline_target = agent.select_target()

    agent.motivation_state = {
        "lack": 1.0,
        "desire": 1.0,
        "satisfaction": 0.0,
        "urgency": 0.0,
    }

    high_lack_target = agent.select_target()

    agent.motivation_state = {
        "lack": 0.0,
        "desire": 1.0,
        "satisfaction": 1.0,
        "urgency": 0.0,
    }

    high_satisfaction_target = (
        agent.select_target()
    )

    print()
    print("=" * 60)
    print("MOTIVATION-DRIVEN TARGET TEST")
    print("=" * 60)

    print(
        "Baseline target:",
        baseline_target
    )

    print(
        "High Lack target:",
        high_lack_target
    )

    print(
        "High Satisfaction target:",
        high_satisfaction_target
    )

    print()
    print("=" * 60)
    print("DAY 4 TEST COMPLETE")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
