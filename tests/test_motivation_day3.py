
from environment.competitive_world import CompetitiveWorld
from agents.value_seeking_agent import ValueSeekingAgent


RESOURCES = {
    (2, 1): 5,
    (3, 1): 15,
    (6, 1): 50,
}


def print_scores(agent, resources, label):
    print()
    print(label)

    scores = {}

    for position, value in resources.items():

        score = float(
            agent.get_resource_score(
                position,
                value,
            )
        )

        scores[position] = score

        print(
            f"Resource={position}, "
            f"Value={value}, "
            f"Distance={agent.get_distance(position):.2f}, "
            f"Score={score:.4f}"
        )

    return scores


def print_target(agent, label):
    target = agent.select_target()

    print(
        f"{label}: {target}"
    )

    return target


def main():

    print("=" * 60)
    print("DAY 3 - MOTIVATION TEST")
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
    # Controlled resources
    # ---------------------------------------------------------

    env.resources = RESOURCES.copy()

    # ---------------------------------------------------------
    # Initial motivation
    # ---------------------------------------------------------

    print()
    print("INITIAL MOTIVATION")

    agent.motivation_state = {
        "lack": 0.0,
        "desire": 0.0,
        "satisfaction": 0.0,
        "urgency": 0.0,
    }

    print(
        agent.get_motivation_state()
    )

    baseline_scores = print_scores(
        agent,
        env.resources,
        "BASELINE SCORES",
    )

    baseline_target = print_target(
        agent,
        "BASELINE TARGET",
    )

    # ---------------------------------------------------------
    # Test 1: Desire
    # ---------------------------------------------------------

    agent.motivation_state = {
        "lack": 0.5,
        "desire": 1.0,
        "satisfaction": 0.0,
        "urgency": 0.0,
    }

    desire_scores = print_scores(
        agent,
        env.resources,
        "HIGH DESIRE SCORES",
    )

    desire_target = print_target(
        agent,
        "HIGH DESIRE TARGET",
    )

    print()
    print("DESIRE EFFECT")

    print(
        f"Low-value score: "
        f"{baseline_scores[(2, 1)]:.4f} "
        f"-> "
        f"{desire_scores[(2, 1)]:.4f}"
    )

    print(
        f"High-value score: "
        f"{baseline_scores[(6, 1)]:.4f} "
        f"-> "
        f"{desire_scores[(6, 1)]:.4f}"
    )

    # ---------------------------------------------------------
    # Test 2: Urgency
    # ---------------------------------------------------------

    agent.motivation_state = {
        "lack": 0.0,
        "desire": 0.0,
        "satisfaction": 0.0,
        "urgency": 1.0,
    }

    urgency_scores = print_scores(
        agent,
        env.resources,
        "HIGH URGENCY SCORES",
    )

    urgency_target = print_target(
        agent,
        "HIGH URGENCY TARGET",
    )

    print()
    print("URGENCY EFFECT")

    print(
        f"Near resource score: "
        f"{urgency_scores[(2, 1)]:.4f}"
    )

    print(
        f"Far resource score: "
        f"{urgency_scores[(6, 1)]:.4f}"
    )

    # ---------------------------------------------------------
    # Test 3: Satisfaction
    # ---------------------------------------------------------

    agent.motivation_state = {
        "lack": 0.0,
        "desire": 0.0,
        "satisfaction": 1.0,
        "urgency": 0.0,
    }

    satisfaction_scores = print_scores(
        agent,
        env.resources,
        "HIGH SATISFACTION SCORES",
    )

    satisfaction_target = print_target(
        agent,
        "HIGH SATISFACTION TARGET",
    )

    print()
    print("SATISFACTION EFFECT")

    print(
        f"Near resource score: "
        f"{satisfaction_scores[(2, 1)]:.4f}"
    )

    print(
        f"Far resource score: "
        f"{satisfaction_scores[(6, 1)]:.4f}"
    )

    # ---------------------------------------------------------
    # Test 4: Lack
    # ---------------------------------------------------------

    agent.motivation_state = {
        "lack": 1.0,
        "desire": 0.0,
        "satisfaction": 0.0,
        "urgency": 0.0,
    }

    lack_scores = print_scores(
        agent,
        env.resources,
        "HIGH LACK SCORES",
    )

    lack_target = print_target(
        agent,
        "HIGH LACK TARGET",
    )

    print()
    print("LACK EFFECT")

    print(
        f"Baseline high-value score: "
        f"{baseline_scores[(6, 1)]:.4f}"
    )

    print(
        f"High-lack high-value score: "
        f"{lack_scores[(6, 1)]:.4f}"
    )

    # ---------------------------------------------------------
    # Motivation update
    # ---------------------------------------------------------

    print()
    print("MOTIVATION UPDATE TEST")

    agent.motivation_state = {
        "lack": 0.0,
        "desire": 0.0,
        "satisfaction": 0.0,
        "urgency": 0.0,
    }

    for step in range(10):

        agent.update_motivation(
            reward=0.0,
            collected_resource=0.0,
        )

        print(
            f"Step {step + 1}: "
            f"{agent.get_motivation_state()}"
        )

    # ---------------------------------------------------------
    # Successful collection
    # ---------------------------------------------------------

    agent.update_motivation(
        reward=50.0,
        collected_resource=50.0,
    )

    print()
    print("AFTER 50-VALUE COLLECTION")

    print(
        agent.get_motivation_state()
    )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("DAY 3 BEHAVIORAL SUMMARY")
    print("=" * 60)

    print(
        f"Baseline target:      {baseline_target}"
    )

    print(
        f"High Desire target:   {desire_target}"
    )

    print(
        f"High Urgency target:  {urgency_target}"
    )

    print(
        f"High Satisfaction:    {satisfaction_target}"
    )

    print(
        f"High Lack target:     {lack_target}"
    )

    print()
    print("=" * 60)
    print("DAY 3 TEST COMPLETE")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
