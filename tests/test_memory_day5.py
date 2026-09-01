
from environment.competitive_world import CompetitiveWorld
from agents.value_seeking_agent import ValueSeekingAgent


def main():

    print("=" * 60)
    print("DAY 5 - EXPERIENCE MEMORY TEST")
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
    # Initial memory
    # ---------------------------------------------------------

    print()
    print("INITIAL MEMORY")

    print(
        "Experience count:",
        len(agent.experience_memory)
    )

    print(
        "Episode count:",
        len(agent.completed_episode_memory)
    )

    assert (
        len(agent.experience_memory) == 0
    )

    # ---------------------------------------------------------
    # Add experiences
    # ---------------------------------------------------------

    agent.remember_experience(
        resource_location=(2, 1),
        resource_value=15,
        distance=1,
        action="move",
        reward=15,
        competition=0.5,
        outcome="collected",
    )

    agent.remember_experience(
        resource_location=(6, 1),
        resource_value=50,
        distance=5,
        action="move",
        reward=50,
        competition=1.5,
        outcome="collected",
    )

    agent.remember_experience(
        resource_location=(3, 1),
        resource_value=5,
        distance=2,
        action="move",
        reward=0,
        competition=1.0,
        outcome="failed",
    )

    print()
    print("AFTER EXPERIENCE RECORDING")

    print(
        "Experience count:",
        len(agent.experience_memory)
    )

    for experience in (
        agent.experience_memory
    ):
        print(experience)

    assert (
        len(agent.experience_memory) == 3
    )

    # ---------------------------------------------------------
    # Memory adjustment
    # ---------------------------------------------------------

    adjustment_15 = (
        agent.get_memory_adjustment(
            position=(2, 1),
            value=15,
        )
    )

    adjustment_50 = (
        agent.get_memory_adjustment(
            position=(6, 1),
            value=50,
        )
    )

    print()
    print("MEMORY ADJUSTMENTS")

    print(
        "Value 15 adjustment:",
        f"{adjustment_15:.4f}"
    )

    print(
        "Value 50 adjustment:",
        f"{adjustment_50:.4f}"
    )

    assert (
        adjustment_15 > 1.0
    )

    assert (
        adjustment_50 > 1.0
    )

    # ---------------------------------------------------------
    # Episode summary
    # ---------------------------------------------------------

    summary = (
        agent.summarize_episode_memory()
    )

    print()
    print("EPISODE SUMMARY")

    for key, value in summary.items():

        print(
            f"{key}: {value}"
        )

    assert (
        summary["steps"] == 3
    )

    assert (
        summary["total_reward"] == 65.0
    )

    assert (
        summary["successful_actions"] == 2
    )

    assert (
        summary["success_rate"]
        > 0.0
    )

    assert (
        len(
            agent.current_episode_experiences
        ) == 0
    )

    assert (
        len(
            agent.completed_episode_memory
        ) == 1
    )

    # ---------------------------------------------------------
    # Memory limit
    # ---------------------------------------------------------

    for i in range(250):

        agent.remember_experience(
            resource_location=(i, 0),
            resource_value=5,
            distance=i,
            action="move",
            reward=0,
            competition=0,
            outcome="failed",
        )

    print()
    print("MEMORY LIMIT")

    print(
        "Configured limit:",
        agent.memory_max_size
    )

    print(
        "Actual memory size:",
        len(agent.experience_memory)
    )

    assert (
        len(agent.experience_memory)
        <= agent.memory_max_size
    )

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("DAY 5 MEMORY ASSERTIONS: PASS")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
