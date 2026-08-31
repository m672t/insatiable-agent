import sys
from pathlib import Path

# اضافه کردن ریشه پروژه به Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.agent_factory import AgentFactory
from environment.competitive_world import CompetitiveWorld


def run_episode(env, agents, seed=None):
    observations, infos = env.reset(seed=seed)

    for agent in agents.values():
        agent.reset_episode()

    episode_rewards = {
        name: 0.0
        for name in agents
    }

    step = 0

    while step < env.max_steps:

        # =============================================
        # 1. Agent decision
        # =============================================

        actions = {}

        for name, agent in agents.items():

            if name not in env.agents:
                continue

            actions[name] = int(
                agent.act(
                    observations[name]
                )
            )

        # =============================================
        # 2. Environment step
        # =============================================

        (
            next_observations,
            rewards,
            terminated,
            truncated,
            infos,
        ) = env.step(actions)

        # =============================================
        # 3. Record REAL experience
        # =============================================

        for name, agent in agents.items():

            if name not in rewards:
                continue

            reward = float(
                rewards[name]
            )

            info = infos.get(
                name,
                {},
            )

            agent.record_experience(
                action=actions[name],
                reward=reward,
                info=info,
            )

            episode_rewards[name] += reward

        observations = next_observations

        step += 1

    return episode_rewards


def main():

    env = CompetitiveWorld(
        render_mode=None
    )

    agents = AgentFactory.create_agents(
        env,
        {
            "agent_0": "random",
            "agent_1": "greedy",
            "agent_2": "value",
            "agent_3": "greedy",
        },
    )

    value_agent = agents["agent_2"]

    # =================================================
    # Episode 1
    # =================================================

    print()
    print("=" * 60)
    print("EPISODE 1")
    print("=" * 60)

    rewards_1 = run_episode(
        env,
        agents,
        seed=42,
    )

    print()
    print("Rewards:")

    for name, reward in rewards_1.items():
        print(
            f"  {name}: {reward:.2f}"
        )

    print()
    print("Value Agent:")

    print(
        "  Memory:",
        len(value_agent.get_memory()),
    )

    print(
        "  Memory Reward:",
        round(
            value_agent.get_memory_reward(),
            3,
        ),
    )

    print(
        "  Risk:",
        round(
            value_agent.get_risk_tolerance(),
            3,
        ),
    )

    print(
        "  Motivation:",
        value_agent.get_motivation_state(),
    )

    # =================================================
    # Snapshot
    # =================================================

    memory_after_episode_1 = (
        value_agent.get_memory()
    )

    print()
    print("First 5 memories:")

    for memory in (
        memory_after_episode_1[:5]
    ):
        print(" ", memory)

    # =================================================
    # Episode 2
    # =================================================

    print()
    print("=" * 60)
    print("EPISODE 2")
    print("=" * 60)

    rewards_2 = run_episode(
        env,
        agents,
        seed=42,
    )

    print()
    print("Rewards:")

    for name, reward in rewards_2.items():
        print(
            f"  {name}: {reward:.2f}"
        )

    print()
    print("Value Agent:")

    print(
        "  Memory:",
        len(value_agent.get_memory()),
    )

    print(
        "  Memory Reward:",
        round(
            value_agent.get_memory_reward(),
            3,
        ),
    )

    print(
        "  Risk:",
        round(
            value_agent.get_risk_tolerance(),
            3,
        ),
    )

    print(
        "  Motivation:",
        value_agent.get_motivation_state(),
    )

    # =================================================
    # Verification
    # =================================================

    memory_after_episode_2 = (
        value_agent.get_memory()
    )

    print()
    print("=" * 60)
    print("MEMORY VERIFICATION")
    print("=" * 60)

    print(
        "Episode 1 memory:",
        len(memory_after_episode_1),
    )

    print(
        "Episode 2 memory:",
        len(memory_after_episode_2),
    )

    print(
        "Memory preserved:",
        len(memory_after_episode_2)
        >= len(memory_after_episode_1),
    )

    # =================================================
    # Location Memory
    # =================================================

    known_locations = []

    for memory in memory_after_episode_1:

        position = memory.get(
            "position"
        )

        collected = memory.get(
            "collected_resource",
            0.0,
        )

        if (
            position is not None
            and collected > 0
        ):
            known_locations.append(
                position
            )

    known_locations = list(
        dict.fromkeys(
            known_locations
        )
    )

    print()
    print(
        "Known resource locations:",
        len(known_locations),
    )

    for position in known_locations[:10]:

        print(
            f"  {position}:",
            value_agent.get_location_memory_value(
                position
            ),
        )

    env.close()


if __name__ == "__main__":
    main()
