import numpy as np

from agents.agent_factory import AgentFactory
from environment.competitive_world import CompetitiveWorld


EPISODES = 20


def run_episode(env, agents, seed):
    observations, infos = env.reset(seed=seed)

    for agent in agents.values():
        agent.reset_episode()

    episode_rewards = {
        name: 0.0
        for name in agents
    }

    collected = {
        name: 0.0
        for name in agents
    }

    steps = 0

    while steps < env.max_steps:

        # =====================================================
        # 1. Every active environment agent must act
        # =====================================================

        actions = {}

        for name in env.agents:

            agent = agents[name]

            actions[name] = int(
                agent.act(
                    observations[name]
                )
            )

        # =====================================================
        # 2. Environment step
        # =====================================================

        (
            next_observations,
            rewards,
            terminated,
            truncated,
            infos,
        ) = env.step(actions)

        # =====================================================
        # 3. Record experience
        # =====================================================

        for name in env.possible_agents:

            if name not in rewards:
                continue

            reward = float(
                rewards[name]
            )

            info = infos.get(
                name,
                {},
            )

            agents[name].record_experience(
                action=actions[name],
                reward=reward,
                info=info,
            )

            episode_rewards[name] += reward

            collected[name] += float(
                info.get(
                    "collected_resource",
                    0.0,
                )
            )

        observations = next_observations

        steps += 1

        # =====================================================
        # 4. End episode
        # =====================================================

        if all(
            truncated.get(
                name,
                False,
            )
            or terminated.get(
                name,
                False,
            )
            for name in env.possible_agents
        ):
            break

    return {
        "rewards": episode_rewards,
        "collected": collected,
        "steps": steps,
    }


def print_episode_result(
    episode,
    result,
    agents,
):
    print()
    print(
        f"Episode {episode:02d}"
    )
    print("-" * 60)

    for name in agents:

        reward = result["rewards"][name]
        collected = result["collected"][name]

        print(
            f"{name:8s} | "
            f"Reward: {reward:7.2f} | "
            f"Collected: {collected:7.2f}"
        )


def print_summary(
    history,
    agents,
):
    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    for name in agents:

        rewards = [
            result["rewards"][name]
            for result in history
        ]

        collected = [
            result["collected"][name]
            for result in history
        ]

        print()
        print(name)
        print("-" * 40)

        print(
            "Total Reward:",
            round(sum(rewards), 2),
        )

        print(
            "Average Reward:",
            round(
                np.mean(rewards),
                2,
            ),
        )

        print(
            "Best Episode:",
            round(
                np.max(rewards),
                2,
            ),
        )

        print(
            "Worst Episode:",
            round(
                np.min(rewards),
                2,
            ),
        )

        print(
            "Average Collected:",
            round(
                np.mean(collected),
                2,
            ),
        )


def main():

    env = CompetitiveWorld(
        render_mode=None
    )

    # ---------------------------------------------------------
    # سه نوع Agent
    # ---------------------------------------------------------

    agent_types = {
        "agent_0": "random",
        "agent_1": "greedy",
        "agent_2": "value",
        "agent_3": "random",
    }

    agents = AgentFactory.create_agents(
        env,
        agent_types,
    )

    value_agent = agents["agent_2"]

    history = []

    print("=" * 70)
    print("MULTI-EPISODE AGENT COMPARISON")
    print("=" * 70)

    print()
    print(
        "Episodes:",
        EPISODES,
    )

    print()
    print("Agent types:")

    for name, agent_type in agent_types.items():
        print(
            f"  {name}: {agent_type}"
        )

    # ---------------------------------------------------------
    # Episodes
    # ---------------------------------------------------------

    for episode in range(
        1,
        EPISODES + 1,
    ):

        result = run_episode(
            env,
            agents,
            seed=episode,
        )

        history.append(result)

        print_episode_result(
            episode,
            result,
            agents,
        )

        # -----------------------------------------------------
        # Memory progression
        # -----------------------------------------------------

        print(
            "Value memory:",
            len(
                value_agent.get_memory()
            ),
        )

        print(
            "Value memory reward:",
            round(
                value_agent.get_memory_reward(),
                3,
            ),
        )

        print(
            "Value risk:",
            round(
                value_agent.get_risk_tolerance(),
                3,
            ),
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print_summary(
        history,
        agents,
    )

    # ---------------------------------------------------------
    # Value Agent learning progression
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("VALUE AGENT LEARNING PROGRESSION")
    print("=" * 70)

    first_half = history[
        : EPISODES // 2
    ]

    second_half = history[
        EPISODES // 2 :
    ]

    first_rewards = [
        x["rewards"]["agent_2"]
        for x in first_half
    ]

    second_rewards = [
        x["rewards"]["agent_2"]
        for x in second_half
    ]

    first_average = np.mean(
        first_rewards
    )

    second_average = np.mean(
        second_rewards
    )

    print(
        "First half average:",
        round(
            first_average,
            2,
        ),
    )

    print(
        "Second half average:",
        round(
            second_average,
            2,
        ),
    )

    print(
        "Improvement:",
        round(
            second_average
            - first_average,
            2,
        ),
    )

    print(
        "Memory size:",
        len(
            value_agent.get_memory()
        ),
    )

    env.close()


if __name__ == "__main__":
    main()
