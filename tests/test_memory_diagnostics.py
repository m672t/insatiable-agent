
import numpy as np

from agents.agent_factory import AgentFactory
from environment.competitive_world import CompetitiveWorld


EPISODES = 20


def run_episode(env, agents, seed):
    observations, infos = env.reset(seed=seed)

    # Reset episode state WITHOUT deleting long-term memory
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

        actions = {}

        # Every active environment agent must act
        for name in env.agents:
            actions[name] = int(
                agents[name].act(
                    observations[name]
                )
            )

        (
            next_observations,
            rewards,
            terminated,
            truncated,
            infos,
        ) = env.step(actions)

        # Record experience
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

        if all(
            truncated.get(name, False)
            or terminated.get(name, False)
            for name in env.possible_agents
        ):
            break

    return {
        "rewards": episode_rewards,
        "collected": collected,
        "steps": steps,
    }


def print_memory_snapshot(value_agent):
    memory = value_agent.get_memory()

    print(
        "Memory size:",
        len(memory),
    )

    print(
        "Memory reward:",
        round(
            value_agent.get_memory_reward(),
            3,
        ),
    )

    print(
        "Risk tolerance:",
        round(
            value_agent.get_risk_tolerance(),
            3,
        ),
    )

    print(
        "Motivation:",
        value_agent.get_motivation_state(),
    )


def main():

    env = CompetitiveWorld(
        render_mode=None
    )

    # ---------------------------------------------------------
    # Agent configuration
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
    print("MEMORY DIAGNOSTICS")
    print("=" * 70)

    print()
    print("Episodes:", EPISODES)
    print("Value agent: agent_2")

    # ---------------------------------------------------------
    # Episodes
    # ---------------------------------------------------------

    for episode in range(
        1,
        EPISODES + 1,
    ):

        print()
        print("=" * 70)
        print(
            f"EPISODE {episode:02d}"
        )
        print("=" * 70)

        result = run_episode(
            env,
            agents,
            seed=episode,
        )

        history.append(result)

        # -----------------------------------------------------
        # Episode results
        # -----------------------------------------------------

        print()

        for name in agents:

            print(
                f"{name:8s} | "
                f"Reward: "
                f"{result['rewards'][name]:7.2f} | "
                f"Collected: "
                f"{result['collected'][name]:7.2f}"
            )

        # -----------------------------------------------------
        # Memory state
        # -----------------------------------------------------

        print()

        print(
            "VALUE AGENT STATE"
        )

        print("-" * 40)

        print_memory_snapshot(
            value_agent
        )

        # -----------------------------------------------------
        # Known locations
        # -----------------------------------------------------

        try:

            known_locations = (
                value_agent.get_known_resource_locations()
            )

            print(
                "Known resource locations:",
                len(known_locations),
            )

        except AttributeError:

            # Compatibility with the current implementation
            try:

                location_memory = (
                    value_agent.location_memory
                )

                print(
                    "Known resource locations:",
                    len(location_memory),
                )

            except AttributeError:

                print(
                    "Known resource locations: unavailable"
                )

    # ---------------------------------------------------------
    # Final analysis
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL MEMORY ANALYSIS")
    print("=" * 70)

    value_rewards = [
        result["rewards"]["agent_2"]
        for result in history
    ]

    value_collected = [
        result["collected"]["agent_2"]
        for result in history
    ]

    # ---------------------------------------------------------
    # First vs second half
    # ---------------------------------------------------------

    midpoint = EPISODES // 2

    first_half = value_rewards[:midpoint]
    second_half = value_rewards[midpoint:]

    first_average = float(
        np.mean(first_half)
    )

    second_average = float(
        np.mean(second_half)
    )

    improvement = (
        second_average
        - first_average
    )

    print()
    print(
        "Value Agent Reward"
    )

    print("-" * 40)

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
            improvement,
            2,
        ),
    )

    print()
    print(
        "Total reward:",
        round(
            sum(value_rewards),
            2,
        ),
    )

    print(
        "Average reward:",
        round(
            np.mean(value_rewards),
            2,
        ),
    )

    print(
        "Best episode:",
        round(
            np.max(value_rewards),
            2,
        ),
    )

    print(
        "Worst episode:",
        round(
            np.min(value_rewards),
            2,
        ),
    )

    print(
        "Average collected:",
        round(
            np.mean(value_collected),
            2,
        ),
    )

    # ---------------------------------------------------------
    # Final memory
    # ---------------------------------------------------------

    print()
    print(
        "FINAL MEMORY STATE"
    )

    print("-" * 40)

    print_memory_snapshot(
        value_agent
    )

    # ---------------------------------------------------------
    # Memory verification
    # ---------------------------------------------------------

    expected_minimum = EPISODES

    actual_memory = len(
        value_agent.get_memory()
    )

    print()
    print(
        "MEMORY VERIFICATION"
    )

    print("-" * 40)

    print(
        "Expected memory >=:",
        expected_minimum,
    )

    print(
        "Actual memory:",
        actual_memory,
    )

    if actual_memory >= expected_minimum:
        print(
            "Memory preserved: PASS"
        )
    else:
        print(
            "Memory preserved: FAIL"
        )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print()
    print("=" * 70)

    if actual_memory >= expected_minimum:
        print(
            "PASS - Memory persists across episodes."
        )
    else:
        print(
            "FAIL - Memory was not preserved."
        )

    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
