import numpy as np

from environment.competitive_world import CompetitiveWorld

from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.value_seeking_agent import ValueSeekingAgent


NUM_EPISODES = 100


def create_agents(env):

    return {
        "agent_0": RandomAgent(
            env,
            "agent_0",
        ),

        "agent_1": GreedyAgent(
            env,
            "agent_1",
        ),

        "agent_2": ValueSeekingAgent(
            env,
            "agent_2",
        ),

        "agent_3": ValueSeekingAgent(
            env,
            "agent_3",
        ),
    }


def run_episode(seed):

    env = CompetitiveWorld(
        grid_size=20,
        num_agents=4,
        num_resources=30,
        render_mode=None,
    )

    observations, infos = env.reset(
        seed=seed
    )

    agents = create_agents(env)

    total_rewards = {
        agent: 0.0
        for agent in env.agents
    }

    collected_resources = {
        agent: 0
        for agent in env.agents
    }

    while True:

        actions = {}

        for agent_name in env.agents:

            actions[agent_name] = (
                agents[agent_name].act(
                    observations[agent_name]
                )
            )

        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = env.step(actions)

        for agent_name in env.agents:

            total_rewards[agent_name] += (
                rewards[agent_name]
            )

            if rewards[agent_name] > 0:

                collected_resources[
                    agent_name
                ] += 1

        if all(
            terminations[agent]
            or truncations[agent]
            for agent in env.agents
        ):
            break

    env.close()

    return (
        total_rewards,
        collected_resources,
    )


def main():

    # نتایج هر سیاست
    policy_rewards = {
        "Random": [],
        "Greedy": [],
        "Value-Seeking": [],
    }

    policy_resources = {
        "Random": [],
        "Greedy": [],
        "Value-Seeking": [],
    }

    for episode in range(
        NUM_EPISODES
    ):

        print(
            f"\rEpisode "
            f"{episode + 1}/"
            f"{NUM_EPISODES}",
            end="",
        )

        (
            rewards,
            resources,
        ) = run_episode(
            seed=episode
        )

        # Agent 0 = Random
        policy_rewards["Random"].append(
            rewards["agent_0"]
        )

        policy_resources["Random"].append(
            resources["agent_0"]
        )

        # Agent 1 = Greedy
        policy_rewards["Greedy"].append(
            rewards["agent_1"]
        )

        policy_resources["Greedy"].append(
            resources["agent_1"]
        )

        # Agent 2 و 3 = Value-Seeking
        value_reward = (
            rewards["agent_2"]
            +
            rewards["agent_3"]
        )

        value_resources = (
            resources["agent_2"]
            +
            resources["agent_3"]
        )

        policy_rewards[
            "Value-Seeking"
        ].append(
            value_reward / 2
        )

        policy_resources[
            "Value-Seeking"
        ].append(
            value_resources / 2
        )

    print()
    print()

    print("=" * 75)
    print("COMPETITIVE MULTI-AGENT BENCHMARK")
    print("=" * 75)

    print()

    print(
        f"{'Policy':<18}"
        f"{'Mean Reward':>15}"
        f"{'Std Reward':>15}"
        f"{'Mean Resources':>18}"
    )

    print("-" * 75)

    for policy in [
        "Random",
        "Greedy",
        "Value-Seeking",
    ]:

        rewards = np.array(
            policy_rewards[policy]
        )

        resources = np.array(
            policy_resources[policy]
        )

        print(
            f"{policy:<18}"
            f"{np.mean(rewards):>15.2f}"
            f"{np.std(rewards):>15.2f}"
            f"{np.mean(resources):>18.2f}"
        )

    print()
    print("-" * 75)

    best_policy = max(
        policy_rewards,
        key=lambda policy:
        np.mean(
            policy_rewards[policy]
        ),
    )

    print(
        "Best policy by mean reward: "
        f"{best_policy}"
    )

    print("=" * 75)
    print()


if __name__ == "__main__":
    main()