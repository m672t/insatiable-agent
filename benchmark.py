import numpy as np

from environment.competitive_world import CompetitiveWorld

from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.value_seeking_agent import ValueSeekingAgent


NUM_EPISODES = 100


def run_episode(agent_type, seed):

    env = CompetitiveWorld(
        grid_size=20,
        num_agents=1,
        num_resources=15,
        render_mode=None,
    )

    observations, infos = env.reset(seed=seed)

    agent_name = "agent_0"

    if agent_type == "random":

        agent = RandomAgent(
            env,
            agent_name,
        )

    elif agent_type == "greedy":

        agent = GreedyAgent(
            env,
            agent_name,
        )

    elif agent_type == "value":

        agent = ValueSeekingAgent(
            env,
            agent_name,
        )

    else:
        raise ValueError(
            f"Unknown agent type: {agent_type}"
        )

    total_reward = 0.0
    total_resources = 0

    while True:

        action = agent.act(
            observations[agent_name]
        )

        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = env.step(
            {
                agent_name: action
            }
        )

        total_reward += rewards[agent_name]

        if infos[agent_name][
            "collected_resource"
        ] > 0:

            total_resources += 1

        if (
            terminations[agent_name]
            or truncations[agent_name]
        ):
            break

    env.close()

    return total_reward, total_resources


def benchmark_agent(agent_type):

    rewards = []
    resources = []

    for episode in range(NUM_EPISODES):

        reward, resource_count = run_episode(
            agent_type,
            seed=episode,
        )

        rewards.append(reward)
        resources.append(resource_count)

    rewards = np.array(rewards)
    resources = np.array(resources)

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "mean_resources": np.mean(resources),
    }


def main():

    agent_types = [
        "random",
        "greedy",
        "value",
    ]

    results = {}

    print()
    print("=" * 65)
    print("MULTI-AGENT POLICY BENCHMARK")
    print("=" * 65)
    print()

    for agent_type in agent_types:

        print(
            f"Running {agent_type}..."
        )

        results[agent_type] = (
            benchmark_agent(agent_type)
        )

    print()
    print("=" * 65)
    print("RESULTS")
    print("=" * 65)
    print()

    for agent_type, result in results.items():

        print(
            f"{agent_type.upper():<15}"
            f"Mean Reward: "
            f"{result['mean_reward']:>8.2f}    "
            f"Std: "
            f"{result['std_reward']:>8.2f}    "
            f"Resources: "
            f"{result['mean_resources']:>6.2f}"
        )

    print()
    print("-" * 65)

    best_agent = max(
        results,
        key=lambda x:
        results[x]["mean_reward"],
    )

    print(
        f"Best policy by mean reward: "
        f"{best_agent.upper()}"
    )

    print("=" * 65)
    print()


if __name__ == "__main__":
    main()