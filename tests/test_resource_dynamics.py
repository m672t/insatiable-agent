from environment.competitive_world import CompetitiveWorld
from agents.value_seeking_agent import ValueSeekingAgent


SCENARIOS = {
    "D1": {
        "spawn_probability": 0.08,
        "resource_lifetime": 120,
        "value_distribution": {
            5: 0.70,
            15: 0.25,
            50: 0.05,
        },
    },

    "D2": {
        "spawn_probability": 0.08,
        "resource_lifetime": 120,
        "value_distribution": {
            5: 0.60,
            15: 0.30,
            50: 0.10,
        },
    },

    "D3": {
        "spawn_probability": 0.08,
        "resource_lifetime": 120,
        "value_distribution": {
            5: 0.50,
            15: 0.30,
            50: 0.20,
        },
    },
}


SEEDS = [42, 43, 44, 45, 46]
STEPS = 500


def run_scenario(
    spawn_probability,
    resource_lifetime,
    value_distribution,
    seed,
):
    env = CompetitiveWorld(
        render_mode=None,
    )

    env.resource_manager.spawn_probability = (
        spawn_probability
    )

    env.resource_manager.resource_lifetime = (
        resource_lifetime
    )

    env.resource_manager.value_distribution = (
        value_distribution.copy()
    )

    observations, infos = env.reset(
        seed=seed
    )

    # -----------------------------------------------------
    # Create agents
    # -----------------------------------------------------

    agents = {
        agent_name: ValueSeekingAgent(
            env=env,
            agent_name=agent_name,
        )
        for agent_name in env.agents
    }

    initial_resources = (
        env.resource_manager.count()
    )

    max_resources_observed = (
        initial_resources
    )

    total_collected_value = 0.0

    total_rewards = {
        agent: 0.0
        for agent in env.agents
    }

    steps_executed = 0

    # -----------------------------------------------------
    # Run episode
    # -----------------------------------------------------

    for _ in range(STEPS):

        actions = {}

        for agent_name, agent in agents.items():

            actions[agent_name] = (
                agent.act(
                    observations[agent_name]
                )
            )

        (
            observations,
            rewards,
            terminated,
            truncated,
            infos,
        ) = env.step(actions)

        steps_executed += 1

        # -------------------------------------------------
        # Rewards
        # -------------------------------------------------

        for agent_name, reward in rewards.items():

            reward = float(reward)

            total_rewards[agent_name] += reward

            total_collected_value += reward

        # -------------------------------------------------
        # Record experience
        # -------------------------------------------------

        for agent_name, agent in agents.items():

            action = actions[agent_name]

            reward = rewards.get(
                agent_name,
                0.0,
            )

            info = infos.get(
                agent_name,
                {},
            )

            agent.record_experience(
                action=action,
                reward=reward,
                info=info,
            )

        # -------------------------------------------------
        # Resource count
        # -------------------------------------------------

        max_resources_observed = max(
            max_resources_observed,
            env.resource_manager.count(),
        )

        # -------------------------------------------------
        # Termination
        # -------------------------------------------------

        if all(truncated.values()):
            break

        if all(terminated.values()):
            break

    metrics = (
        env.resource_manager.get_metrics()
    )

    result = {
        "initial_resources":
            initial_resources,

        "final_resources":
            env.resource_manager.count(),

        "max_resources":
            max_resources_observed,

        "spawned":
            metrics["spawned"],

        "collected":
            metrics["collected"],

        "expired":
            metrics["expired"],

        "mean_spawned_value":
            metrics["mean_value"],

        "mean_collected_value":
            metrics["mean_collected_value"],

        "mean_lifetime":
            metrics["mean_lifetime"],

        "total_collected_value":
            total_collected_value,

        "steps":
            steps_executed,

        "agent_rewards":
            total_rewards,
    }

    env.close()

    return result


def print_scenario_summary(
    name,
    config,
    results,
):
    count = len(results)

    def mean(key):
        return sum(
            result[key]
            for result in results
        ) / count

    print("=" * 60)
    print(f"SCENARIO {name}")
    print("=" * 60)

    print(
        f"Spawn Probability: "
        f"{config['spawn_probability']}"
    )

    print(
        f"Lifetime: "
        f"{config['resource_lifetime']}"
    )

    print(
        f"Value Distribution: "
        f"{config['value_distribution']}"
    )

    print("-" * 60)

    print(
        f"Mean Initial Resources: "
        f"{mean('initial_resources'):.3f}"
    )

    print(
        f"Mean Final Resources: "
        f"{mean('final_resources'):.3f}"
    )

    print(
        f"Mean Maximum Resources: "
        f"{mean('max_resources'):.3f}"
    )

    print(
        f"Mean Spawned: "
        f"{mean('spawned'):.3f}"
    )

    print(
        f"Mean Collected: "
        f"{mean('collected'):.3f}"
    )

    print(
        f"Mean Expired: "
        f"{mean('expired'):.3f}"
    )

    print(
        f"Mean Spawned Value: "
        f"{mean('mean_spawned_value'):.3f}"
    )

    print(
        f"Mean Collected Value: "
        f"{mean('mean_collected_value'):.3f}"
    )

    print(
        f"Mean Lifetime: "
        f"{mean('mean_lifetime'):.3f}"
    )

    print(
        f"Mean Total Collected Value: "
        f"{mean('total_collected_value'):.3f}"
    )

    print(
        f"Mean Steps: "
        f"{mean('steps'):.3f}"
    )

    print()


def main():

    print("=" * 60)
    print("RESOURCE VALUE DISTRIBUTION EXPERIMENT")
    print("=" * 60)

    print(
        f"Seeds: {SEEDS}"
    )

    print(
        f"Steps per run: {STEPS}"
    )

    print()

    all_results = {}

    for name, config in SCENARIOS.items():

        results = []

        for seed in SEEDS:

            result = run_scenario(
                spawn_probability=(
                    config["spawn_probability"]
                ),
                resource_lifetime=(
                    config["resource_lifetime"]
                ),
                value_distribution=(
                    config["value_distribution"]
                ),
                seed=seed,
            )

            results.append(result)

        all_results[name] = results

        print_scenario_summary(
            name,
            config,
            results,
        )

    # -----------------------------------------------------
    # Comparison
    # -----------------------------------------------------

    print("=" * 60)
    print("VALUE DISTRIBUTION COMPARISON")
    print("=" * 60)

    print()

    for name, config in SCENARIOS.items():

        results = all_results[name]

        mean_reward = (
            sum(
                result[
                    "total_collected_value"
                ]
                for result in results
            )
            / len(results)
        )

        mean_collected = (
            sum(
                result["collected"]
                for result in results
            )
            / len(results)
        )

        mean_resources = (
            sum(
                result["final_resources"]
                for result in results
            )
            / len(results)
        )

        mean_spawned_value = (
            sum(
                result["mean_spawned_value"]
                for result in results
            )
            / len(results)
        )

        print(
            f"{name}: "
            f"Distribution="
            f"{config['value_distribution']}, "
            f"Resources={mean_resources:.2f}, "
            f"Collected={mean_collected:.2f}, "
            f"MeanValue={mean_spawned_value:.2f}, "
            f"Reward={mean_reward:.2f}"
        )

    print()
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
