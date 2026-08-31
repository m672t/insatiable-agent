from environment.competitive_world import CompetitiveWorld


def run_baseline(
    seed=42,
    steps=500,
):
    env = CompetitiveWorld(
        render_mode=None
    )

    observations, infos = env.reset(
        seed=seed
    )

    initial_resources = (
        env.resource_manager.count()
    )

    max_resources_observed = (
        initial_resources
    )

    total_rewards = {
        agent: 0.0
        for agent in env.agents
    }

    total_collected = 0.0

    for step in range(steps):

        # Baseline policy:
        # حرکت تصادفی برای آزمایش خود Environment
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }

        (
            observations,
            rewards,
            terminated,
            truncated,
            infos,
        ) = env.step(actions)

        # ---------------------------------
        # Rewards
        # ---------------------------------

        for agent, reward in rewards.items():
            total_rewards[agent] += float(
                reward
            )

        # ---------------------------------
        # Collection
        # ---------------------------------

        for agent_info in infos.values():
            total_collected += float(
                agent_info.get(
                    "collected_resource",
                    0.0,
                )
            )

        # ---------------------------------
        # Resource count
        # ---------------------------------

        current_count = (
            env.resource_manager.count()
        )

        max_resources_observed = max(
            max_resources_observed,
            current_count,
        )

        # ---------------------------------
        # Stop if episode ended
        # ---------------------------------

        if all(
            terminated[agent]
            or truncated[agent]
            for agent in env.agents
        ):
            break

    metrics = (
        env.resource_manager.get_metrics()
    )

    print("=" * 60)
    print("ENVIRONMENT BASELINE")
    print("=" * 60)

    print(
        f"Seed: {seed}"
    )

    print(
        f"Steps requested: {steps}"
    )

    print(
        f"Steps executed: {env.step_count}"
    )

    print("-" * 60)

    print(
        f"Initial resources: "
        f"{initial_resources}"
    )

    print(
        f"Final resources: "
        f"{env.resource_manager.count()}"
    )

    print(
        f"Maximum resources observed: "
        f"{max_resources_observed}"
    )

    print("-" * 60)

    print(
        f"Spawned: "
        f"{metrics['spawned']}"
    )

    print(
        f"Collected: "
        f"{metrics['collected']}"
    )

    print(
        f"Expired: "
        f"{metrics['expired']}"
    )

    print(
        f"Mean spawned value: "
        f"{metrics['mean_value']:.3f}"
    )

    print(
        f"Mean collected value: "
        f"{metrics['mean_collected_value']:.3f}"
    )

    print(
        f"Mean lifetime: "
        f"{metrics['mean_lifetime']:.3f}"
    )

    print("-" * 60)

    print(
        f"Total collected value: "
        f"{total_collected:.3f}"
    )

    print("Rewards:")

    for agent, reward in total_rewards.items():
        print(
            f"  {agent}: {reward:.3f}"
        )

    print("-" * 60)

    # ---------------------------------
    # Basic integrity checks
    # ---------------------------------

    assert (
        initial_resources
        <= env.resource_manager.max_resources
    ), (
        "Initial resources exceed "
        "max_resources."
    )

    assert (
        max_resources_observed
        <= env.resource_manager.max_resources
    ), (
        "Resource count exceeded "
        "max_resources."
    )

    assert (
        metrics["spawned"] >= initial_resources
    ), (
        "Spawned count is smaller than "
        "initial resource count."
    )

    assert (
        metrics["collected"] >= 0
    )

    assert (
        metrics["expired"] >= 0
    )

    assert (
        metrics["mean_value"] >= 0.0
    )

    assert (
        metrics["mean_collected_value"]
        >= 0.0
    )

    print(
        "Baseline integrity checks: PASS"
    )

    print("=" * 60)

    env.close()


if __name__ == "__main__":
    run_baseline(
        seed=42,
        steps=500,
    )
