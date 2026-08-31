from agents.agent_factory import AgentFactory
from environment.competitive_world import CompetitiveWorld


def run_episode(
    env,
    agents,
):
    observations, infos = env.reset()

    for agent in agents.values():
        agent.reset_episode()

    total_rewards = {
        agent_name: 0.0
        for agent_name in agents
    }

    while env.agents:

        # -----------------------------------------------------
        # 1. Agents decide
        # -----------------------------------------------------

        actions = {}

        for agent_name, agent in agents.items():

            if agent_name not in env.agents:
                continue

            action = agent.act(
                observations[agent_name]
            )

            actions[agent_name] = int(action)

        # -----------------------------------------------------
        # 2. Environment executes actions
        # -----------------------------------------------------

        (
            next_observations,
            rewards,
            terminated,
            truncated,
            infos,
        ) = env.step(actions)

        # -----------------------------------------------------
        # 3. Record REAL experience
        # -----------------------------------------------------

        for agent_name, agent in agents.items():

            if agent_name not in rewards:
                continue

            reward = rewards[agent_name]

            info = infos.get(
                agent_name,
                {},
            )

            agent.record_experience(
                action=actions[agent_name],
                reward=reward,
                info=info,
            )

            total_rewards[agent_name] += float(
                reward
            )

        # -----------------------------------------------------
        # 4. Next observation
        # -----------------------------------------------------

        observations = next_observations

        # -----------------------------------------------------
        # 5. Termination
        # -----------------------------------------------------

        if all(
            terminated.get(agent_name, False)
            or truncated.get(agent_name, False)
            for agent_name in env.agents
        ):
            break

    return total_rewards


def main():

    env = CompetitiveWorld(
        render_mode=None
    )

    agent_types = {
        "agent_0": "random",
        "agent_1": "greedy",
        "agent_2": "value",
        "agent_3": "greedy",
    }

    agents = AgentFactory.create_agents(
        env,
        agent_types,
    )

    total_rewards = run_episode(
        env,
        agents,
    )

    print()
    print("Episode finished.")
    print("Total rewards:")

    for agent_name, reward in (
        total_rewards.items()
    ):
        print(
            f"  {agent_name}: "
            f"{reward:.2f}"
        )

    print()
    print("Value Agent State:")

    value_agent = agents["agent_2"]

    print(
        "  Memory:",
        len(value_agent.get_memory()),
    )

    print(
        "  Motivation:",
        value_agent.get_motivation_state(),
    )

    print(
        "  Risk:",
        round(
            value_agent.get_risk_tolerance(),
            3,
        ),
    )

    env.close()


if __name__ == "__main__":
    main()
