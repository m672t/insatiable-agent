from environment.competitive_world import CompetitiveWorld

from agents.agent_factory import AgentFactory
from simulation.episode_runner import EpisodeRunner


def main():

    env = CompetitiveWorld(
        render_mode="human",
    )

    agent_types = {
        "agent_0": "random",
        "agent_1": "greedy",
        "agent_2": "value",
        "agent_3": "value",
    }

    agents = AgentFactory.create_agents(
        env,
        agent_types,
    )

    runner = EpisodeRunner(
        env,
        agents,
    )

    total_rewards, infos = runner.run(
        seed=42,
        episode_id=1,
    )

    print()
    print("Episode finished.")
    print("Total rewards:")

    for agent_name, reward in total_rewards.items():

        print(
            f"  {agent_name}: "
            f"{reward:.2f}"
        )

    print(
        "Steps logged:",
        runner.logger.get_step_count(),
    )

    env.close()


if __name__ == "__main__":
    main()