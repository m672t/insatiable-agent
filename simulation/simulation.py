from environment.competitive_world import CompetitiveWorld

from agents.agent_factory import AgentFactory

from simulation.episode_runner import EpisodeRunner


class Simulation:
    """
    لایه اصلی مدیریت Simulation.

    مسئولیت این کلاس:
    - ساخت Environment
    - ساخت Agentها
    - ساخت EpisodeRunner
    - اجرای Episode
    """

    def __init__(
        self,
        render_mode=None,
        log_directory="logs",
    ):

        # -----------------------------
        # Environment
        # -----------------------------

        self.env = CompetitiveWorld(
            render_mode=render_mode
        )

        # -----------------------------
        # Agent configuration
        # -----------------------------

        agent_types = {
            "agent_0": "random",
            "agent_1": "greedy",
            "agent_2": "value",
            "agent_3": "value",
        }

        # -----------------------------
        # Create Agents
        # -----------------------------

        self.agents = AgentFactory.create_agents(
            self.env,
            agent_types,
        )

        # -----------------------------
        # Episode Runner
        # -----------------------------

        self.runner = EpisodeRunner(
            env=self.env,
            agents=self.agents,
            log_directory=log_directory,
        )

    def run_episode(
        self,
        seed=None,
        episode_id=None,
        save_log=True,
    ):
        """
        اجرای یک Episode.
        """

        return self.runner.run(
            seed=seed,
            episode_id=episode_id,
            save_log=save_log,
        )

    def close(self):
        """
        بستن Simulation و Environment.
        """

        self.env.close()
