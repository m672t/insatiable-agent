from pathlib import Path

from simulation.logger import EpisodeLogger


class EpisodeRunner:
    """
    مدیریت اجرای یک Episode.

    مسئول:
    - اجرای Episode
    - گرفتن Action از Agentها
    - اجرای Environment
    - ثبت Stepها
    - ثبت تجربه در InternalState
    - ذخیره Log
    """

    def __init__(
        self,
        env,
        agents,
        logger=None,
        log_directory="logs",
    ):
        self.env = env
        self.agents = agents

        self.logger = (
            logger
            if logger is not None
            else EpisodeLogger()
        )

        self.log_directory = Path(
            log_directory
        )

    def run(
        self,
        seed=None,
        episode_id=None,
        save_log=True,
    ):
        # -----------------------------
        # 1. Reset Logger
        # -----------------------------

        self.logger.reset()

        # -----------------------------
        # 2. Reset Agentها
        # -----------------------------

        for agent in self.agents.values():

            if hasattr(agent, "reset_episode"):
                agent.reset_episode()

        # -----------------------------
        # 3. Reset Environment
        # -----------------------------

        observations, infos = self.env.reset(
            seed=seed
        )

        total_rewards = {
            agent_name: 0.0
            for agent_name in self.env.agents
        }

        step = 0

        # -----------------------------
        # 4. Episode Loop
        # -----------------------------

        while True:

            actions = {}

            # -----------------------------
            # انتخاب Action
            # -----------------------------

            for agent_name in self.env.agents:

                action = self.agents[
                    agent_name
                ].act(
                    observations[agent_name]
                )

                actions[agent_name] = action

            # -----------------------------
            # اجرای Step
            # -----------------------------

            (
                observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = self.env.step(actions)

            # -----------------------------
            # ثبت تجربه Agentها
            # -----------------------------

            for agent_name in self.env.agents:

                agent = self.agents[
                    agent_name
                ]

                if hasattr(
                    agent,
                    "record_experience",
                ):
                    agent.record_experience(
                        action=actions[agent_name],
                        reward=rewards[agent_name],
                        info=infos.get(
                            agent_name,
                            {},
                        ),
                    )

            # -----------------------------
            # ثبت Step در Logger
            # -----------------------------

            
# -----------------------------
# جمع‌آوری وضعیت داخلی Agentها
# -----------------------------

            internal_states = {}

            for agent_name, agent in self.agents.items():

                if hasattr(agent, "get_decision_context"):

                    internal_states[agent_name] = (
                        agent.get_decision_context()
                    )

# -----------------------------
# ثبت Step در Logger
# -----------------------------

            self.logger.log_step(
                step=step,
                actions=actions,
                rewards=rewards,
                infos=infos,
                internal_states=internal_states,
            )



            # -----------------------------
            # جمع Reward
            # -----------------------------

            for agent_name in self.env.agents:

                total_rewards[agent_name] += (
                    rewards[agent_name]
                )

            step += 1

            # -----------------------------
            # پایان Episode
            # -----------------------------

            if all(
                terminations[agent_name]
                or truncations[agent_name]
                for agent_name in self.env.agents
            ):
                break

        # -----------------------------
        # 5. ذخیره Log
        # -----------------------------

        if save_log:

            if episode_id is None:
                episode_id = 0

            filename = (
                f"episode_{episode_id:04d}.json"
            )

            filepath = (
                self.log_directory
                / filename
            )

            self.logger.save_json(
                filepath
            )

        return total_rewards, infos
