from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.value_seeking_agent import ValueSeekingAgent


class AgentFactory:
    """
    ساخت Agentها.
    """

    @staticmethod
    def create_agent(
        env,
        agent_name,
        agent_type,
    ):

        if agent_type == "random":

            return RandomAgent(
                env,
                agent_name,
            )

        if agent_type == "greedy":

            return GreedyAgent(
                env,
                agent_name,
            )

        if agent_type == "value":

            return ValueSeekingAgent(
                env,
                agent_name,
            )

        raise ValueError(
            f"Unknown agent type: {agent_type}"
        )

    @staticmethod
    def create_agents(
        env,
        agent_types,
    ):

        agents = {}

        for agent_name, agent_type in (
            agent_types.items()
        ):

            agents[agent_name] = (
                AgentFactory.create_agent(
                    env,
                    agent_name,
                    agent_type,
                )
            )

        return agents

    @staticmethod
    def reset_agents(agents):
        """
        Reset Episode بدون حذف Memory.
        """

        for agent in agents.values():
            agent.reset_episode()
