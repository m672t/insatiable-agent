from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):

    def __init__(
        self,
        env,
        agent_name,
        internal_state=None,
    ):
        super().__init__(
            env=env,
            agent_name=agent_name,
            internal_state=internal_state,
        )

    def act(self, observation):

        action = self.env.action_space(
            self.agent_name
        ).sample()

        action = int(action)

        self.record_action(action)

        return action
