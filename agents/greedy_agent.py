from agents.base_agent import BaseAgent


class GreedyAgent(BaseAgent):
    """
    Agentی که نزدیک‌ترین Resource را انتخاب می‌کند.
    """

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

    # =========================================================
    # Target
    # =========================================================

    def select_target(self):

        if not self.env.resources:
            return None

        current_position = (
            self.env.positions[self.agent_name]
        )

        best_resource = None
        best_distance = float("inf")

        for position in self.env.resources:

            distance = (
                abs(
                    int(current_position[0])
                    - int(position[0])
                )
                +
                abs(
                    int(current_position[1])
                    - int(position[1])
                )
            )

            if distance < best_distance:
                best_distance = distance
                best_resource = position

        return best_resource

    # =========================================================
    # Action
    # =========================================================

    def action_toward(self, target):

        if target is None:
            return 4

        current = self.env.positions[
            self.agent_name
        ]

        current_x = int(current[0])
        current_y = int(current[1])

        target_x = int(target[0])
        target_y = int(target[1])

        if current_x < target_x:
            return 3

        if current_x > target_x:
            return 2

        if current_y < target_y:
            return 1

        if current_y > target_y:
            return 0

        return 4

    def act(self, observation):

        target = self.select_target()

        action = self.action_toward(
            target
        )

        self.record_action(action)

        return action
