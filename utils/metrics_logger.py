
from collections import defaultdict

import numpy as np


class MetricsLogger:
    """
    Day 6 - Experiment Metrics Logger

    Collects:
        Resource metrics
        Agent metrics
        Competitive metrics
        Motivation metrics
    """

    def __init__(
        self,
        agent_names=None,
        resource_manager=None,
        env=None,
    ):
        self.agent_names = list(
            agent_names or []
        )

        self.resource_manager = (
            resource_manager
        )

        self.env = env

        self.reset()

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        self.steps = 0

        # -----------------------------------------------------
        # Optional injected summaries
        #
        # These are intentionally supported because experiment
        # tests may inject controlled metrics directly.
        # -----------------------------------------------------

        self.agent_metrics = None

        self.competitive_metrics = None

        # -----------------------------------------------------
        # Resource metrics
        # -----------------------------------------------------

        self.resource_spawned = 0
        self.resource_collected = 0
        self.resource_expired = 0

        self.resource_lifetimes = []
        self.resource_values = []
        self.collected_resource_values = []

        # -----------------------------------------------------
        # Agent metrics
        # -----------------------------------------------------

        self.total_rewards = defaultdict(float)
        self.reward_samples = defaultdict(list)

        self.resources_collected = defaultdict(int)

        self.distances = defaultdict(list)

        self.action_counts = defaultdict(int)
        self.move_counts = defaultdict(int)

        # -----------------------------------------------------
        # Competitive metrics
        # -----------------------------------------------------

        self.shared_resource_collisions = 0
        self.competition_events = 0

        self.competition_wins = defaultdict(int)
        self.competition_attempts = defaultdict(int)

        self.contender_counts = []

        # -----------------------------------------------------
        # Motivation metrics
        # -----------------------------------------------------

        self.motivation_history = defaultdict(
            lambda: {
                "lack": [],
                "desire": [],
                "satisfaction": [],
                "urgency": [],
            }
        )

    # =========================================================
    # RESOURCE
    # =========================================================

    def record_resource_metrics(
        self,
        spawned=0,
        collected=0,
        expired=0,
        lifetime=None,
        value=None,
        collected_value=None,
    ):

        self.resource_spawned += int(
            spawned
        )

        self.resource_collected += int(
            collected
        )

        self.resource_expired += int(
            expired
        )

        if lifetime is not None:
            self.resource_lifetimes.append(
                float(lifetime)
            )

        if value is not None:
            self.resource_values.append(
                float(value)
            )

        if collected_value is not None:
            self.collected_resource_values.append(
                float(collected_value)
            )

    def _get_resource_metrics(self):

        # -----------------------------------------------------
        # If ResourceManager is available, prefer its metrics.
        # -----------------------------------------------------

        if self.resource_manager is not None:

            manager_metrics = getattr(
                self.resource_manager,
                "metrics",
                None,
            )

            if isinstance(
                manager_metrics,
                dict,
            ):

                # -------------------------------------------------
                # Direct summary metrics
                # -------------------------------------------------

                if (
                    "mean_lifetime"
                    in manager_metrics
                    and
                    "mean_value"
                    in manager_metrics
                ):

                    return {
                        "spawned": int(
                            manager_metrics.get(
                                "spawned",
                                0,
                            )
                        ),

                        "collected": int(
                            manager_metrics.get(
                                "collected",
                                0,
                            )
                        ),

                        "expired": int(
                            manager_metrics.get(
                                "expired",
                                0,
                            )
                        ),

                        "mean_lifetime": float(
                            manager_metrics.get(
                                "mean_lifetime",
                                0.0,
                            )
                        ),

                        "mean_value": float(
                            manager_metrics.get(
                                "mean_value",
                                0.0,
                            )
                        ),

                        "mean_collected_value": float(
                            manager_metrics.get(
                                "mean_collected_value",
                                0.0,
                            )
                        ),
                    }

        # -----------------------------------------------------
        # Logger-owned metrics
        # -----------------------------------------------------

        return {
            "spawned": int(
                self.resource_spawned
            ),

            "collected": int(
                self.resource_collected
            ),

            "expired": int(
                self.resource_expired
            ),

            "mean_lifetime": (
                float(
                    np.mean(
                        self.resource_lifetimes
                    )
                )
                if self.resource_lifetimes
                else 0.0
            ),

            "mean_value": (
                float(
                    np.mean(
                        self.resource_values
                    )
                )
                if self.resource_values
                else 0.0
            ),

            "mean_collected_value": (
                float(
                    np.mean(
                        self.collected_resource_values
                    )
                )
                if self.collected_resource_values
                else 0.0
            ),
        }

    # =========================================================
    # AGENT
    # =========================================================

    def record_agent_step(
        self,
        agent_name,
        reward=0.0,
        distance=None,
        action=None,
        resource_collected=False,
    ):

        self.total_rewards[
            agent_name
        ] += float(reward)

        self.reward_samples[
            agent_name
        ].append(
            float(reward)
        )

        if resource_collected:
            self.resources_collected[
                agent_name
            ] += 1

        if distance is not None:
            self.distances[
                agent_name
            ].append(
                float(distance)
            )

        self.action_counts[
            agent_name
        ] += 1

        if self._is_move_action(action):
            self.move_counts[
                agent_name
            ] += 1

    def _is_move_action(self, action):

        if action is None:
            return False

        if isinstance(action, str):
            return action.lower() in {
                "move",
                "up",
                "down",
                "left",
                "right",
            }

        # Numeric actions:
        # 0,1,2,3 = movement
        # 4 = stay
        try:
            return int(action) in {
                0,
                1,
                2,
                3,
            }
        except (
            TypeError,
            ValueError,
        ):
            return False

    def _get_agent_metrics(self):

        # -----------------------------------------------------
        # Controlled test injection
        # -----------------------------------------------------

        if isinstance(
            self.agent_metrics,
            dict,
        ):
            return self.agent_metrics

        # -----------------------------------------------------
        # Normal calculated metrics
        # -----------------------------------------------------

        result = {}

        names = self.agent_names

        # Include dynamically observed agents too.
        names = list(
            dict.fromkeys(
                list(names)
                + list(self.total_rewards.keys())
                + list(self.action_counts.keys())
            )
        )

        for agent in names:

            rewards = self.reward_samples[
                agent
            ]

            distances = self.distances[
                agent
            ]

            attempts = self.competition_attempts[
                agent
            ]

            wins = self.competition_wins[
                agent
            ]

            result[agent] = {
                "total_reward": float(
                    self.total_rewards[
                        agent
                    ]
                ),

                "average_reward": (
                    float(
                        np.mean(rewards)
                    )
                    if rewards
                    else 0.0
                ),

                "resources_collected": int(
                    self.resources_collected[
                        agent
                    ]
                ),

                "average_distance_to_resource": (
                    float(
                        np.mean(distances)
                    )
                    if distances
                    else 0.0
                ),

                "actions": int(
                    self.action_counts[
                        agent
                    ]
                ),

                "moves": int(
                    self.move_counts[
                        agent
                    ]
                ),

                "win_rate": (
                    float(
                        wins / attempts
                    )
                    if attempts > 0
                    else 0.0
                ),
            }

        return result

    # =========================================================
    # COMPETITION
    # =========================================================

    def record_competition(
        self,
        contenders=0,
        collision=False,
        competition=False,
        winner=None,
    ):

        if contenders is not None:
            self.contender_counts.append(
                int(contenders)
            )

        if collision:
            self.shared_resource_collisions += 1

        if competition:
            self.competition_events += 1

            if winner is not None:
                self.competition_wins[
                    winner
                ] += 1

    def record_competition_attempt(
        self,
        agent_name,
        won=False,
    ):

        self.competition_attempts[
            agent_name
        ] += 1

        if won:
            self.competition_wins[
                agent_name
            ] += 1

    def _get_competitive_metrics(self):

        # -----------------------------------------------------
        # Controlled test injection
        # -----------------------------------------------------

        if isinstance(
            self.competitive_metrics,
            dict,
        ):
            return self.competitive_metrics

        # -----------------------------------------------------
        # Normal calculated metrics
        # -----------------------------------------------------

        return {
            "shared_resource_collisions":
                int(
                    self.shared_resource_collisions
                ),

            "competition_events":
                int(
                    self.competition_events
                ),

            "mean_contenders":
                (
                    float(
                        np.mean(
                            self.contender_counts
                        )
                    )
                    if self.contender_counts
                    else 0.0
                ),
        }

    # =========================================================
    # MOTIVATION
    # =========================================================

    def record_motivation(
        self,
        agent_name,
        motivation,
    ):

        history = self.motivation_history[
            agent_name
        ]

        history["lack"].append(
            float(
                motivation.get(
                    "lack",
                    0.0,
                )
            )
        )

        history["desire"].append(
            float(
                motivation.get(
                    "desire",
                    0.0,
                )
            )
        )

        history["satisfaction"].append(
            float(
                motivation.get(
                    "satisfaction",
                    0.0,
                )
            )
        )

        history["urgency"].append(
            float(
                motivation.get(
                    "urgency",
                    0.0,
                )
            )
        )

    def _get_motivation_metrics(self):

        result = {}

        for agent, history in (
            self.motivation_history.items()
        ):

            result[agent] = {
                "mean_lack": (
                    float(
                        np.mean(
                            history["lack"]
                        )
                    )
                    if history["lack"]
                    else 0.0
                ),

                "mean_desire": (
                    float(
                        np.mean(
                            history["desire"]
                        )
                    )
                    if history["desire"]
                    else 0.0
                ),

                "mean_satisfaction": (
                    float(
                        np.mean(
                            history["satisfaction"]
                        )
                    )
                    if history[
                        "satisfaction"
                    ]
                    else 0.0
                ),

                "mean_urgency": (
                    float(
                        np.mean(
                            history["urgency"]
                        )
                    )
                    if history["urgency"]
                    else 0.0
                ),
            }

        return result

    # =========================================================
    # STEP
    # =========================================================

    def step(self):

        self.steps += 1

    # =========================================================
    # SUMMARY
    # =========================================================

    def get_metrics(self):

        # -----------------------------------------------------
        # Prefer environment step counter if available.
        # -----------------------------------------------------

        steps = self.steps

        if self.env is not None:
            env_step = getattr(
                self.env,
                "step_count",
                None,
            )

            if env_step is not None:
                steps = int(env_step)

        return {
            "steps": int(steps),

            "resource_metrics":
                self._get_resource_metrics(),

            "agent_metrics":
                self._get_agent_metrics(),

            "competitive_metrics":
                self._get_competitive_metrics(),

            "motivation_metrics":
                self._get_motivation_metrics(),
        }

    # =========================================================
    # RESET INJECTION
    # =========================================================

    def set_resource_manager(
        self,
        resource_manager,
    ):
        self.resource_manager = (
            resource_manager
        )

    def set_environment(
        self,
        env,
    ):
        self.env = env
