import functools

import numpy as np
import pygame

from gymnasium import spaces
from pettingzoo import ParallelEnv

from config import EnvironmentConfig
from environment.resource_manager import ResourceManager
from environment.metrics_logger import MetricsLogger


class CompetitiveWorld(ParallelEnv):
    """
    محیط رقابتی چند-Agent برای جمع‌آوری Resource.

    Features:
        - حرکت هم‌زمان Agentها
        - رقابت برای Resource
        - انتخاب تصادفی برنده در برخورد هم‌زمان
        - Reward بر اساس ارزش Resource
        - Resourceهای پویا
        - Spawn تصادفی Resourceهای جدید
        - Expire شدن Resourceهای قدیمی
        - محدودیت حداکثر تعداد Resource
        - Metrics Logger
    """

    metadata = {
        "name": "competitive_world_v0",
        "render_modes": ["human"],
        "render_fps": 10,
    }

    def __init__(
        self,
        grid_size=EnvironmentConfig.grid_size,
        num_agents=EnvironmentConfig.num_agents,
        num_resources=EnvironmentConfig.num_resources,
        render_mode=EnvironmentConfig.render_mode,
    ):

        self.grid_size = int(grid_size)
        self.n_agents = int(num_agents)
        self.num_resources = int(num_resources)
        self.render_mode = render_mode

        self.max_steps = int(
            EnvironmentConfig.max_steps
        )

        # =====================================================
        # Agents
        # =====================================================

        self.possible_agents = [
            f"agent_{i}"
            for i in range(self.n_agents)
        ]

        self.agents = []

        # =====================================================
        # Agent colors
        # =====================================================

        self.agent_colors = [
            (220, 70, 70),
            (70, 120, 220),
            (70, 180, 100),
            (220, 170, 60),
        ]

        # =====================================================
        # Actions
        # =====================================================

        self.action_spaces = {
            agent: spaces.Discrete(5)
            for agent in self.possible_agents
        }

        # =====================================================
        # Observation
        # =====================================================

        observation_size = (
            2 + (self.num_resources * 3)
        )

        self.observation_spaces = {
            agent: spaces.Box(
                low=-1,
                high=max(
                    self.grid_size - 1,
                    50,
                ),
                shape=(observation_size,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        # =====================================================
        # World State
        # =====================================================

        self.positions = {}
        self.resources = {}

        # =====================================================
        # Resource Manager
        # =====================================================

        self.resource_manager = ResourceManager(
            grid_size=self.grid_size,
            initial_resources=self.num_resources,
            max_resources=25,
            spawn_probability=0.08,
            min_value=5,
            max_value=50,
            resource_lifetime=120,
        )

        # =====================================================
        # Metrics Logger
        # =====================================================

        self.metrics_logger = MetricsLogger(
            agent_names=self.possible_agents,
            resource_manager=self.resource_manager,
        )

        # Explicit binding for compatibility.
        self.metrics_logger.bind_resource_manager(
            self.resource_manager
        )

        # =====================================================
        # Step
        # =====================================================

        self.step_count = 0

        # =====================================================
        # Total Rewards
        # =====================================================

        self.total_rewards = {
            agent: 0.0
            for agent in self.possible_agents
        }

        # =====================================================
        # Rendering
        # =====================================================

        self.window = None
        self.clock = None
        self.font = None

        self.cell_size = 30
        self.hud_height = 100

    # =========================================================
    # Spaces
    # =========================================================

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    # =========================================================
    # RESET
    # =========================================================

    def reset(
        self,
        seed=None,
        options=None,
    ):

        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents.copy()

        self.step_count = 0

        self.total_rewards = {
            agent: 0.0
            for agent in self.possible_agents
        }

        # Reset logger first.
        self.metrics_logger.reset()

        # Re-bind manager after logger reset.
        self.metrics_logger.bind_resource_manager(
            self.resource_manager
        )

        # -----------------------------------------------------
        # Initial positions
        # -----------------------------------------------------

        self.positions = {}

        for i, agent in enumerate(
            self.possible_agents
        ):

            if i == 0:
                position = (1, 1)

            elif i == 1:
                position = (
                    self.grid_size - 2,
                    1,
                )

            elif i == 2:
                position = (
                    1,
                    self.grid_size - 2,
                )

            elif i == 3:
                position = (
                    self.grid_size - 2,
                    self.grid_size - 2,
                )

            else:
                position = self._find_agent_position(i)

            self.positions[agent] = np.array(
                position,
                dtype=np.int32,
            )

        # -----------------------------------------------------
        # Initial resources
        # -----------------------------------------------------

        self.resource_manager.reset(
            occupied_positions=self.positions.values(),
            step=self.step_count,
        )

        self._sync_resources()

        self.metrics_logger.sync_resource_manager(
            self.resource_manager
        )

        # -----------------------------------------------------
        # Observations
        # -----------------------------------------------------

        observations = {
            agent: self._get_observation(agent)
            for agent in self.agents
        }

        infos = {
            agent: {}
            for agent in self.agents
        }

        if self.render_mode == "human":
            self.render()

        return observations, infos

    # =========================================================
    # HELPERS
    # =========================================================

    def _find_agent_position(self, index):
        occupied = {
            tuple(position)
            for position in self.positions.values()
        }

        for x in range(self.grid_size):
            for y in range(self.grid_size):

                if (x, y) not in occupied:
                    return (x, y)

        return (0, 0)

    # =========================================================
    # RESOURCE SYNCHRONIZATION
    # =========================================================

    def _sync_resources(self):
        self.resources = (
            self.resource_manager.get_resources()
        )

    # =========================================================
    # OBSERVATION
    # =========================================================

    def _get_observation(self, agent):

        observation = []

        x, y = self.positions[agent]

        observation.extend([
            float(x),
            float(y),
        ])

        resource_items = sorted(
            self.resources.items()
        )

        for i in range(self.num_resources):

            if i < len(resource_items):

                (rx, ry), value = (
                    resource_items[i]
                )

                observation.extend([
                    float(rx),
                    float(ry),
                    float(value),
                ])

            else:

                observation.extend([
                    -1.0,
                    -1.0,
                    0.0,
                ])

        return np.array(
            observation,
            dtype=np.float32,
        )

    # =========================================================
    # STEP
    # =========================================================

    def step(self, actions):

        # -----------------------------------------------------
        # 1. Save old positions
        # -----------------------------------------------------

        old_positions = {
            agent: self.positions[agent].copy()
            for agent in self.agents
        }

        # -----------------------------------------------------
        # 2. Proposed positions
        # -----------------------------------------------------

        proposed_positions = {}

        for agent in self.agents:

            action = actions.get(agent, 4)

            current = old_positions[agent].copy()

            try:
                action = int(action)
            except (TypeError, ValueError):
                action = 4

            if action == 0:
                current[1] -= 1

            elif action == 1:
                current[1] += 1

            elif action == 2:
                current[0] -= 1

            elif action == 3:
                current[0] += 1

            # action == 4 => stay

            current[0] = np.clip(
                current[0],
                0,
                self.grid_size - 1,
            )

            current[1] = np.clip(
                current[1],
                0,
                self.grid_size - 1,
            )

            proposed_positions[agent] = current

        # -----------------------------------------------------
        # 3. Simultaneous movement
        # -----------------------------------------------------

        self.positions = proposed_positions

        # -----------------------------------------------------
        # 4. Rewards
        # -----------------------------------------------------

        rewards = {
            agent: 0.0
            for agent in self.agents
        }

        collected_resources = {
            agent: 0.0
            for agent in self.agents
        }

        # -----------------------------------------------------
        # 5. Find resource claims
        # -----------------------------------------------------

        resource_claims = {}

        for agent in self.agents:

            position = tuple(
                self.positions[agent]
            )

            if position in self.resources:

                if position not in resource_claims:
                    resource_claims[position] = []

                resource_claims[position].append(
                    agent
                )

        # -----------------------------------------------------
        # 6. Resolve competition
        # -----------------------------------------------------

        for position, contenders in (
            resource_claims.items()
        ):

            contender_count = len(contenders)

            collision = contender_count > 1
            competition = contender_count > 1

            winner = np.random.choice(contenders)

            self.metrics_logger.record_competition(
                contenders=contender_count,
                collision=collision,
                competition=competition,
                winner=winner,
            )

            for contender in contenders:

                self.metrics_logger.record_competition_attempt(
                    agent_name=contender,
                    won=(contender == winner),
                )

            value = self.resource_manager.collect(
                position
            )

            if value is None:
                continue

            value = float(value)

            rewards[winner] += value

            self.total_rewards[winner] += value

            collected_resources[winner] += value

        # -----------------------------------------------------
        # 7. Agent metrics
        #
        # Record exactly once per environment step.
        # -----------------------------------------------------

        for agent in self.agents:

            action = actions.get(agent, 4)

            distance = (
                self._distance_to_nearest_resource(
                    agent
                )
            )

            self.metrics_logger.record_agent_step(
                agent_name=agent,
                reward=rewards[agent],
                distance=distance,
                action=action,
                resource_collected=(
                    collected_resources[agent] > 0.0
                ),
            )

        # -----------------------------------------------------
        # 8. Advance time
        # -----------------------------------------------------

        self.step_count += 1

        # -----------------------------------------------------
        # 9. Dynamic resource update
        # -----------------------------------------------------

        self.resource_manager.update(
            step=self.step_count,
            occupied_positions=self.positions.values(),
        )

        self._sync_resources()

        # -----------------------------------------------------
        # 10. Sync resource metrics
        # -----------------------------------------------------

        self.metrics_logger.sync_resource_manager(
            self.resource_manager
        )

        # -----------------------------------------------------
        # 11. Logger step
        #
        # Exactly once per environment step.
        # -----------------------------------------------------

        self.metrics_logger.step()

        # -----------------------------------------------------
        # 12. Termination
        # -----------------------------------------------------

        terminated = {
            agent: False
            for agent in self.agents
        }

        truncated = {
            agent: (
                self.step_count >= self.max_steps
            )
            for agent in self.agents
        }

        # -----------------------------------------------------
        # 13. New observations
        # -----------------------------------------------------

        observations = {
            agent: self._get_observation(agent)
            for agent in self.agents
        }

        # -----------------------------------------------------
        # 14. Info
        # -----------------------------------------------------

        infos = {}

        for agent in self.agents:

            infos[agent] = {
                "position":
                    self.positions[agent].copy(),

                "collected_resource":
                    collected_resources[agent],

                "total_reward":
                    self.total_rewards[agent],

                "remaining_resources":
                    self.resource_manager.count(),
            }

        # -----------------------------------------------------
        # 15. Render
        # -----------------------------------------------------

        if self.render_mode == "human":
            self.render()

        return (
            observations,
            rewards,
            terminated,
            truncated,
            infos,
        )

    # =========================================================
    # RESOURCE LOGGER SYNC
    # =========================================================

    def _sync_resource_logger(self, metrics=None):

        if metrics is None:
            metrics = self.resource_manager.get_metrics()

        self.metrics_logger.sync_resource_manager(
            self.resource_manager
        )

    # =========================================================
    # DISTANCE
    # =========================================================

    def _distance_to_nearest_resource(self, agent):

        if not self.resources:
            return 0.0

        agent_position = self.positions[agent]

        distances = []

        for position in self.resources.keys():

            resource_position = np.array(
                position,
                dtype=np.float32,
            )

            distance = np.linalg.norm(
                agent_position
                - resource_position
            )

            distances.append(float(distance))

        return (
            min(distances)
            if distances
            else 0.0
        )

    # =========================================================
    # RENDER
    # =========================================================

    def render(self):

        if self.render_mode != "human":
            return

        if self.window is None:

            pygame.init()

            window_width = (
                self.grid_size * self.cell_size
            )

            window_height = (
                self.grid_size * self.cell_size
                + self.hud_height
            )

            self.window = pygame.display.set_mode(
                (
                    window_width,
                    window_height,
                )
            )

            pygame.display.set_caption(
                "Competitive Resource World"
            )

            self.clock = pygame.time.Clock()

            self.font = pygame.font.SysFont(
                "Arial",
                18,
            )

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                pygame.quit()
                self.window = None

                return

        self.window.fill(
            (245, 245, 245)
        )

        # -----------------------------------------------------
        # Grid
        # -----------------------------------------------------

        for x in range(self.grid_size):

            for y in range(self.grid_size):

                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                pygame.draw.rect(
                    self.window,
                    (210, 210, 210),
                    rect,
                    1,
                )

        # -----------------------------------------------------
        # Resources
        # -----------------------------------------------------

        for (x, y), value in (
            self.resources.items()
        ):

            center = (
                x * self.cell_size
                + self.cell_size // 2,

                y * self.cell_size
                + self.cell_size // 2,
            )

            if value == 5:
                radius = 5
            elif value == 15:
                radius = 8
            else:
                radius = 11

            pygame.draw.circle(
                self.window,
                (80, 80, 80),
                center,
                radius,
            )

        # -----------------------------------------------------
        # Agents
        # -----------------------------------------------------

        for i, agent in enumerate(self.agents):

            x, y = self.positions[agent]

            center = (
                x * self.cell_size
                + self.cell_size // 2,

                y * self.cell_size
                + self.cell_size // 2,
            )

            radius = self.cell_size // 3

            color = self.agent_colors[
                i % len(self.agent_colors)
            ]

            pygame.draw.circle(
                self.window,
                color,
                center,
                radius,
            )

        # -----------------------------------------------------
        # HUD
        # -----------------------------------------------------

        hud_y = (
            self.grid_size * self.cell_size
        )

        pygame.draw.rect(
            self.window,
            (230, 230, 230),
            pygame.Rect(
                0,
                hud_y,
                self.grid_size * self.cell_size,
                self.hud_height,
            ),
        )

        # -----------------------------------------------------
        # Agent rewards
        # -----------------------------------------------------

        for i, agent in enumerate(self.agents):

            text = (
                f"{agent}: "
                f"{self.total_rewards[agent]:.0f}"
            )

            color = self.agent_colors[
                i % len(self.agent_colors)
            ]

            surface = self.font.render(
                text,
                True,
                color,
            )

            x_position = 10 + (i * 145)

            self.window.blit(
                surface,
                (
                    x_position,
                    hud_y + 10,
                ),
            )

        # -----------------------------------------------------
        # Step
        # -----------------------------------------------------

        step_text = (
            f"Step: {self.step_count}"
        )

        step_surface = self.font.render(
            step_text,
            True,
            (40, 40, 40),
        )

        self.window.blit(
            step_surface,
            (
                10,
                hud_y + 45,
            ),
        )

        # -----------------------------------------------------
        # Resource count
        # -----------------------------------------------------

        resource_text = (
            f"Resources: "
            f"{self.resource_manager.count()}"
        )

        resource_surface = self.font.render(
            resource_text,
            True,
            (40, 40, 40),
        )

        self.window.blit(
            resource_surface,
            (
                150,
                hud_y + 45,
            ),
        )

        # -----------------------------------------------------
        # Display
        # -----------------------------------------------------

        pygame.display.flip()

        self.clock.tick(
            self.metadata["render_fps"]
        )

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):

        if self.window is not None:

            pygame.quit()

            self.window = None
            self.clock = None
            self.font = None
