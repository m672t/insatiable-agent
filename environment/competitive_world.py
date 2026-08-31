import functools

import numpy as np
import pygame
from config import EnvironmentConfig
from environment.resource_manager import ResourceManager
from gymnasium import spaces
from pettingzoo import ParallelEnv


class CompetitiveWorld(ParallelEnv):
    """
    محیط رقابتی چند-Agent برای جمع‌آوری Resource.

    ویژگی‌ها:
    - حرکت هم‌زمان Agentها
    - رقابت برای Resource
    - انتخاب تصادفی برنده در برخورد هم‌زمان
    - Reward بر اساس ارزش Resource
    - Resourceهای پویا
    - Spawn تصادفی Resourceهای جدید
    - Expire شدن Resourceهای قدیمی
    - محدودیت حداکثر تعداد Resource
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
        self.grid_size = grid_size
        self.n_agents = num_agents
        self.num_resources = num_resources
        self.render_mode = render_mode
        self.max_steps = EnvironmentConfig.max_steps

        # ---------------------------------
        # Agents
        # ---------------------------------

        self.possible_agents = [
            f"agent_{i}"
            for i in range(num_agents)
        ]

        self.agents = []

        # ---------------------------------
        # رنگ Agentها
        # ---------------------------------

        self.agent_colors = [
            (220, 70, 70),
            (70, 120, 220),
            (70, 180, 100),
            (220, 170, 60),
        ]

        # ---------------------------------
        # Actions
        # ---------------------------------
        #
        # 0 = UP
        # 1 = DOWN
        # 2 = LEFT
        # 3 = RIGHT
        # 4 = STAY
        #

        self.action_spaces = {
            agent: spaces.Discrete(5)
            for agent in self.possible_agents
        }

        # ---------------------------------
        # Observation
        # ---------------------------------
        #
        # [agent_x, agent_y,
        #
        #  resource_1_x,
        #  resource_1_y,
        #  resource_1_value,
        #
        #  resource_2_x,
        #  resource_2_y,
        #  resource_2_value,
        #
        #  ...]
        #
        # اگر Resource وجود نداشته باشد:
        #
        # x = -1
        # y = -1
        # value = 0
        #

        observation_size = (
            2 + (num_resources * 3)
        )

        self.observation_spaces = {
            agent: spaces.Box(
                low=-1,
                high=max(
                    grid_size - 1,
                    50,
                ),
                shape=(observation_size,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        # ---------------------------------
        # World State
        # ---------------------------------

        self.positions = {}

        self.resources = {}

        # ---------------------------------
        # Resource Manager
        # ---------------------------------

        self.resource_manager = ResourceManager(
            grid_size=self.grid_size,
            initial_resources=self.num_resources,
            max_resources=25,
            spawn_probability=0.08,
            min_value=5,
            max_value=50,
            resource_lifetime=120,
        )

        # ---------------------------------
        # Step
        # ---------------------------------

        self.step_count = 0

        # ---------------------------------
        # Total Rewards
        # ---------------------------------

        self.total_rewards = {
            agent: 0.0
            for agent in self.possible_agents
        }

        # ---------------------------------
        # Rendering
        # ---------------------------------

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
    # Reset
    # =========================================================

    def reset(self, seed=None, options=None):

        if seed is not None:
            np.random.seed(seed)

        # ---------------------------------
        # Reset agents
        # ---------------------------------

        self.agents = self.possible_agents.copy()

        self.step_count = 0

        # ---------------------------------
        # Reset rewards
        # ---------------------------------

        self.total_rewards = {
            agent: 0.0
            for agent in self.possible_agents
        }

        # ---------------------------------
        # Initial positions
        # ---------------------------------

        self.positions = {
            "agent_0": np.array(
                [1, 1],
                dtype=np.int32,
            ),

            "agent_1": np.array(
                [
                    self.grid_size - 2,
                    1,
                ],
                dtype=np.int32,
            ),

            "agent_2": np.array(
                [
                    1,
                    self.grid_size - 2,
                ],
                dtype=np.int32,
            ),

            "agent_3": np.array(
                [
                    self.grid_size - 2,
                    self.grid_size - 2,
                ],
                dtype=np.int32,
            ),
        }

        # ---------------------------------
        # Initial dynamic resources
        # ---------------------------------

        self.resource_manager.reset(
            occupied_positions=self.positions.values(),
            step=self.step_count,
        )

        self._sync_resources()

        # ---------------------------------
        # Observations
        # ---------------------------------

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
    # Resource synchronization
    # =========================================================

    def _sync_resources(self):
        """
        هماهنگ کردن Snapshot منابع Environment
        با ResourceManager.
        """

        self.resources = (
            self.resource_manager.get_resources()
        )

    # =========================================================
    # Observation
    # =========================================================

    def _get_observation(self, agent):

        observation = []

        # ---------------------------------
        # Agent position
        # ---------------------------------

        x, y = self.positions[agent]

        observation.extend([
            float(x),
            float(y),
        ])

        # ---------------------------------
        # Resources
        # ---------------------------------

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
    # Step
    # =========================================================

    def step(self, actions):

        # ---------------------------------
        # 1. Save old positions
        # ---------------------------------

        old_positions = {
            agent: self.positions[agent].copy()
            for agent in self.agents
        }

        # ---------------------------------
        # 2. Calculate proposed positions
        # ---------------------------------

        proposed_positions = {}

        for agent, action in actions.items():

            current = old_positions[agent].copy()

            if action == 0:
                # UP
                current[1] -= 1

            elif action == 1:
                # DOWN
                current[1] += 1

            elif action == 2:
                # LEFT
                current[0] -= 1

            elif action == 3:
                # RIGHT
                current[0] += 1

            elif action == 4:
                # STAY
                pass

            # ---------------------------------
            # Keep inside grid
            # ---------------------------------

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

        # ---------------------------------
        # Simultaneous movement
        # ---------------------------------

        self.positions = proposed_positions

        # ---------------------------------
        # 3. Initialize rewards
        # ---------------------------------

        rewards = {
            agent: 0.0
            for agent in self.agents
        }

        collected_resources = {
            agent: 0.0
            for agent in self.agents
        }

        # ---------------------------------
        # 4. Find Resource claims
        # ---------------------------------

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

        # ---------------------------------
        # 5. Resolve Resource competition
        # ---------------------------------

        for position, contenders in (
            resource_claims.items()
        ):

            # ---------------------------------
            # Select winner
            # ---------------------------------

            winner = np.random.choice(
                contenders
            )

            # ---------------------------------
            # Collect from ResourceManager
            # ---------------------------------

            value = self.resource_manager.collect(
                position
            )

            if value is None:
                continue

            value = float(value)

            # ---------------------------------
            # Reward
            # ---------------------------------

            rewards[winner] += value

            self.total_rewards[winner] += value

            collected_resources[winner] += value

        # ---------------------------------
        # 6. Advance time
        # ---------------------------------

        self.step_count += 1

        # ---------------------------------
        # 7. Dynamic Resource update
        # ---------------------------------
        #
        # شامل:
        # - حذف Resourceهای قدیمی
        # - Spawn Resourceهای جدید
        #

        self.resource_manager.update(
            step=self.step_count,
            occupied_positions=self.positions.values(),
        )

        # ---------------------------------
        # Sync resources
        # ---------------------------------

        self._sync_resources()

        # ---------------------------------
        # 8. Termination
        # ---------------------------------

        terminated = {
            agent: False
            for agent in self.agents
        }

        truncated = {
            agent: (
                self.step_count
                >= self.max_steps
            )
            for agent in self.agents
        }

        # ---------------------------------
        # 9. New observations
        # ---------------------------------

        observations = {
            agent: self._get_observation(agent)
            for agent in self.agents
        }

        # ---------------------------------
        # 10. Info
        # ---------------------------------

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

        # ---------------------------------
        # Render
        # ---------------------------------

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
    # Render
    # =========================================================

    def render(self):

        if self.render_mode != "human":
            return

        # ---------------------------------
        # Create window
        # ---------------------------------

        if self.window is None:

            pygame.init()

            window_width = (
                self.grid_size
                * self.cell_size
            )

            window_height = (
                self.grid_size
                * self.cell_size
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

        # ---------------------------------
        # Events
        # ---------------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                pygame.quit()

                self.window = None

                return

        # ---------------------------------
        # Background
        # ---------------------------------

        self.window.fill(
            (245, 245, 245)
        )

        # ---------------------------------
        # Grid
        # ---------------------------------

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

        # ---------------------------------
        # Resources
        # ---------------------------------

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

        # ---------------------------------
        # Agents
        # ---------------------------------

        for i, agent in enumerate(
            self.agents
        ):

            x, y = self.positions[agent]

            center = (
                x * self.cell_size
                + self.cell_size // 2,

                y * self.cell_size
                + self.cell_size // 2,
            )

            radius = (
                self.cell_size // 3
            )

            pygame.draw.circle(
                self.window,
                self.agent_colors[i],
                center,
                radius,
            )

        # ---------------------------------
        # HUD
        # ---------------------------------

        hud_y = (
            self.grid_size
            * self.cell_size
        )

        pygame.draw.rect(
            self.window,
            (230, 230, 230),
            pygame.Rect(
                0,
                hud_y,
                self.grid_size
                * self.cell_size,
                self.hud_height,
            ),
        )

        # ---------------------------------
        # Agent rewards
        # ---------------------------------

        for i, agent in enumerate(
            self.agents
        ):

            text = (
                f"{agent}: "
                f"{self.total_rewards[agent]:.0f}"
            )

            surface = self.font.render(
                text,
                True,
                self.agent_colors[i],
            )

            x_position = 10 + (
                i * 145
            )

            self.window.blit(
                surface,
                (
                    x_position,
                    hud_y + 10,
                ),
            )

        # ---------------------------------
        # Step
        # ---------------------------------

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

        # ---------------------------------
        # Resource count
        # ---------------------------------

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

        # ---------------------------------
        # Display
        # ---------------------------------

        pygame.display.flip()

        self.clock.tick(
            self.metadata["render_fps"]
        )

    # =========================================================
    # Close
    # =========================================================

    def close(self):

        if self.window is not None:

            pygame.quit()

            self.window = None
