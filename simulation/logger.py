
import json
from pathlib import Path

import numpy as np


class EpisodeLogger:
    """
    ثبت و ذخیره اطلاعات مربوط به اجرای Episode.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        پاک کردن داده‌های Episode قبلی.
        """

        self.steps = []

    def _to_serializable(self, value):
        """
        تبدیل انواع داده‌های NumPy به انواع قابل ذخیره در JSON.
        """

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, dict):
            return {
                key: self._to_serializable(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                self._to_serializable(item)
                for item in value
            ]

        return value

    def log_step(
        self,
        step,
        actions,
        rewards,
        infos,
        internal_states=None,
    ):
        """
        ثبت اطلاعات یک Step.

        علاوه بر Action و Reward،
        وضعیت داخلی Agentها نیز ذخیره می‌شود.
        """

        step_data = {
            "step": int(step),

            "actions": self._to_serializable(
                actions
            ),

            "rewards": {
                agent: float(reward)
                for agent, reward in rewards.items()
            },

            "infos": {},

            "internal_states": {},
        }

        # -----------------------------
        # ثبت اطلاعات Environment
        # -----------------------------

        for agent_name, info in infos.items():

            position = info.get("position")

            if position is not None:
                position = self._to_serializable(
                    position
                )

            step_data["infos"][agent_name] = {

                "position": position,

                "collected_resource":
                    float(
                        info.get(
                            "collected_resource",
                            0,
                        )
                    ),

                "total_reward":
                    float(
                        info.get(
                            "total_reward",
                            0.0,
                        )
                    ),

                "remaining_resources":
                    int(
                        info.get(
                            "remaining_resources",
                            0,
                        )
                    ),
            }

        # -----------------------------
        # ثبت وضعیت داخلی Agentها
        # -----------------------------

        if internal_states is not None:

            for agent_name, state in internal_states.items():

                step_data["internal_states"][
                    agent_name
                ] = self._to_serializable(
                    state
                )

        self.steps.append(step_data)

    def get_data(self):
        """
        دریافت تمام داده‌های ثبت‌شده.
        """

        return self.steps.copy()

    def get_step_count(self):
        """
        تعداد Stepهای ثبت‌شده.
        """

        return len(self.steps)

    def save_json(self, filepath):
        """
        ذخیره داده‌های Episode در فایل JSON.
        """

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            filepath,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.steps,
                file,
                ensure_ascii=False,
                indent=2,
            )
