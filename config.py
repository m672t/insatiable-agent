class EnvironmentConfig:
    """
    تنظیمات اصلی Environment.
    """

    grid_size = 20
    num_agents = 4
    num_resources = 15

    max_steps = 500

    render_mode = "human"

    seed = 42


class SimulationConfig:
    """
    تنظیمات مربوط به اجرای Simulation.
    """

    num_episodes = 1

    save_logs = True

    log_directory = "logs"