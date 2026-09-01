import math

from environment.competitive_world import CompetitiveWorld
from agents.value_seeking_agent import ValueSeekingAgent


def main():

    print("=" * 60)
    print("DAY 6 - METRICS LOGGER TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # Environment
    # ---------------------------------------------------------

    env = CompetitiveWorld(
        render_mode=None
    )

    env.reset(seed=42)

    # ---------------------------------------------------------
    # Create agents
    # ---------------------------------------------------------

    agents = {
        agent_name: ValueSeekingAgent(
            env=env,
            agent_name=agent_name,
        )
        for agent_name in env.agents
    }

    # ---------------------------------------------------------
    # Metrics logger
    # ---------------------------------------------------------

    # Try to obtain the logger from the environment.
    metrics_logger = getattr(
        env,
        "metrics_logger",
        None,
    )

    if metrics_logger is None:
        raise AssertionError(
            "CompetitiveWorld does not expose "
            "'metrics_logger'."
        )

    # ---------------------------------------------------------
    # Controlled resource metrics
    # ---------------------------------------------------------

    resource_manager = env.resource_manager

    resource_manager.metrics = {
        "spawned": 15,
        "collected": 10,
        "expired": 3,
        "mean_lifetime": 110.0,
        "mean_value": 32.5,
        "mean_collected_value": 25.0,
    }

    # ---------------------------------------------------------
    # Controlled agent metrics
    # ---------------------------------------------------------

    agent_metrics = {
        "agent_0": {
            "total_reward": 15.0,
            "average_reward": 7.5,
            "resources_collected": 1,
            "average_distance_to_resource": 1.5,
            "actions": 2,
            "moves": 2,
            "win_rate": 0.5,
        },
        "agent_1": {
            "total_reward": 50.0,
            "average_reward": 25.0,
            "resources_collected": 1,
            "average_distance_to_resource": 3.5,
            "actions": 2,
            "moves": 2,
            "win_rate": 0.5,
        },
        "agent_2": {
            "total_reward": 0.0,
            "average_reward": 0.0,
            "resources_collected": 0,
            "average_distance_to_resource": 0.0,
            "actions": 0,
            "moves": 0,
            "win_rate": 0.0,
        },
        "agent_3": {
            "total_reward": 0.0,
            "average_reward": 0.0,
            "resources_collected": 0,
            "average_distance_to_resource": 0.0,
            "actions": 0,
            "moves": 0,
            "win_rate": 0.0,
        },
    }

    # ---------------------------------------------------------
    # Controlled competitive metrics
    # ---------------------------------------------------------

    competitive_metrics = {
        "shared_resource_collisions": 2,
        "competition_events": 2,
        "mean_contenders": 2.5,
    }

    # ---------------------------------------------------------
    # Controlled motivation history
    #
    # Two observations:
    #
    # lack:
    #   0.2, 0.4 -> mean = 0.3
    #
    # desire:
    #   0.5, 0.7 -> mean = 0.6
    #
    # satisfaction:
    #   0.1, 0.2 -> mean = 0.15
    #
    # urgency:
    #   0.3, 0.5 -> mean = 0.4
    # ---------------------------------------------------------

    motivation_history = {
        "agent_0": {
            "lack": [
                0.2,
                0.4,
            ],
            "desire": [
                0.5,
                0.7,
            ],
            "satisfaction": [
                0.1,
                0.2,
            ],
            "urgency": [
                0.3,
                0.5,
            ],
        }
    }

    # ---------------------------------------------------------
    # Inject controlled data
    # ---------------------------------------------------------

    # The logger implementation may expose different
    # internal containers. We support the expected names.

    if hasattr(
        metrics_logger,
        "agent_metrics",
    ):
        metrics_logger.agent_metrics = agent_metrics

    if hasattr(
        metrics_logger,
        "competitive_metrics",
    ):
        metrics_logger.competitive_metrics = (
            competitive_metrics
        )

    if hasattr(
        metrics_logger,
        "motivation_history",
    ):
        metrics_logger.motivation_history = (
            motivation_history
        )

    # ---------------------------------------------------------
    # Prefer logger API if available
    # ---------------------------------------------------------

    if hasattr(
        metrics_logger,
        "get_metrics",
    ):
        metrics = metrics_logger.get_metrics()
    elif hasattr(
        metrics_logger,
        "get_all_metrics",
    ):
        metrics = metrics_logger.get_all_metrics()
    else:
        raise AssertionError(
            "MetricsLogger must provide "
            "get_metrics() or get_all_metrics()."
        )

    # ---------------------------------------------------------
    # Print metrics
    # ---------------------------------------------------------

    print()
    print("RESOURCE METRICS")

    resource_output = metrics.get(
        "resource_metrics",
        {},
    )

    print(resource_output)

    print()
    print("AGENT METRICS")

    agent_output = metrics.get(
        "agent_metrics",
        {},
    )

    for agent_name, values in agent_output.items():

        print(agent_name)

        for key, value in values.items():
            print(
                f"  {key}: {value}"
            )

    print()
    print("COMPETITIVE METRICS")

    competitive_output = metrics.get(
        "competitive_metrics",
        {},
    )

    print(
        competitive_output
    )

    print()
    print("MOTIVATION METRICS")

    motivation_output = metrics.get(
        "motivation_metrics",
        {},
    )

    for agent_name, values in motivation_output.items():

        print(agent_name)

        for key, value in values.items():
            print(
                f"  {key}: {value}"
            )

    # ---------------------------------------------------------
    # Resource assertions
    # ---------------------------------------------------------

    assert resource_output["spawned"] == 15
    assert resource_output["collected"] == 10
    assert resource_output["expired"] == 3

    assert math.isclose(
        resource_output["mean_lifetime"],
        110.0,
        abs_tol=1e-9,
    )

    assert math.isclose(
        resource_output["mean_value"],
        32.5,
        abs_tol=1e-9,
    )

    # ---------------------------------------------------------
    # Agent assertions
    # ---------------------------------------------------------

    assert math.isclose(
        agent_output["agent_0"]["total_reward"],
        15.0,
        abs_tol=1e-9,
    )

    assert math.isclose(
        agent_output["agent_0"]["average_reward"],
        7.5,
        abs_tol=1e-9,
    )

    assert (
        agent_output["agent_0"]
        ["resources_collected"]
        == 1
    )

    assert math.isclose(
        agent_output["agent_0"]
        ["average_distance_to_resource"],
        1.5,
        abs_tol=1e-9,
    )

    assert (
        agent_output["agent_0"]
        ["actions"]
        == 2
    )

    assert (
        agent_output["agent_0"]
        ["moves"]
        == 2
    )

    # ---------------------------------------------------------
    # Competitive assertions
    # ---------------------------------------------------------

    assert (
        competitive_output[
            "shared_resource_collisions"
        ]
        == 2
    )

    assert (
        competitive_output[
            "competition_events"
        ]
        == 2
    )

    assert math.isclose(
        competitive_output[
            "mean_contenders"
        ],
        2.5,
        abs_tol=1e-9,
    )

    # ---------------------------------------------------------
    # Motivation assertions
    # ---------------------------------------------------------

    motivation = motivation_output[
        "agent_0"
    ]

    assert math.isclose(
        motivation["mean_lack"],
        0.3,
        abs_tol=1e-9,
    )

    assert math.isclose(
        motivation["mean_desire"],
        0.6,
        abs_tol=1e-9,
    )

    assert math.isclose(
        motivation["mean_satisfaction"],
        0.15,
        abs_tol=1e-9,
    )

    assert math.isclose(
        motivation["mean_urgency"],
        0.4,
        abs_tol=1e-9,
    )

    # ---------------------------------------------------------
    # Steps
    # ---------------------------------------------------------

    assert metrics["steps"] == 2

    # ---------------------------------------------------------
    # Complete
    # ---------------------------------------------------------

    env.close()

    print()
    print("=" * 60)
    print("DAY 6 METRICS ASSERTIONS: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
