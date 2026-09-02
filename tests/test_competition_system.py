from agents.competition import CompetitionSystem


def test_competition_detects_other_agents():
    system = CompetitionSystem()

    result = system.evaluate(
        resource_value=20.0,
        resource_location=(5, 0),
        agent_position=(0, 0),
        other_agents=[
            {"position": (4, 0)}
        ],
    )

    assert len(result["agents"]) == 1


def test_competition_measures_competitor_distance():
    system = CompetitionSystem()

    result = system.evaluate(
        resource_value=20.0,
        resource_location=(5, 0),
        agent_position=(0, 0),
        other_agents=[
            {"position": (3, 0)}
        ],
    )

    competitor = result["agents"][0]

    assert competitor[
        "distance_to_resource"
    ] == 2


def test_competition_increases_when_competitor_is_close():
    system = CompetitionSystem()

    far = system.evaluate(
        resource_value=50.0,
        resource_location=(5, 0),
        agent_position=(0, 0),
        other_agents=[
            {"position": (0, 0)}
        ],
    )

    close = system.evaluate(
        resource_value=50.0,
        resource_location=(5, 0),
        agent_position=(0, 0),
        other_agents=[
            {"position": (4, 0)}
        ],
    )

    assert (
        close["competition"]
        > far["competition"]
    )


def test_no_competitor_means_no_competition():
    system = CompetitionSystem()

    result = system.evaluate(
        resource_value=50.0,
        resource_location=(5, 0),
        agent_position=(0, 0),
        other_agents=[],
    )

    assert result["competition"] == 0.0
    assert result["risk"] == 0.0
    assert result["strategy"] == "neutral"


def test_competition_produces_risk_signal():
    system = CompetitionSystem()

    result = system.evaluate(
        resource_value=100.0,
        resource_location=(5, 0),
        agent_position=(0, 0),
        other_agents=[
            {"position": (4, 0)}
        ],
    )

    assert result["risk"] > 0.0


def test_competition_strategy_exists():
    system = CompetitionSystem()

    result = system.evaluate(
        resource_value=50.0,
        resource_location=(5, 0),
        agent_position=(0, 0),
        other_agents=[
            {"position": (4, 0)}
        ],
    )

    assert result["strategy"] in {
        "neutral",
        "pursue",
        "avoid",
        "alternative",
    }
