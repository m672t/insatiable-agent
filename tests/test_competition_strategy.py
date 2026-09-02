from agents.competition import CompetitionSystem


def test_low_competition_resource_is_approached():
    system = CompetitionSystem()

    result = system.evaluate_resource(
        agent_position=(0, 0),
        resource={
            "position": (2, 0),
            "value": 10,
        },
        competitors=[],
    )

    assert result["competition"] == 0.0
    assert result["strategy"] == "approach"


def test_close_competitor_creates_race_strategy():
    system = CompetitionSystem()

    result = system.evaluate_resource(
        agent_position=(0, 0),
        resource={
            "position": (5, 0),
            "value": 10,
        },
        competitors=[
            {
                "position": (5, 1),
            }
        ],
    )

    assert result["competition"] > 0.0
    assert result["strategy"] == "race"


def test_high_competition_can_be_avoided():
    system = CompetitionSystem()

    result = system.evaluate_resource(
        agent_position=(0, 0),
        resource={
            "position": (5, 0),
            "value": 10,
        },
        competitors=[
            {
                "position": (5, 0),
            }
        ],
    )

    assert result["competition"] >= 0.75
    assert result["risk"] > 0.0
    assert result["strategy"] == "avoid"


def test_resource_selection_prefers_less_competitive_resource():
    system = CompetitionSystem()

    result = system.choose_resource(
        agent_position=(0, 0),
        resources=[
            {
                "position": (2, 0),
                "value": 10,
            },
            {
                "position": (3, 0),
                "value": 10,
            },
        ],
        competitors=[
            {
                "position": (2, 0),
            }
        ],
    )

    assert result is not None

    selected = result["resource"]

    assert selected["position"] == (3, 0)


def test_competition_priority_contains_risk():
    system = CompetitionSystem()

    result = system.evaluate_resource(
        agent_position=(0, 0),
        resource={
            "position": (2, 0),
            "value": 10,
        },
        competitors=[
            {
                "position": (2, 1),
            }
        ],
    )

    assert "priority" in result
    assert "risk" in result
    assert result["priority"] >= 0.0
