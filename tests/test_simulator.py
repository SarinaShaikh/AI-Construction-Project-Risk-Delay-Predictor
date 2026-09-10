from scenarios.simulator import WhatIfSimulator


def sample_data():
    activities = [
        {
            "project_id": "P001",
            "activity_id": "A",
            "planned_duration_days": 10,
        },
        {
            "project_id": "P001",
            "activity_id": "B",
            "planned_duration_days": 20,
        },
        {
            "project_id": "P001",
            "activity_id": "C",
            "planned_duration_days": 10,
        },
    ]

    dependencies = [
        {
            "project_id": "P001",
            "predecessor_id": "A",
            "successor_id": "B",
            "relationship": "FS",
            "lag_days": 0,
        },
        {
            "project_id": "P001",
            "predecessor_id": "B",
            "successor_id": "C",
            "relationship": "FS",
            "lag_days": 0,
        },
    ]

    return activities, dependencies


def create_simulator():
    activities, dependencies = sample_data()

    return WhatIfSimulator(
    activity_rows=activities,
    dependency_rows=dependencies,
)


def test_activity_delay():
    simulator = create_simulator()

    result = simulator.simulate_activity_delay(
        project_id="P001",
        activity_id="A",
        delay_days=5,
    )

    assert result.project_id == "P001"
    assert result.scenario_type == "activity_delay"
    assert result.activity_id == "A"

    assert result.baseline_project_duration == 40
    assert result.scenario_project_duration == 45
    assert result.project_duration_delta == 5


def test_resource_reduction():
    simulator = create_simulator()

    result = simulator.simulate_resource_reduction(
        project_id="P001",
        activity_id="B",
        reduction_percent=20,
    )

    assert result.project_id == "P001"
    assert result.scenario_type == "resource_reduction"
    assert result.activity_id == "B"

    assert result.scenario_project_duration > result.baseline_project_duration
    assert result.project_duration_delta > 0


def test_weather_delay():
    simulator = create_simulator()

    result = simulator.simulate_weather_delay(
        project_id="P001",
        activity_id="A",
        delay_days=7,
    )

    assert result.project_id == "P001"
    assert result.scenario_type == "weather_delay"
    assert result.activity_id == "A"

    assert result.baseline_project_duration == 40
    assert result.scenario_project_duration == 47
    assert result.project_duration_delta == 7


def test_activity_delay_recalculates_cpm():
    simulator = create_simulator()

    result = simulator.simulate_activity_delay(
        project_id="P001",
        activity_id="A",
        delay_days=10,
    )

    # A -> B -> C is the only path,
    # therefore delaying A must increase project duration.
    assert result.scenario_project_duration == 50
    assert result.project_duration_delta == 10


def test_negative_activity_delay_rejected():
    simulator = create_simulator()

    try:
        simulator.simulate_activity_delay(
            project_id="P001",
            activity_id="A",
            delay_days=-5,
        )
        assert False, "Negative delay should raise ValueError"
    except ValueError:
        pass


def test_invalid_resource_reduction_rejected():
    simulator = create_simulator()

    try:
        simulator.simulate_resource_reduction(
            project_id="P001",
            activity_id="B",
            reduction_percent=100,
        )
        assert False, "100% resource reduction should raise ValueError"
    except ValueError:
        pass


def test_missing_activity_rejected():
    simulator = create_simulator()

    try:
        simulator.simulate_activity_delay(
            project_id="P001",
            activity_id="INVALID",
            delay_days=5,
        )
        assert False, "Missing activity should raise ValueError"
    except ValueError:
        pass