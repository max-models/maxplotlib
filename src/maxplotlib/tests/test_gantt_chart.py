"""
Tests for gantt chart functionality across all backends.
"""

import numpy as np
import pytest
from maxplotlib import Canvas


@pytest.fixture
def sample_gantt_data():
    """Sample project task data for testing."""
    return {
        "tasks": ["Planning", "Design", "Development", "Testing", "Deployment"],
        "start_times": [0, 5, 10, 25, 35],
        "durations": [5, 5, 15, 10, 5],
    }


def test_gantt_chart_basic(sample_gantt_data):
    """Test basic gantt chart creation."""
    canvas = Canvas(nrows=1, ncols=1)
    canvas.gantt(
        tasks=sample_gantt_data["tasks"],
        start_times=sample_gantt_data["start_times"],
        durations=sample_gantt_data["durations"],
    )
    
    # Verify subplot was created
    assert len(canvas._subplots) == 1
    subplot = canvas._subplots[(0, 0)]
    
    # Verify gantt chart data was added
    assert len(subplot.line_data) == 1
    gantt_data = subplot.line_data[0]
    assert gantt_data["plot_type"] == "gantt"
    assert gantt_data["tasks"] == sample_gantt_data["tasks"]
    np.testing.assert_array_equal(gantt_data["start_times"], sample_gantt_data["start_times"])
    np.testing.assert_array_equal(gantt_data["durations"], sample_gantt_data["durations"])


def test_gantt_chart_with_kwargs(sample_gantt_data):
    """Test gantt chart with additional kwargs."""
    canvas = Canvas(nrows=1, ncols=1)
    canvas.gantt(
        tasks=sample_gantt_data["tasks"],
        start_times=sample_gantt_data["start_times"],
        durations=sample_gantt_data["durations"],
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
        label="Project Tasks",
    )
    
    subplot = canvas._subplots[(0, 0)]
    gantt_data = subplot.line_data[0]
    assert gantt_data["kwargs"]["color"] == "steelblue"
    assert gantt_data["kwargs"]["alpha"] == 0.7
    assert gantt_data["kwargs"]["edgecolor"] == "black"
    assert gantt_data["kwargs"]["label"] == "Project Tasks"


def test_gantt_chart_matplotlib_backend(sample_gantt_data, tmp_path):
    """Test gantt chart rendering with matplotlib backend."""
    canvas = Canvas(nrows=1, ncols=1, figsize=(10, 6))
    canvas.gantt(
        tasks=sample_gantt_data["tasks"],
        start_times=sample_gantt_data["start_times"],
        durations=sample_gantt_data["durations"],
        color="skyblue",
        edgecolor="navy",
    )
    canvas.set_xlabel("Time (days)")
    canvas.set_title("Project Timeline")
    
    output_file = tmp_path / "test_gantt_matplotlib.png"
    canvas.savefig(str(output_file), backend="matplotlib")
    
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_gantt_chart_plotly_backend(sample_gantt_data, tmp_path):
    """Test gantt chart rendering with plotly backend."""
    canvas = Canvas(nrows=1, ncols=1, figsize=(10, 6))
    canvas.gantt(
        tasks=sample_gantt_data["tasks"],
        start_times=sample_gantt_data["start_times"],
        durations=sample_gantt_data["durations"],
        color="lightblue",
    )
    canvas.set_xlabel("Time (days)")
    canvas.set_title("Project Timeline")
    
    output_file = tmp_path / "test_gantt_plotly.html"
    canvas.savefig(str(output_file), backend="plotly")
    
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_gantt_chart_plotext_backend(sample_gantt_data, tmp_path):
    """Test gantt chart rendering with plotext backend."""
    canvas = Canvas(nrows=1, ncols=1)
    canvas.gantt(
        tasks=sample_gantt_data["tasks"],
        start_times=sample_gantt_data["start_times"],
        durations=sample_gantt_data["durations"],
    )
    canvas.set_xlabel("Time (days)")
    canvas.set_title("Project Timeline")
    
    output_file = tmp_path / "test_gantt_plotext.txt"
    canvas.savefig(str(output_file), backend="plotext")
    
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_gantt_chart_tikzfigure_backend(sample_gantt_data, tmp_path):
    """Test gantt chart rendering with tikzfigure backend."""
    canvas = Canvas(nrows=1, ncols=1)
    canvas.gantt(
        tasks=sample_gantt_data["tasks"],
        start_times=sample_gantt_data["start_times"],
        durations=sample_gantt_data["durations"],
    )
    canvas.set_xlabel("Time (days)")
    canvas.set_title("Project Timeline")
    
    output_file = tmp_path / "test_gantt_tikz.pdf"
    canvas.savefig(str(output_file), backend="tikzfigure")
    
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_gantt_chart_single_task():
    """Test gantt chart with single task."""
    canvas = Canvas(nrows=1, ncols=1)
    canvas.gantt(
        tasks=["Single Task"],
        start_times=[0],
        durations=[10],
    )
    
    subplot = canvas._subplots[(0, 0)]
    assert len(subplot.line_data) == 1
    gantt_data = subplot.line_data[0]
    assert len(gantt_data["tasks"]) == 1


def test_gantt_chart_many_tasks():
    """Test gantt chart with many tasks."""
    n_tasks = 20
    canvas = Canvas(nrows=1, ncols=1)
    canvas.gantt(
        tasks=[f"Task {i+1}" for i in range(n_tasks)],
        start_times=list(range(0, n_tasks * 2, 2)),
        durations=[2] * n_tasks,
    )
    
    subplot = canvas._subplots[(0, 0)]
    gantt_data = subplot.line_data[0]
    assert len(gantt_data["tasks"]) == n_tasks


def test_gantt_chart_overlapping_tasks():
    """Test gantt chart with overlapping tasks."""
    canvas = Canvas(nrows=1, ncols=1)
    canvas.gantt(
        tasks=["Task A", "Task B", "Task C"],
        start_times=[0, 5, 5],  # B and C start at same time
        durations=[10, 8, 6],
    )
    
    subplot = canvas._subplots[(0, 0)]
    gantt_data = subplot.line_data[0]
    assert gantt_data["start_times"][1] == gantt_data["start_times"][2]


def test_gantt_chart_sequential_tasks():
    """Test gantt chart with sequential non-overlapping tasks."""
    canvas = Canvas(nrows=1, ncols=1)
    canvas.gantt(
        tasks=["Phase 1", "Phase 2", "Phase 3"],
        start_times=[0, 10, 20],
        durations=[10, 10, 10],
    )
    
    subplot = canvas._subplots[(0, 0)]
    gantt_data = subplot.line_data[0]
    # Verify tasks are sequential
    assert gantt_data["start_times"][1] == gantt_data["start_times"][0] + gantt_data["durations"][0]


def test_gantt_chart_with_layers():
    """Test gantt chart with different layers."""
    canvas = Canvas(nrows=1, ncols=1)
    
    # Add gantt chart to layer 0
    canvas.gantt(
        tasks=["Task 1", "Task 2"],
        start_times=[0, 5],
        durations=[5, 5],
        layer=0,
        color="blue",
    )
    
    # Add another gantt chart to layer 1
    canvas.gantt(
        tasks=["Task 3", "Task 4"],
        start_times=[0, 5],
        durations=[5, 5],
        layer=1,
        color="red",
    )
    
    subplot = canvas._subplots[(0, 0)]
    assert len(subplot.line_data) == 2
    assert subplot.line_data[0]["layer"] == 0
    assert subplot.line_data[1]["layer"] == 1


def test_gantt_chart_variable_durations():
    """Test gantt chart with varying task durations."""
    canvas = Canvas(nrows=1, ncols=1)
    canvas.gantt(
        tasks=["Short", "Medium", "Long", "Very Long"],
        start_times=[0, 2, 7, 15],
        durations=[2, 5, 8, 20],
    )
    
    subplot = canvas._subplots[(0, 0)]
    gantt_data = subplot.line_data[0]
    durations = gantt_data["durations"]
    assert durations[0] < durations[1] < durations[2] < durations[3]


def test_gantt_chart_zero_duration():
    """Test gantt chart with zero duration task (milestone)."""
    canvas = Canvas(nrows=1, ncols=1)
    canvas.gantt(
        tasks=["Start", "Milestone", "End"],
        start_times=[0, 5, 5],
        durations=[5, 0, 5],  # Milestone has zero duration
    )
    
    subplot = canvas._subplots[(0, 0)]
    gantt_data = subplot.line_data[0]
    assert gantt_data["durations"][1] == 0


def test_gantt_chart_empty_data():
    """Test gantt chart with empty data."""
    canvas = Canvas(nrows=1, ncols=1)
    canvas.gantt(
        tasks=[],
        start_times=[],
        durations=[],
    )
    
    subplot = canvas._subplots[(0, 0)]
    gantt_data = subplot.line_data[0]
    assert len(gantt_data["tasks"]) == 0
    assert len(gantt_data["durations"]) == 0
