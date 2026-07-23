"""
Tests for flame chart functionality across all backends.
"""

import numpy as np
import pytest

from maxplotlib import Canvas


@pytest.fixture
def sample_flame_data():
    """Sample hierarchical profiling data for testing."""
    return {
        "labels": [
            "main()",
            "process_data()",
            "load_file()",
            "parse_json()",
            "validate()",
            "compute()",
            "algorithm_a()",
            "algorithm_b()",
            "save_results()",
        ],
        "parents": [None, 0, 1, 1, 1, 0, 5, 5, 0],
        "values": [100, 40, 10, 15, 15, 50, 25, 25, 10],
        "start_times": [0, 0, 0, 10, 25, 40, 40, 65, 90],
    }


def test_flame_chart_basic(sample_flame_data):
    """Test basic flame chart creation."""
    canvas = Canvas(nrows=1, ncols=1)
    canvas.flame_chart(
        labels=sample_flame_data["labels"],
        parents=sample_flame_data["parents"],
        values=sample_flame_data["values"],
        start_times=sample_flame_data["start_times"],
    )

    # Verify subplot was created
    assert len(canvas._subplots) == 1
    subplot = canvas._subplots[(0, 0)]

    # Verify flame chart data was added
    assert len(subplot.line_data) == 1
    flame_data = subplot.line_data[0]
    assert flame_data["plot_type"] == "flame_chart"
    assert flame_data["labels"] == sample_flame_data["labels"]
    assert flame_data["parents"] == sample_flame_data["parents"]
    np.testing.assert_array_equal(flame_data["values"], sample_flame_data["values"])


def test_flame_chart_without_start_times(sample_flame_data):
    """Test flame chart with auto-computed start times."""
    canvas = Canvas(nrows=1, ncols=1)
    canvas.flame_chart(
        labels=sample_flame_data["labels"],
        parents=sample_flame_data["parents"],
        values=sample_flame_data["values"],
        start_times=None,  # Should be auto-computed
    )

    subplot = canvas._subplots[(0, 0)]
    flame_data = subplot.line_data[0]
    assert flame_data["start_times"] is None


def test_flame_chart_with_kwargs(sample_flame_data):
    """Test flame chart with additional kwargs."""
    canvas = Canvas(nrows=1, ncols=1)
    canvas.flame_chart(
        labels=sample_flame_data["labels"],
        parents=sample_flame_data["parents"],
        values=sample_flame_data["values"],
        start_times=sample_flame_data["start_times"],
        colormap="plasma",
        edgecolor="red",
        label="Test Flame",
    )

    subplot = canvas._subplots[(0, 0)]
    flame_data = subplot.line_data[0]
    assert flame_data["kwargs"]["colormap"] == "plasma"
    assert flame_data["kwargs"]["edgecolor"] == "red"
    assert flame_data["kwargs"]["label"] == "Test Flame"


def test_flame_chart_matplotlib_backend(sample_flame_data, tmp_path):
    """Test flame chart rendering with matplotlib backend."""
    canvas = Canvas(nrows=1, ncols=1, figsize=(10, 6))
    canvas.flame_chart(
        labels=sample_flame_data["labels"],
        parents=sample_flame_data["parents"],
        values=sample_flame_data["values"],
        start_times=sample_flame_data["start_times"],
        colormap="viridis",
    )
    canvas.set_xlabel("Time (ms)")
    canvas.set_ylabel("Stack Depth")
    canvas.set_title("Test Flame Chart")

    output_file = tmp_path / "test_flame_matplotlib.png"
    canvas.savefig(str(output_file), backend="matplotlib")

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_flame_chart_plotly_backend(sample_flame_data, tmp_path):
    """Test flame chart rendering with plotly backend."""
    canvas = Canvas(nrows=1, ncols=1, figsize=(10, 6))
    canvas.flame_chart(
        labels=sample_flame_data["labels"],
        parents=sample_flame_data["parents"],
        values=sample_flame_data["values"],
        start_times=sample_flame_data["start_times"],
        colormap="Plasma",
    )
    canvas.set_xlabel("Time (ms)")
    canvas.set_ylabel("Stack Depth")
    canvas.set_title("Test Flame Chart")

    output_file = tmp_path / "test_flame_plotly.html"
    canvas.savefig(str(output_file), backend="plotly")

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_flame_chart_plotext_backend(sample_flame_data, tmp_path):
    """Test flame chart rendering with plotext backend."""
    canvas = Canvas(nrows=1, ncols=1)
    canvas.flame_chart(
        labels=sample_flame_data["labels"],
        parents=sample_flame_data["parents"],
        values=sample_flame_data["values"],
        start_times=sample_flame_data["start_times"],
    )
    canvas.set_xlabel("Time (ms)")
    canvas.set_ylabel("Stack Depth")
    canvas.set_title("Test Flame Chart")

    output_file = tmp_path / "test_flame_plotext.txt"
    canvas.savefig(str(output_file), backend="plotext")

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_flame_chart_tikzfigure_backend(sample_flame_data, tmp_path):
    """Test flame chart rendering with tikzfigure backend."""
    canvas = Canvas(nrows=1, ncols=1)
    canvas.flame_chart(
        labels=sample_flame_data["labels"],
        parents=sample_flame_data["parents"],
        values=sample_flame_data["values"],
        start_times=sample_flame_data["start_times"],
    )
    canvas.set_xlabel("Time (ms)")
    canvas.set_ylabel("Stack Depth")
    canvas.set_title("Test Flame Chart")

    output_file = tmp_path / "test_flame_tikz.pdf"
    canvas.savefig(str(output_file), backend="tikzfigure")

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_flame_chart_simple_hierarchy():
    """Test flame chart with simple 2-level hierarchy."""
    canvas = Canvas(nrows=1, ncols=1)
    canvas.flame_chart(
        labels=["root", "child1", "child2"],
        parents=[None, 0, 0],
        values=[100, 50, 50],
        start_times=[0, 0, 50],
    )

    subplot = canvas._subplots[(0, 0)]
    assert len(subplot.line_data) == 1
    flame_data = subplot.line_data[0]
    assert len(flame_data["labels"]) == 3


def test_flame_chart_deep_hierarchy():
    """Test flame chart with deep nesting (4 levels)."""
    canvas = Canvas(nrows=1, ncols=1)
    canvas.flame_chart(
        labels=["level0", "level1", "level2", "level3"],
        parents=[None, 0, 1, 2],
        values=[100, 80, 60, 40],
        start_times=[0, 0, 0, 0],
    )

    subplot = canvas._subplots[(0, 0)]
    flame_data = subplot.line_data[0]
    assert len(flame_data["labels"]) == 4


def test_flame_chart_multiple_roots():
    """Test flame chart with multiple root nodes."""
    canvas = Canvas(nrows=1, ncols=1)
    canvas.flame_chart(
        labels=["root1", "root2", "child1", "child2"],
        parents=[None, None, 0, 1],
        values=[50, 50, 30, 30],
        start_times=[0, 50, 0, 50],
    )

    subplot = canvas._subplots[(0, 0)]
    flame_data = subplot.line_data[0]
    assert flame_data["parents"].count(None) == 2


def test_flame_chart_with_layers():
    """Test flame chart with different layers."""
    canvas = Canvas(nrows=1, ncols=1)

    # Add flame chart to layer 0
    canvas.flame_chart(
        labels=["func1", "func2"],
        parents=[None, 0],
        values=[100, 50],
        layer=0,
    )

    # Add another flame chart to layer 1
    canvas.flame_chart(
        labels=["func3", "func4"],
        parents=[None, 0],
        values=[80, 40],
        layer=1,
    )

    subplot = canvas._subplots[(0, 0)]
    assert len(subplot.line_data) == 2
    assert subplot.line_data[0]["layer"] == 0
    assert subplot.line_data[1]["layer"] == 1


def test_flame_chart_empty_data():
    """Test flame chart with empty data."""
    canvas = Canvas(nrows=1, ncols=1)
    canvas.flame_chart(
        labels=[],
        parents=[],
        values=[],
        start_times=[],
    )

    subplot = canvas._subplots[(0, 0)]
    flame_data = subplot.line_data[0]
    assert len(flame_data["labels"]) == 0
    assert len(flame_data["values"]) == 0


def test_flame_chart_single_node():
    """Test flame chart with single root node."""
    canvas = Canvas(nrows=1, ncols=1)
    canvas.flame_chart(
        labels=["main"],
        parents=[None],
        values=[100],
        start_times=[0],
    )

    subplot = canvas._subplots[(0, 0)]
    flame_data = subplot.line_data[0]
    assert len(flame_data["labels"]) == 1
    assert flame_data["parents"][0] is None
