"""Web figures: theme-neutral Plotly JSON for the docs sites."""

import json

import numpy as np
import pytest

go = pytest.importorskip("plotly.graph_objects")

from maxplotlib import Canvas  # noqa: E402
from maxplotlib.web import (  # noqa: E402
    FORMAT_VERSION,
    color_slot,
    web_figure,
    write_web_figure,
)


def test_color_slot_finds_the_default_cycles():
    assert color_slot("#1f77b4") == 0  # Matplotlib C0
    assert color_slot("#FF7F0E") == 1  # upper case
    assert color_slot("rgb(44, 160, 44)") == 2  # C2 as rgb()
    assert color_slot("rgba(214, 39, 40, 0.5)") == 3  # C3 with alpha
    assert color_slot("#EF553B") == 1  # Plotly's colorway
    assert color_slot("#e69f00") == 1  # Okabe-Ito
    assert color_slot("#000") == 7  # Okabe-Ito black, short form
    assert color_slot("black") is None
    assert color_slot("#123456") is None
    assert color_slot(["#1f77b4"]) is None  # one colour per point is not a slot


def test_web_figure_strips_the_look_and_keeps_the_data():
    fig = go.Figure(go.Scatter(x=[0, 1, 2], y=[1, 3, 2], name="a"))
    fig.update_layout(
        width=800,
        height=400,
        title="Energy",
        paper_bgcolor="white",
        template="plotly_dark",
    )
    web = web_figure(fig, alt="A rising line")
    layout = web["layout"]
    assert "template" not in layout
    assert "width" not in layout and "height" not in layout
    assert layout["paper_bgcolor"] == "rgba(0,0,0,0)"
    assert layout["plot_bgcolor"] == "rgba(0,0,0,0)"
    assert layout["meta"]["web"] == {
        "version": FORMAT_VERSION,
        "white_plot": False,
        "title": "Energy",
        "alt": "A rising line",
        "aspect": 2.0,
    }
    assert web["data"][0]["name"] == "a"
    assert "frames" not in web


def test_default_colours_get_slots_and_others_stay():
    fig = go.Figure(
        [
            go.Scatter(y=[1, 2], line={"color": "#ff7f0e"}),  # C1
            go.Scatter(y=[2, 1], line={"color": "black"}, meta="reference"),
            go.Scatter(
                y=[1, 1], mode="markers", marker={"color": "#2ca02c"}, meta="points"
            ),
            go.Scatter(y=[0, 1]),  # no colour: the page's colorway
        ]
    )
    data = web_figure(fig)["data"]
    assert data[0]["meta"] == {"color_slot": 1}
    assert data[1]["meta"] == "reference"
    assert data[2]["meta"] == {"value": "points", "color_slot": 2}
    assert "meta" not in data[3]


def test_a_matplotlib_conversion_keeps_a_white_plot_area_and_its_meta():
    fig = go.Figure(
        go.Scatter(y=[1, 2]), layout={"meta": {"converted_from": "matplotlib"}}
    )
    layout = web_figure(fig)["layout"]
    assert layout["plot_bgcolor"] == "white"
    assert layout["meta"]["converted_from"] == "matplotlib"
    assert layout["meta"]["web"]["white_plot"] is True
    assert (
        web_figure(fig, white_plot=False)["layout"]["plot_bgcolor"] == "rgba(0,0,0,0)"
    )


def test_frames_keep_their_slots():
    fig = go.Figure(
        go.Scatter(y=[0, 1], line={"color": "#1f77b4"}),
        frames=[
            go.Frame(data=[go.Scatter(y=[1, 0], line={"color": "#1f77b4"})], name="1")
        ],
    )
    web = web_figure(fig)
    assert web["frames"][0]["data"][0]["meta"] == {"color_slot": 0}
    assert web["frames"][0]["name"] == "1"


def test_a_dict_with_arrays_is_accepted():
    figure = {
        "data": [{"type": "scatter", "y": np.arange(3)}],
        "layout": {"title": {"text": "t"}},
    }
    web = web_figure(figure)
    assert web["data"][0]["y"] == [0, 1, 2]
    assert web["layout"]["meta"]["web"]["title"] == "t"


def test_write_web_figure(tmp_path):
    path = write_web_figure(
        go.Figure(go.Bar(y=[1, 2])), tmp_path / "sub" / "bars.plotly.json", aspect=1.5
    )
    web = json.loads(path.read_text())
    assert web["layout"]["meta"]["web"]["aspect"] == 1.5
    assert web["data"][0]["type"] == "bar"


def test_canvas_to_json_and_savefig(tmp_path):
    canvas = Canvas(width="10cm", ratio=0.5)
    x = np.linspace(0, 1, 5)
    canvas.plot(x, x, label="line")
    canvas.set_title("A canvas")
    web = json.loads(canvas.to_json(alt="A straight line"))
    hints = web["layout"]["meta"]["web"]
    assert hints["aspect"] == 2.0
    assert hints["alt"] == "A straight line"
    assert "template" not in web["layout"]
    assert web["layout"]["yaxis"]["automargin"] is True  # room for the tick labels

    canvas.savefig(tmp_path / "canvas.plotly.json", backend="plotly")
    saved = json.loads((tmp_path / "canvas.plotly.json").read_text())
    assert saved["layout"]["meta"]["web"]["version"] == FORMAT_VERSION
    assert saved["data"][0]["name"] == "line"


def test_canvas_aspect_from_figsize_and_the_golden_ratio():
    assert Canvas(figsize=(6, 3))._web_aspect() == 2.0
    assert Canvas(width="10cm")._web_aspect() == pytest.approx(1.618, abs=1e-3)


def test_plotly_canvas_has_room_for_axis_titles_and_no_placeholder_title():
    canvas = Canvas(width="10cm", ratio=0.5)
    canvas.plot([0, 1], [0, 1])
    layout = json.loads(canvas.to_json())["layout"]
    texts = [a.get("text") for a in layout.get("annotations", [])]
    assert "(0, 0)" not in texts  # an untitled subplot has no title
    assert layout["margin"]["b"] == 10 and layout["margin"]["l"] == 10

    canvas.set_xlabel("t")
    canvas.set_ylabel("amplitude")
    layout = json.loads(canvas.to_json())["layout"]
    assert layout["margin"]["b"] == 60 and layout["margin"]["l"] == 70
    assert layout["xaxis"]["title"]["standoff"] == 8
