import pytest

from maxplotlib.colors.colors import Color
from maxplotlib.linestyle.linestyle import Linestyle


@pytest.mark.parametrize(
    ("style", "expected"),
    [
        ("solid", "solid"),
        ("dashed", "dashed"),
        ("dotted", "dotted"),
        ("dashdot", "dashdot"),
    ],
)
def test_named_linestyles_are_preserved(style, expected):
    assert Linestyle(style).to_matplotlib() == expected


def test_custom_linestyle_pattern_is_converted_to_matplotlib_dash_tuple():
    linestyle = Linestyle("dash pattern=on 5pt off 2.5pt")

    assert linestyle.to_matplotlib() == (0, (5.0, 2.5))


def test_unknown_linestyle_defaults_to_solid(capsys):
    linestyle = Linestyle("long-dash")

    assert linestyle.to_matplotlib() == "solid"
    assert "Unknown line style: 'long-dash'" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("specification", "expected"),
    [
        ((255, 128, 0), (1.0, 128 / 255, 0.0)),
        ([0.25, 0.5, 0.75], (0.25, 0.5, 0.75)),
        ("#336699", (0.2, 0.4, 0.6)),
        ("navy", (0.0, 0.0, 0.5019607843137255)),
    ],
)
def test_color_specifications_are_converted_to_rgb(specification, expected):
    assert Color(specification).to_rgb() == pytest.approx(expected)


def test_tikz_color_mix_is_interpolated_with_white():
    color = Color("red!20")

    assert color.to_rgb() == pytest.approx((1.0, 0.8, 0.8))


def test_color_output_helpers_include_alpha_and_hex_values():
    color = Color("#336699")

    assert color.to_hex() == "#336699"
    assert color.to_rgba(alpha=0.35) == pytest.approx((0.2, 0.4, 0.6, 0.35))


def test_invalid_color_specification_raises_value_error():
    with pytest.raises(ValueError, match="Invalid color specification"):
        Color("not-a-real-color")
