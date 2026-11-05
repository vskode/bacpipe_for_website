"""
Unit tests for the dashboard's guided tour
(``bacpipe.embedding_evaluation.visualization.tour``).
"""

import json
from pathlib import Path

import panel as pn
import pytest

from bacpipe.embedding_evaluation.visualization import tour

DESKTOP_TABS = [
    "Single model",
    "Two models",
    "All models",
    "Single Model Predictions",
    "Two Model Predictions",
]
DASHBOARD_SRC = (
    Path(tour.__file__).with_name("dashboard.py").read_text(encoding="utf-8")
)
# ``mark_target`` is also used for the "Save Figure" buttons, which are built
# in dashboard_utils.py.
MARKING_SRC = DASHBOARD_SRC + Path(tour.__file__).with_name(
    "dashboard_utils.py"
).read_text(encoding="utf-8")
ALL_STEPS = [step for steps in tour.TOURS.values() for step in steps]


def _specs(step):
    specs = [*step.targets, *step.extra_targets]
    return specs + ([step.advance_when] if step.advance_when else [])


def test_every_step_target_is_marked_in_the_dashboard():
    for step in ALL_STEPS:
        for spec in _specs(step):
            name = tour.target_name(spec)
            assert f'"{name}"' in MARKING_SRC, (
                f"step {step.title!r} highlights {name!r}, but the dashboard "
                f"never calls mark_target(..., {name!r})"
            )


def test_every_step_tab_exists_on_desktop():
    for step in ALL_STEPS:
        assert step.tab is None or step.tab in DESKTOP_TABS, step.title


def test_every_tour_has_a_header_button():
    assert set(tour.TOURS) == set(tour.TOUR_BUTTONS) == {"quick", "extended"}


def test_desktop_keeps_all_steps_and_mobile_only_single_model_ones():
    desktop = tour.tour_config(DESKTOP_TABS)["tours"]
    mobile = tour.tour_config(["Single model"])["tours"]
    assert len(desktop["quick"]) == len(tour.QUICK_TOUR) == 7
    assert len(desktop["extended"]) == len(tour.EXTENDED_TOUR)
    assert [s["tab"] for s in mobile["quick"]] == ["Single model"] * 4
    assert mobile["extended"] and all(
        s["tab"] in ("Single model", None) for s in mobile["extended"]
    )
    # the closing card is part of both layouts
    assert mobile["extended"][-1]["title"] == desktop["extended"][-1]["title"]
    everything = tour.tour_config(None)["tours"]
    assert {k: len(v) for k, v in everything.items()} == {
        k: len(v) for k, v in tour.TOURS.items()
    }


def test_extended_tour_starts_by_pointing_to_the_quick_tour():
    first = tour.EXTENDED_TOUR[0]
    assert first.targets == [] and 'data-bp-tour="quick"' in first.text
    assert 'data-bp-tour="extended"' in tour.QUICK_TOUR[-1].text


def test_box_select_step_waits_for_the_select_button():
    step = next(
        s for s in tour.EXTENDED_TOUR if s.advance_when is not None
    )
    assert "data-val='select'" in step.targets[0]
    assert ".active" in step.advance_when
    after = tour.EXTENDED_TOUR[tour.EXTENDED_TOUR.index(step) + 1]
    assert after.targets == ["save-selection"]
    assert after.extra_targets == ["embedding-plot"]


def test_extended_tour_ends_with_the_download_invitation():
    last = tour.EXTENDED_TOUR[-1]
    assert last.targets == [] and last.tab is None
    assert "github.com/bioacoustic-ai/bacpipe" in last.text
    assert "without needing internet" in last.text


def test_target_name_strips_the_inner_selector():
    assert tour.target_name("contact >> a[href*='github']") == "contact"
    assert tour.target_name("spectrogram") == "spectrogram"


@pytest.mark.parametrize(
    "value, expected",
    [
        (8, [8, 8, 8, 8]),
        ((4,), [4, 4, 4, 4]),
        ((4, 10), [4, 10, 4, 10]),
        ((1, 2, 3, 4), [1, 2, 3, 4]),
        (-6, [-6, -6, -6, -6]),
    ],
)
def test_padding_is_normalised_like_css(value, expected):
    assert tour._padding(value) == expected


def test_bad_step_definitions_fail_loudly():
    with pytest.raises(ValueError):
        tour._padding((1, 2, 3))
    with pytest.raises(ValueError):
        _config([tour.TourStep("t", "x", ["a"], placement="middle")])
    with pytest.raises(ValueError):
        _config([tour.TourStep("t", "x", "model-select")])
    with pytest.raises(ValueError):
        _config([tour.TourStep("t", "x", [" >> .modebar"])])
    with pytest.raises(ValueError):
        _config([tour.TourStep("t", "x", [], extra_targets="spectrogram")])


def _config(steps):
    return tour.tour_config(tours={"quick": steps})


def test_config_is_json_and_inlined_into_the_script():
    config = tour.tour_config(DESKTOP_TABS)
    assert "__" not in config["css"]  # every placeholder was replaced
    code = tour.tour_script(config, "extended")
    assert "__BACPIPE_TOUR" not in code
    assert "const CONFIG = " + json.dumps(config) + ";" in code
    assert 'const TOUR = "extended";' in code


def test_header_has_the_question_and_one_button_per_tour():
    row = tour.make_tour(DESKTOP_TABS)
    label, quick, extended, bridge = row.objects
    assert label.object == tour.HEADER_LABEL and label.visible
    assert [quick.name, extended.name] == ["Quick Tour", "Extended Tour"]
    assert quick.visible and extended.visible
    assert isinstance(bridge, pn.widgets.TextInput) and not bridge.visible


def test_buttons_are_hidden_when_no_step_fits_the_layout():
    row = tour.make_tour(["Some other tab"])
    assert not any(obj.visible for obj in row.objects)


def test_bridge_runs_known_actions_and_ignores_the_rest():
    calls = []
    row = tour.make_tour(
        DESKTOP_TABS,
        actions={
            tour.ACTION_EXAMPLE_POINT: lambda: calls.append("point"),
        },
    )
    bridge = row[-1]
    bridge.value = f"{tour.ACTION_EXAMPLE_POINT}|1-123"
    bridge.value = "load-predictions,rm-rf,|2-456"
    bridge.value = f"{tour.ACTION_EXAMPLE_POINT}|3-789"
    assert calls == ["point", "point"]


def test_a_failing_action_does_not_stop_the_others():
    def broken():
        raise RuntimeError("no audio here")

    calls = []
    ran = tour.run_actions(
        "a,b|1", {"a": broken, "b": lambda: calls.append("b")}
    )
    assert ran == ["a", "b"] and calls == ["b"]
    assert tour.run_actions("", {"a": broken}) == []
    assert tour.run_actions(None, {"a": broken}) == []


def test_steps_only_use_known_actions_and_the_dashboard_provides_them():
    for step in ALL_STEPS:
        assert set(step.actions) <= set(tour.ACTIONS), step.title
    assert "ACTION_EXAMPLE_POINT:" in DASHBOARD_SRC
    with pytest.raises(ValueError):
        _config([tour.TourStep("t", "x", [], actions=["nope"])])
    with pytest.raises(ValueError):
        _config([tour.TourStep("t", "x", [], actions=tour.ACTION_EXAMPLE_POINT)])
    with pytest.raises(ValueError):
        _config([tour.TourStep("t", "x", [], highlight_tab="yes")])


def test_tour_shows_an_example_dot_and_loads_predictions_one_step_early():
    spectrogram = next(s for s in tour.QUICK_TOUR if "spectrogram" in s.targets)
    assert tour.ACTION_EXAMPLE_POINT in spectrogram.actions
    # Predictions load while the step before the predictions tab is shown, so
    # the heatmap is ready when the visitor gets there.
    for steps in tour.TOURS.values():
        first_preds = next(
            i for i, s in enumerate(steps) if s.tab == "Single Model Predictions"
        )
        assert tour.ACTION_LOAD_PREDICTIONS in steps[first_preds - 1].actions
    # Opening the dashboard must never load them: only tour actions may.
    assert "preload_predictions" not in DASHBOARD_SRC


class _Trace:
    def __init__(self, x, y, customdata):
        self.x, self.y, self.customdata = x, y, customdata


class _Figure:
    def __init__(self, *data):
        self.data = data


def test_example_point_is_the_most_typical_dot_of_the_largest_group():
    import numpy as np

    small = _Trace([0.0], [0.0], [["small.wav", 0, 3, 0, "a", "{}"]])
    big = _Trace(
        np.array([10.0, 11.0, 12.0, 50.0]),
        np.array([10.0, 11.0, 12.0, 50.0]),
        np.array(
            [
                ["b0.wav", np.float64(0.0), 3.0, 1, "b", "{}"],
                ["b1.wav", np.float64(3.0), 6.0, 2, "b", "{}"],
                ["b2.wav", np.float64(6.0), 9.0, 3, "b", "{}"],
                ["b3.wav", np.float64(9.0), 12.0, 4, "b", "{}"],
            ],
            dtype=object,
        ),
    )
    point = tour.pick_example_point(_Figure(small, big))
    # median of the big group is (11.5, 11.5): b1 and b2 tie, argmin takes b1
    assert point["customdata"][0] == "b1.wav"
    assert type(point["customdata"][1]) is float  # numpy -> plain Python


def test_no_example_point_without_clickable_dots():
    assert tour.pick_example_point(None) is None
    assert tour.pick_example_point(_Figure()) is None
    image_only = _Figure(type("Image", (), {"customdata": None})())
    assert tour.pick_example_point(image_only) is None
    all_nan = _Trace([float("nan")], [float("nan")], [["x.wav", 0, 1, 0, "a", "{}"]])
    assert tour.pick_example_point(_Figure(all_nan)) is None


def test_config_carries_actions_tab_highlight_and_tab_padding():
    config = tour.tour_config(DESKTOP_TABS)
    assert config["tab_padding"] == tour._padding(tour.TAB_PADDING)
    for name, steps in tour.TOURS.items():
        for step, cfg in zip(steps, config["tours"][name]):
            assert cfg["actions"] == list(step.actions)
            assert cfg["highlight_tab"] is step.highlight_tab
            assert cfg["extra_targets"] == list(step.extra_targets)
            assert cfg["advance_when"] == step.advance_when
            assert cfg["optional"] is step.optional
            assert cfg["scrollable"] is step.scrollable


def test_mark_target_adds_the_class_once():
    column = pn.Column()
    assert tour.mark_target(column, "spectrogram") is column
    tour.mark_target(column, "spectrogram")
    assert column.css_classes.count("bp-tour-spectrogram") == 1
    assert tour.mark_target(None, "spectrogram") is None
    assert tour.mark_target(column, None).css_classes == ["bp-tour-spectrogram"]
