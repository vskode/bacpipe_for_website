"""Guided tours ("How does bacpipe work?") for the bacpipe dashboard.

The header shows "How does bacpipe work?" with two buttons: **Quick Tour**
(the essentials in a few steps) and **Extended Tour** (picks up where the
quick tour ends and explains every remaining control).

Everything you would want to change lives at the top of this file:

* ``QUICK_TOUR`` / ``EXTENDED_TOUR`` - the steps, in order: title, hint text,
  what to highlight, which tab to open first, and how far the highlight
  reaches past the target. ``TOURS`` maps each header button to its steps.
* ``HEADER_LABEL`` / ``TOUR_BUTTONS`` / ``LABELS`` - the header and
  navigation texts.
* ``ACCENT`` / ``DIM`` / ``CARD_WIDTH`` - the look of the highlight.

How it works
------------
The dashboard marks the components a step can point at with a CSS class
(``mark_target(obj, "model-select")`` in ``dashboard.py``). When the visitor
clicks a tour button, a small script (``_TOUR_JS`` below, run in the browser)
finds those components, switches to the step's tab, and covers the rest of the
page with a grey layer that swallows every click, scroll and touch. Only
the highlighted areas stay usable, so visitors can try things out (e.g. click
a point) while the hint is shown.

When a step is on another tab than the first one, the tab's header at the top
is highlighted as well (without being clickable), so it is obvious which page
the tour is on. ``highlight_tab`` on a step overrides that. A step that
switches to another tab first scrolls the page to the top, so the tab bar is
in view; it only scrolls down again if its target would otherwise be off
screen.

A hint text can start the other tour with a link:
``<a href="#" data-bp-tour="extended">Extended Tour</a>``.

Steps can also ask the server to do something when they are shown
(``actions``), e.g. load an example dot into the spectrogram, so the visitor
sees real data instead of an empty plot. The browser only sends the action
*names* (through a hidden text field next to the tour buttons); the dashboard
decides what each name does (``DashBoard.tour_actions``) and ignores unknown
names. Every action is harmless to repeat, so going back and forth through
the tour is fine.

Keep actions cheap. All visitors' sessions share one server process, so while
an action runs, every other visitor's dashboard (including freshly opened
ones) waits. Nothing is ever loaded before the visitor starts a tour, so
the tours never slow down opening the dashboard.

Available actions
-----------------
=====================  =====================================================
action name            what the server does
=====================  =====================================================
show-example-point     load a typical dot of the embedding plot into the
                       spectrogram and audio player, as if it had been
                       clicked (skipped if the visitor already clicked one).
                       Costs the same as one click.
load-predictions       load the integrated classifier's predictions and draw
                       the heatmap, as the "Load predictions from integrated
                       classifier" button does (skipped once loaded). Takes
                       ~0.3s on test_data, ~2s on AnuranSet, so it is listed
                       on the step *before* the heatmap is shown.
=====================  =====================================================

Available targets
-----------------
=====================  ========================  ================================
target name            tab                       what it is
=====================  ========================  ================================
model-select           Single model              the "Model" dropdown
label-select           Single model              the "Label by" dropdown
view-annotated         Single model              "View only annotated?" + radio
                                                 (only with annotations)
embedding-plot         Single model              the 2D embedding plot card
spectrogram            Single model              spectrogram + play buttons
save-selection         Single model              "Save selection to file"
clustering             Single model              clustering results card
probing                Single model              probing performance card
single-model-sidebar   Single model              the whole settings column
contact                every tab                 contact text below the logo
two-models             Two models                the whole page
all-models             All models                everything right of the sidebar
all-models-sidebar     All models                the settings column
all-models-probing     All models                "Probing Metrics" comparison plot
predictions-sidebar    Single Model Predictions  the settings column
clfier-select          Single Model Predictions  "Integrated or linear" radio
run-classifier         Single Model Predictions  "Load predictions" / "Apply" button
species-select         Single Model Predictions  "Select species" dropdown
accumulate-select      Single Model Predictions  day / week / month dropdown
threshold              Single Model Predictions  classification threshold
predictions-settings   Single Model Predictions  "Classification settings" card
predictions-heatmap    Single Model Predictions  the heatmap card
predictions            Single Model Predictions  everything right of the sidebar
save-figure            (every plot)              the "Save Figure" buttons
=====================  ========================  ================================

A target can be narrowed down to an element *inside* it with ``>>`` and a CSS
selector, e.g. ``"contact >> a[href*='github']"`` (the GitHub link) or
``"embedding-plot >> .modebar-btn[data-val='select']"`` (plotly's Box Select
button). Plotly's toolbar normally only shows on hover; a highlighted toolbar
button keeps it visible.

Several targets in one step are highlighted together as one box around all of
them; ``extra_targets`` get boxes of their own. To make a new component
highlightable, wrap it in ``mark_target(obj, "<new-name>")`` where it is built
in ``dashboard.py`` and use ``"<new-name>"`` here.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import panel as pn

logger = logging.getLogger("bacpipe")


# ---------------------------------------------------------------------------
# Editable settings
# ---------------------------------------------------------------------------

HEADER_LABEL = "How does bacpipe work?"
# Header button text per tour (keys of ``TOURS``), in the order shown.
TOUR_BUTTONS = {
    "quick": "Quick Tour",
    "extended": "Extended Tour",
}

LABELS = {
    "next": "Next",
    "back": "Back",
    "finish": "Got it",
    "close": "Close the tour",
}

# Highlight ring and primary button colour (the website's accent blue).
ACCENT = "#7298ff"
# Colour laid over everything that is *not* highlighted.
DIM = "rgba(12, 12, 16, 0.62)"
# Width of the hint card in px (shrinks automatically on narrow screens).
CARD_WIDTH = 340
# After the last step (or when the tour is closed), go back to the tab the
# visitor was on when they started it.
RESTORE_TAB = True
# How far the highlight around a tab header reaches past it, in px (same
# formats as ``TourStep.padding``).
TAB_PADDING = 3
# How long an ``optional`` step waits for its targets before it is skipped,
# in seconds.
OPTIONAL_WAIT = 2.5

# Names a step can list in ``actions`` (see "Available actions" above).
ACTION_EXAMPLE_POINT = "show-example-point"
ACTION_LOAD_PREDICTIONS = "load-predictions"
ACTIONS = (ACTION_EXAMPLE_POINT, ACTION_LOAD_PREDICTIONS)


@dataclass
class TourStep:
    """One step of a tour.

    Parameters
    ----------
    title : str
        heading of the hint card
    text : str
        hint text. Plain HTML is allowed (``<b>``, ``<i>``, ``<br>``, links).
    targets : sequence of str
        targets to highlight (see the table in the module docstring).
        Several are highlighted as one box around all of them. An empty
        list greys out the whole page and centres the card.
    tab : str or None
        dashboard tab to open before highlighting, exactly as the tab is
        labelled. ``None`` stays on the current tab.
    padding : int or tuple
        how far the highlight reaches beyond the targets, in px. One number
        for all sides, ``(vertical, horizontal)``, or
        ``(top, right, bottom, left)`` like CSS. Negative values shrink it.
    placement : str
        where the card goes: ``"auto"`` (first side with room, trying right,
        left, below, above), or ``"right"``, ``"left"``, ``"bottom"``,
        ``"top"`` to try that side first.
    actions : sequence of str
        things the server does when the step is shown, from ``ACTIONS``
        (see "Available actions" in the module docstring). Listing an action
        one step early gives it time to finish before it is needed.
    highlight_tab : bool or None
        also highlight the header of ``tab`` at the top. ``None`` (default)
        does so for every tab except the first one.
    extra_targets : sequence of str
        further targets that are highlighted (and usable) in boxes of their
        own, e.g. the plot a highlighted button works on. The card is placed
        next to ``targets``.
    advance_when : str or None
        a target that, once it exists, moves the tour on to the next step by
        itself, e.g. ``"embedding-plot >> .modebar-btn.active[data-val='select']"``
        (the Box Select button got switched on). *Next* still works too.
    optional : bool
        skip this step when its targets do not show up within
        ``OPTIONAL_WAIT`` seconds (e.g. controls that only exist for
        datasets with annotations).
    scrollable : bool
        let the visitor scroll the page while the step is shown (the
        highlights follow). Normally the grey layer blocks scrolling. Useful
        when a step needs two things that are far apart on a phone, like a
        plot and a button below it.
    """

    title: str
    text: str
    targets: Sequence[str]
    tab: Optional[str] = None
    padding: Union[int, Tuple[int, ...]] = 8
    placement: str = "auto"
    actions: Sequence[str] = ()
    highlight_tab: Optional[bool] = None
    extra_targets: Sequence[str] = ()
    advance_when: Optional[str] = None
    optional: bool = False
    scrollable: bool = False


QUICK_TOUR = [
    TourStep(
        title="1 · Pick a model",
        text=(
            "bacpipe runs your audio through deep-learning models made for "
            "bioacoustics. Each model turns every few seconds of sound into an "
            "<b>embedding</b>: a list of numbers describing what the model "
            "hears.<br><br>Choose whose view of the data you want to see."
        ),
        targets=["model-select"],
        tab="Single model",
    ),
    TourStep(
        title="2 · Colour the points",
        text=(
            "<b>Label by</b> decides how the points are coloured: by the folder "
            "a recording came from, the time of day, annotations, clusterings "
            "or classifier predictions.<br><br>If colours form tidy groups, the "
            "model separates sounds by that property."
        ),
        targets=["label-select"],
        tab="Single model",
    ),
    TourStep(
        title="3 · The embedding plot",
        text=(
            "Every dot is one short audio segment. The embeddings are squashed "
            "into 2D, so dots that are close together sound similar to the "
            "model.<br><br>Zoom, pan, and <b>click any dot</b>; go ahead and "
            "try it now."
        ),
        targets=["embedding-plot"],
        tab="Single model",
        placement="right",
    ),
    TourStep(
        title="4 · Look and listen",
        text=(
            "A clicked dot appears here as a <b>spectrogram</b>, with its file "
            "and label above it. If you haven't clicked one yet, we picked an "
            "example for you: press <b>Play audio</b> to hear it.<br><br>"
            "Normally the sound plays as soon as you click a dot; set "
            "<i>Audio on click</i> to <i>stay silent</i> to turn that off."
        ),
        targets=["spectrogram"],
        tab="Single model",
        placement="left",
        actions=[ACTION_EXAMPLE_POINT],
    ),
    TourStep(
        title="5 · Compare all models",
        text=(
            "The <b>All models</b> page puts every model's embedding next to "
            "each other, coloured by the same label, followed by clustering "
            "and probing scores.<br><br>This shows at a glance which model "
            "separates your data best."
        ),
        targets=["all-models"],
        tab="All models",
        # Loads while the visitor reads this step, so the heatmap is ready
        # on the next two.
        actions=[ACTION_LOAD_PREDICTIONS],
    ),
    TourStep(
        title="6 · Classifier predictions",
        text=(
            "Many models ship with their own classifier, for example BirdNET's "
            "species list. We already loaded its predictions for you; "
            "<b>Load predictions from integrated classifier</b> does the same "
            "for another model.<br><br>"
            "Pick a species, or count by day, week or month, and the "
            "heatmap updates."
        ),
        targets=["predictions-sidebar", "predictions-settings"],
        tab="Single Model Predictions",
    ),
    TourStep(
        title="7 · When was it calling?",
        text=(
            "The heatmap shows when the classifier detected the selected "
            "species, across the dates and the hours of the day. Still empty? "
            "Go back one step and press the load button.<br><br>That's the "
            "quick tour. Want to know what every control does? Take the "
            '<a href="#" data-bp-tour="extended">Extended Tour</a>.'
        ),
        targets=["predictions-heatmap"],
        tab="Single Model Predictions",
    ),
]


EXTENDED_TOUR = [
    TourStep(
        title="Extended tour",
        text=(
            "This tour picks up where the Quick Tour ends and explains every "
            "remaining control of the dashboard.<br><br>Haven't taken the "
            "quick tour yet? Please complete it first: "
            '<a href="#" data-bp-tour="quick">start the Quick Tour</a>.'
        ),
        targets=[],
        tab="Single model",
    ),
    TourStep(
        title="Select many dots at once",
        text=(
            "Besides clicking single dots, you can select a whole group. "
            "Click the <b>Box Select</b> icon (the dashed square) in the "
            "plot's toolbar to switch from zooming to selecting."
        ),
        targets=["embedding-plot >> .modebar-btn[data-val='select']"],
        tab="Single model",
        padding=4,
        placement="bottom",
        advance_when="embedding-plot >> .modebar-btn.active[data-val='select']",
    ),
    TourStep(
        title="Save selection to file",
        text=(
            "Now drag a box over some dots in the embedding plot and press "
            "<b>Save selection to file</b>. This saves a CSV file with the "
            "files and timestamps of all the selected points, so you can go "
            "back to exactly those sounds, e.g. to annotate them."
        ),
        targets=["save-selection"],
        extra_targets=["embedding-plot"],
        tab="Single model",
        # On a phone the plot is far above the button.
        scrollable=True,
    ),
    TourStep(
        title="View only annotated?",
        text=(
            "This dataset comes with annotations. Choose <b>True</b> to hide "
            "every point that has no annotation, so only the labelled sounds "
            "are left in the plot. Works best with <i>Label by</i> set to an "
            "annotation."
        ),
        targets=["view-annotated"],
        tab="Single model",
        optional=True,
    ),
    TourStep(
        title="GitHub",
        text=(
            "This is the main page of the bacpipe repository: the code, how "
            "to install it and the documentation. Ran into a problem or have "
            "an idea? Open an issue there."
        ),
        targets=["contact >> a[href*='github']"],
        tab="Single model",
        padding=(3, 6),
    ),
    TourStep(
        title="Newsletter",
        text=(
            "bacpipe gets updated frequently. To get notified about these "
            "updates and potentially join online sessions about bacpipe, "
            "sign up for the newsletter."
        ),
        targets=["contact >> a[href*='buttondown']"],
        tab="Single model",
        padding=(3, 6),
    ),
    TourStep(
        title="Clustering",
        text=(
            "bacpipe clusters the embeddings and checks how well the clusters "
            "match the labels (folder, time of day or your annotations). The "
            "bars show two agreement scores, <b>AMI</b> and <b>ARI</b>: the "
            "higher, the more the model groups the sounds by that property on "
            "its own, without ever being told about it.<br><br>The ⓘ icon "
            "explains the scores."
        ),
        targets=["clustering"],
        tab="Single model",
    ),
    TourStep(
        title="Probing",
        text=(
            "Probing trains a small classifier on top of the embeddings, using "
            "your annotations: a <i>linear</i> or a <i>k-nearest-neighbour</i> "
            "probe (pick one with <b>Classification Type</b> on the left). The "
            "per-class scores show how well the model tells your sound types "
            "apart. This needs annotations."
        ),
        targets=["probing"],
        tab="Single model",
    ),
    TourStep(
        title="Two models",
        text=(
            "The <b>Two models</b> page shows two single-model views side by "
            "side, each with its own settings. Compare two models on the same "
            "data, or one model coloured by two different labels."
        ),
        targets=["two-models"],
        tab="Two models",
    ),
    TourStep(
        title="Compare probing across models",
        text=(
            "On the <b>All models</b> page, <b>Probing Metrics</b> puts the "
            "probing results of every model into one plot: each line is a "
            "model, each dot its accuracy on one class. The legend ranks the "
            "models by their overall accuracy, best first.<br><br>"
            "Where the lines split apart shows which sound types one model "
            "tells apart better than another. <b>Classification Type</b> on "
            "the left switches between the linear and the k-nearest-neighbour "
            "probe."
        ),
        targets=["all-models-probing"],
        tab="All models",
        # Loads while the visitor reads this step, so the heatmap is ready
        # on the predictions steps.
        actions=[ACTION_LOAD_PREDICTIONS],
    ),
    TourStep(
        title="Predictions",
        text=(
            "The <b>Single Model Predictions</b> page shows what a classifier "
            "detected in your recordings, and when. The settings on the left "
            "decide which predictions you see; the next steps go through them."
        ),
        targets=["predictions"],
        tab="Single Model Predictions",
    ),
    TourStep(
        title="Integrated or linear",
        text=(
            "<b>Integrated</b> uses the classifier that ships with the model, "
            "e.g. BirdNET's species list: nothing to train, but it only knows "
            "the classes it was built with.<br><br><b>Linear</b> uses the "
            "linear probe bacpipe trained on <i>your</i> annotations during "
            "probing, so it predicts your own labels.<br><br>Try it: choose "
            "<b>Linear</b>."
        ),
        targets=["clfier-select"],
        tab="Single Model Predictions",
        advance_when="clfier-select >> input[value='1']:checked",
    ),
    TourStep(
        title="Apply the linear classifier",
        text=(
            "Now press <b>Apply linear classifier</b>. The heatmap below then "
            "counts the classes of your annotations instead of the model's "
            "own species list.<br><br>Switch back to <b>Integrated</b> and "
            "load again to compare the two."
        ),
        targets=["run-classifier"],
        extra_targets=["clfier-select", "predictions-heatmap"],
        tab="Single Model Predictions",
        placement="bottom",
    ),
    TourStep(
        title="Select species",
        text=(
            "Choose which of the classifier's classes the heatmap shows. The "
            "list fills once the predictions are loaded."
        ),
        targets=["species-select"],
        tab="Single Model Predictions",
    ),
    TourStep(
        title="Day, week or month",
        text=(
            "How the detections are grouped along the time axis: per "
            "<b>day</b>, per <b>week</b> (the default) or per <b>month</b>. "
            "Longer bins make long deployments easier to read."
        ),
        targets=["accumulate-select"],
        tab="Single Model Predictions",
    ),
    TourStep(
        title="Threshold",
        text=(
            "Only predictions with a confidence above this value count as "
            "detections (0.5 when left empty). Raise it for fewer but surer "
            "detections, lower it to catch fainter or rarer calls. Type a "
            "value and press Enter."
        ),
        targets=["threshold"],
        tab="Single Model Predictions",
    ),
    TourStep(
        title="Save figure",
        text=(
            "Saves the heatmap as a high-resolution PNG in the results folder. "
            "Every plot of the dashboard has this button."
        ),
        targets=["predictions-heatmap >> .bp-tour-save-figure"],
        tab="Single Model Predictions",
        padding=4,
    ),
    TourStep(
        title="Like what you saw?",
        text=(
            "That's everything the dashboard can do. If you like what you saw, "
            "go ahead and download bacpipe onto your own computer and explore "
            "your own recordings the same way.<br><br>Once downloaded, "
            "everything runs locally, without needing internet access.<br><br>"
            '<a href="https://github.com/bioacoustic-ai/bacpipe" '
            'target="_blank" rel="noopener">Get bacpipe on GitHub</a>'
        ),
        # No target and no tab: a card in the middle of whatever page the
        # tour ended on (so it is part of the phone tour as well).
        targets=[],
    ),
]

# Header buttons (see ``TOUR_BUTTONS``) -> their steps.
TOURS = {
    "quick": QUICK_TOUR,
    "extended": EXTENDED_TOUR,
}


# ---------------------------------------------------------------------------
# Implementation (no need to touch this to change texts or highlights)
# ---------------------------------------------------------------------------

TARGET_CLASS_PREFIX = "bp-tour-"
_PLACEMENTS = ("auto", "right", "left", "bottom", "top")


def mark_target(obj, name):
    """Make ``obj`` highlightable by tour steps that list ``name``.

    Adds the CSS class ``bp-tour-<name>`` to the component's outer element.
    Returns ``obj`` so it can be used inline in a layout definition. A
    ``name`` of ``None`` leaves ``obj`` unmarked.
    """
    if obj is None or name is None:
        return obj
    css_class = TARGET_CLASS_PREFIX + name
    if css_class not in obj.css_classes:
        # Replaced, not mutated: ``css_classes`` defaults are shared.
        obj.css_classes = [*obj.css_classes, css_class]
    return obj


def target_name(spec):
    """The marked component a target refers to (the part before ``>>``)."""
    return spec.split(">>", 1)[0].strip()


def _padding(value):
    """Normalise a padding spec to ``[top, right, bottom, left]``."""
    if isinstance(value, (int, float)):
        return [value] * 4
    value = list(value)
    if len(value) == 1:
        return value * 4
    if len(value) == 2:
        return [value[0], value[1], value[0], value[1]]
    if len(value) == 4:
        return value
    raise ValueError(
        f"Tour padding must be 1, 2 or 4 numbers, got {len(value)}: {value!r}"
    )


def _step_config(step):
    if step.placement not in _PLACEMENTS:
        raise ValueError(
            f"Tour step {step.title!r}: placement must be one of "
            f"{_PLACEMENTS}, got {step.placement!r}"
        )
    for field in ("targets", "actions", "extra_targets"):
        value = getattr(step, field)
        if isinstance(value, str):
            raise ValueError(
                f"Tour step {step.title!r}: {field} must be a list, e.g. "
                f"[{value!r}]"
            )
    specs = [*step.targets, *step.extra_targets]
    if step.advance_when is not None:
        specs.append(step.advance_when)
    for spec in specs:
        if not isinstance(spec, str) or not target_name(spec):
            raise ValueError(
                f"Tour step {step.title!r}: bad target {spec!r}; use a "
                f"target name, optionally followed by '>> <css selector>'"
            )
    unknown = [a for a in step.actions if a not in ACTIONS]
    if unknown:
        raise ValueError(
            f"Tour step {step.title!r}: unknown actions {unknown}; "
            f"available are {list(ACTIONS)}"
        )
    if step.highlight_tab not in (None, True, False):
        raise ValueError(
            f"Tour step {step.title!r}: highlight_tab must be None, True or "
            f"False, got {step.highlight_tab!r}"
        )
    return {
        "title": step.title,
        "text": step.text,
        "targets": list(step.targets),
        "extra_targets": list(step.extra_targets),
        "tab": step.tab,
        "padding": _padding(step.padding),
        "placement": step.placement,
        "actions": list(step.actions),
        "highlight_tab": step.highlight_tab,
        "advance_when": step.advance_when,
        "optional": bool(step.optional),
        "scrollable": bool(step.scrollable),
    }


def tour_config(available_tabs=None, tours=None):
    """Build the JSON-serialisable config the browser script runs on.

    Steps whose ``tab`` is not in ``available_tabs`` are dropped, so the phone
    layout (which only has the single-model page) gets shorter tours instead
    of steps that can never be shown.

    Parameters
    ----------
    available_tabs : list of str or None
        labels of the tabs the layout has; ``None`` keeps every step
    tours : dict or None
        tour name -> list of ``TourStep``, by default ``TOURS``
    """
    tours = TOURS if tours is None else tours
    steps_by_tour = {}
    for name, steps in tours.items():
        if available_tabs is not None:
            steps = [s for s in steps if s.tab is None or s.tab in available_tabs]
            # Tab-less steps (like the closing card) alone are no tour.
            if not any(s.tab is not None for s in steps):
                steps = []
        steps_by_tour[name] = [_step_config(s) for s in steps]
    return {
        "tours": steps_by_tour,
        "labels": dict(LABELS),
        "prefix": TARGET_CLASS_PREFIX,
        "restore_tab": RESTORE_TAB,
        "tab_padding": _padding(TAB_PADDING),
        "optional_wait_ms": int(OPTIONAL_WAIT * 1000),
        "css": _TOUR_CSS
        .replace("__ACCENT__", ACCENT)
        .replace("__DIM__", DIM)
        .replace("__CARD_WIDTH__", str(int(CARD_WIDTH))),
    }


def run_actions(message, actions):
    """Run the actions named in a message the tour script sent.

    The script writes ``"<name>,<name>|<counter>"`` into the hidden bridge
    field; the counter only makes every message a new value. Names that are
    not in ``actions`` are ignored (anyone can type into the browser console),
    and a failing action is logged instead of breaking the session.

    Returns the names that were run.
    """
    names = (message or "").split("|", 1)[0]
    ran = []
    for name in filter(None, (n.strip() for n in names.split(","))):
        func = actions.get(name)
        if func is None:
            logger.warning(f"Tour: ignoring unknown action {name!r}")
            continue
        try:
            func()
        except Exception:
            logger.exception(f"Tour action {name!r} failed")
        ran.append(name)
    return ran


# On phones the question is hidden so the two buttons fit next to the title.
_HEADER_LABEL_CSS = """
:host {
  font-weight: 700;
  font-size: 1.25rem;
  white-space: nowrap;
  text-decoration: underline;
  text-underline-offset: 4px;
  text-decoration-thickness: 2px;
}
@media (max-width: 600px) { :host { display: none !important; } }
"""


def make_tour(available_tabs=None, actions=None, tours=None):
    """Return the "How does bacpipe work?" header with one button per tour.

    The row also holds a hidden text field, the "bridge" the browser script
    uses to ask the server for a step's ``actions``.

    Parameters
    ----------
    available_tabs : list of str or None
        labels of the tabs this layout has; steps on other tabs are skipped.
        ``None`` keeps every step.
    actions : dict or None
        maps action names (``ACTIONS``) to functions without arguments that
        do them. Steps may list actions that are missing here; those are
        skipped.
    tours : dict or None
        tour name -> list of ``TourStep``, by default ``TOURS``. Every tour
        needs a button text in ``TOUR_BUTTONS``.

    Returns
    -------
    pn.Row
        the label (``row[0]``), one button per tour in ``TOUR_BUTTONS`` order,
        and the hidden bridge field (``row[-1]``)
    """
    config = tour_config(available_tabs, tours)
    actions = dict(actions or {})
    # Hidden widgets still exist in the browser, so the script can set the
    # value and Panel sends the change to this session's server side.
    bridge = pn.widgets.TextInput(value="", visible=False, margin=0)
    bridge.param.watch(lambda event: run_actions(event.new, actions), "value")

    buttons = []
    for name, label in TOUR_BUTTONS.items():
        if name not in config["tours"]:
            continue
        button = pn.widgets.Button(
            name=label,
            button_type="light",
            align="center",
            margin=(4, 4),
            visible=bool(config["tours"][name]),
        )
        # Buttons and bridge must be in the same layout so the click
        # callback can get hold of the bridge's browser-side model.
        button.js_on_click(args={"bridge": bridge}, code=tour_script(config, name))
        buttons.append(button)
    header_label = pn.pane.HTML(
        HEADER_LABEL,
        align="center",
        margin=(4, 6, 4, 10),
        stylesheets=[_HEADER_LABEL_CSS],
        visible=any(b.visible for b in buttons),
    )
    return pn.Row(header_label, *buttons, bridge, margin=0, align="center")


def pick_example_point(fig):
    """Choose a dot of an embedding figure to show when nobody clicked one.

    Takes the largest group (trace) of dots and the dot closest to its
    median position: a typical sound of the most common label, rather than an
    outlier. Returns it in the shape plotly click events have
    (``{"customdata": [...]}``), or ``None`` if the figure has no clickable
    dots (e.g. the "Loading..." placeholder).
    """
    best = None
    for trace in getattr(fig, "data", None) or ():
        customdata = getattr(trace, "customdata", None)
        x, y = getattr(trace, "x", None), getattr(trace, "y", None)
        if customdata is None or x is None or y is None:
            continue
        try:
            xy = np.column_stack(
                [np.asarray(x, dtype=float), np.asarray(y, dtype=float)]
            )
        except (TypeError, ValueError):
            continue
        n = min(len(xy), len(customdata))
        if n and (best is None or n > best[0]):
            best = (n, xy[:n], customdata)
    if best is None:
        return None
    n, xy, customdata = best
    finite = np.isfinite(xy).all(axis=1)
    if not finite.any():
        return None
    centre = np.median(xy[finite], axis=0)
    dist = np.where(finite, np.linalg.norm(xy - centre, axis=1), np.inf)
    row = np.asarray(customdata, dtype=object)[int(np.argmin(dist))]
    # Plain Python values, like a real click event delivers them.
    return {"customdata": [v.item() if hasattr(v, "item") else v for v in row]}


def tour_script(config, tour="quick"):
    """Return the browser script that starts ``tour`` from ``config``."""
    # ``json.dumps`` escapes everything that could end the JS literal early, and
    # ``ensure_ascii`` (the default) also escapes U+2028/U+2029.
    return _TOUR_JS.replace(
        "__BACPIPE_TOUR_CONFIG__", json.dumps(config)
    ).replace("__BACPIPE_TOUR_NAME__", json.dumps(tour))


# Styles for the overlay. Everything sits in the normal document (not in a
# Bokeh shadow root), so plain selectors work.
_TOUR_CSS = """
.bp-tour {
  position: fixed; inset: 0; z-index: 2147483000; pointer-events: none;
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}
/* One grey layer over the whole page. Its clip-path cuts the highlighted
   areas out, and clicks only land where it is drawn, so the cut-outs stay
   usable while the rest of the page swallows every click. */
.bp-tour-shade {
  position: fixed; inset: 0;
  background: __DIM__; pointer-events: auto; cursor: default;
  touch-action: none;
}
/* Steps with ``scrollable``: swipes on the grey layer scroll the page. */
.bp-tour-scrollable .bp-tour-shade { touch-action: pan-x pan-y; }
/* Invisible cover over the highlighted tab header: visible, not clickable. */
.bp-tour-shield {
  position: fixed; left: 0; top: 0; width: 0; height: 0;
  pointer-events: auto; cursor: default; touch-action: none; display: none;
}
.bp-tour-ring {
  position: fixed; border: 2px solid __ACCENT__; border-radius: 8px;
  box-shadow: 0 0 0 4px rgba(114, 152, 255, 0.28); pointer-events: none;
  box-sizing: border-box; display: none;
}
.bp-tour-ring.bp-tour-ring-tab { border-radius: 6px; }
.bp-tour-card {
  position: fixed; left: 0; top: 0; pointer-events: auto;
  width: __CARD_WIDTH__px; max-width: calc(100vw - 24px); box-sizing: border-box;
  background: #fff; color: #222; border-radius: 10px;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
  padding: 14px 18px 14px; font-size: 14px; line-height: 1.5;
}
.bp-tour-animating .bp-tour-ring,
.bp-tour-animating .bp-tour-card {
  transition: left .3s ease, top .3s ease, width .3s ease, height .3s ease;
}
/* Only animates between shapes with the same number of cut-outs; otherwise
   the browser simply jumps to the new shape. */
.bp-tour-animating .bp-tour-shade { transition: clip-path .3s ease; }
.bp-tour-head {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 4px;
}
.bp-tour-count {
  font-size: 12px; font-weight: 700; letter-spacing: 1px; color: #888;
}
.bp-tour-close {
  border: 0; background: none; font-size: 22px; line-height: 1;
  color: #888; cursor: pointer; padding: 0 2px;
}
.bp-tour-close:hover { color: #222; }
.bp-tour-title { margin: 0 0 6px; font-size: 17px; font-weight: 700; color: #111; }
.bp-tour-text a { color: __ACCENT__; font-weight: 700; }
.bp-tour-nav {
  display: flex; justify-content: flex-end; gap: 8px; margin-top: 14px;
}
.bp-tour-nav button {
  border-radius: 6px; padding: 7px 14px; font-size: 14px; cursor: pointer;
  border: 1px solid #ccc; background: #fff; color: #333; min-height: 36px;
}
.bp-tour-nav button.bp-tour-next {
  background: __ACCENT__; border-color: __ACCENT__; color: #fff; font-weight: 700;
}
.bp-tour-nav button[disabled] { visibility: hidden; }
.bp-tour-nav button:focus-visible, .bp-tour-close:focus-visible {
  outline: 2px solid __ACCENT__; outline-offset: 2px;
}
"""


# Runs in the visitor's browser on every click of a tour button. The engine
# is created once per page and reused; the config (tours, texts, CSS) and the
# name of the tour to start are inlined from Python by ``make_tour``.
_TOUR_JS = r"""
const CONFIG = __BACPIPE_TOUR_CONFIG__;
const TOUR = __BACPIPE_TOUR_NAME__;

if (!window.__bacpipeTour) {
  window.__bacpipeTour = (function () {
    let cfg = null, steps = [], idx = 0, dir = 1, active = false;
    let ui = null, targets = [], extras = [], searchTimer = null, rafId = null;
    let lastKey = "", startTab = null, stepToken = 0;
    // tabEl: the highlighted tab header (or null); bridge: the hidden text
    // field's model that carries action names to the server.
    let tabEl = null, bridge = null, actionCount = 0;
    // advanceTimer: polls a step's ``advance_when``; revealed: plotly
    // toolbars forced visible for this step, with their previous opacity.
    let advanceTimer = null, revealed = [];
    // Bottom edge of headers stuck to the top of the window (the template's
    // sticky navbar): whatever scrolls underneath is hidden by them.
    let coverTop = 0, lastVw = 0, lastTabLookup = 0;

    // Bokeh renders every component inside its own shadow root, so a plain
    // querySelector cannot see most of the dashboard.
    function deepQueryAll(selector, root, out) {
      if (root.shadowRoot) deepQueryAll(selector, root.shadowRoot, out);
      root.querySelectorAll(selector).forEach((el) => out.push(el));
      root.querySelectorAll("*").forEach((el) => {
        if (el.shadowRoot) deepQueryAll(selector, el.shadowRoot, out);
      });
      return out;
    }

    // Inactive tabs and collapsed cards are display:none -> zero size.
    function isShown(el) {
      if (!el.isConnected) return false;
      const r = el.getBoundingClientRect();
      return r.width > 0 && r.height > 0;
    }

    function uniqueShown(els) {
      const out = [];
      els.forEach((el) => { if (isShown(el) && !out.includes(el)) out.push(el); });
      return out;
    }

    // "name" or "name >> css selector >> ...": the marked component, or
    // elements inside it (searched through shadow roots too).
    function resolve(spec) {
      const parts = spec.split(">>").map((p) => p.trim());
      let els = uniqueShown(deepQueryAll("." + cfg.prefix + parts[0], document, []));
      parts.slice(1).forEach((sel) => {
        const inner = [];
        els.forEach((el) => deepQueryAll(sel, el, inner));
        els = uniqueShown(inner);
      });
      return els;
    }

    function findTargets(specs) {
      const found = [];
      specs.forEach((spec) => {
        resolve(spec).forEach((el) => { if (!found.includes(el)) found.push(el); });
      });
      return found;
    }

    function tabHeaders() {
      return deepQueryAll(".bk-tab", document, []);
    }

    function activeTabLabel() {
      const tab = tabHeaders().find((t) => t.classList.contains("bk-active"));
      return tab ? tab.textContent.trim() : null;
    }

    // Clicking the header is exactly what a visitor does, so Panel builds the
    // tab's content on the server as usual (the tabs are ``dynamic``).
    function switchTab(label) {
      if (!label) return;
      const tab = tabHeaders().find((t) => t.textContent.trim() === label);
      if (tab && !tab.classList.contains("bk-active")) tab.click();
    }

    function el(tag, cls, parent) {
      const node = document.createElement(tag);
      if (cls) node.className = cls;
      if (parent) parent.appendChild(node);
      return node;
    }

    function swallow(e) {
      e.preventDefault();
      e.stopPropagation();
    }

    // The element the page scrolls in: the nearest scrollable ancestor of
    // the step's target (across shadow roots), else the window. The
    // Bootstrap template scrolls ``#main`` rather than the window.
    function pageScroller() {
      for (let node = targets[0] || tabHeaders()[0]; node; ) {
        if (node.nodeType === 1) {
          const oy = getComputedStyle(node).overflowY;
          if ((oy === "auto" || oy === "scroll")
              && node.scrollHeight > node.clientHeight) return node;
        }
        node = node.parentElement
          || (node.getRootNode && node.getRootNode().host) || null;
      }
      return null;
    }

    // Clicks on the grey layer never reach the page. Scrolling only does on
    // ``scrollable`` steps: touch scrolls natively (see the CSS), the mouse
    // wheel is handed to the page's scroller.
    function blockOrScroll(e) {
      const step = active && steps[idx];
      if (step && step.scrollable) {
        if (e.type === "touchstart" || e.type === "touchmove") return;
        if (e.type === "wheel") {
          swallow(e);
          const scroller = pageScroller();
          if (scroller) scroller.scrollBy(e.deltaX, e.deltaY);
          else window.scrollBy(e.deltaX, e.deltaY);
          return;
        }
      }
      swallow(e);
    }

    function build() {
      let style = document.getElementById("bp-tour-style");
      if (!style) {
        style = el("style");
        style.id = "bp-tour-style";
        document.head.appendChild(style);
      }
      style.textContent = cfg.css;

      const root = el("div", "bp-tour");
      const shade = el("div", "bp-tour-shade", root);
      const shield = el("div", "bp-tour-shield", root);
      [shade, shield].forEach((b) => {
        ["click", "dblclick", "mousedown", "mouseup", "pointerdown",
         "pointerup", "touchstart", "touchmove", "wheel", "contextmenu"]
          .forEach((type) => b.addEventListener(type, blockOrScroll, { passive: false }));
      });
      // rings[0] goes around ``targets``, the others around ``extras``.
      const rings = [el("div", "bp-tour-ring", root)];
      const tabRing = el("div", "bp-tour-ring bp-tour-ring-tab", root);

      const card = el("div", "bp-tour-card", root);
      card.setAttribute("role", "dialog");
      card.setAttribute("aria-live", "polite");
      // Keep wheel/touch on the card from scrolling the page underneath.
      ["wheel", "touchmove"].forEach((type) =>
        card.addEventListener(type, swallow, { passive: false }));
      const head = el("div", "bp-tour-head", card);
      const count = el("span", "bp-tour-count", head);
      const close = el("button", "bp-tour-close", head);
      close.type = "button";
      close.innerHTML = "&times;";
      close.setAttribute("aria-label", cfg.labels.close);
      close.title = cfg.labels.close;
      const title = el("h3", "bp-tour-title", card);
      const text = el("div", "bp-tour-text", card);
      const nav = el("div", "bp-tour-nav", card);
      const back = el("button", "bp-tour-back", nav);
      back.type = "button";
      back.textContent = cfg.labels.back;
      const next = el("button", "bp-tour-next", nav);
      next.type = "button";

      close.addEventListener("click", () => stop());
      back.addEventListener("click", () => { if (idx > 0) show(idx - 1, -1); });
      next.addEventListener("click", () => goNext());
      // Links like <a data-bp-tour="quick"> in a hint start that tour.
      text.addEventListener("click", (e) => {
        const link = e.target.closest && e.target.closest("[data-bp-tour]");
        if (!link) return;
        swallow(e);
        start(cfg, bridge, link.getAttribute("data-bp-tour"));
      });

      document.body.appendChild(root);
      ui = { root, shade, shield, rings, tabRing, card, count, title, text, back, next };
    }

    function goNext() {
      if (idx < steps.length - 1) show(idx + 1, 1); else stop();
    }

    // Ask the server to do the step's actions (see ``run_actions`` in
    // tour.py). The counter makes every request a new value, so repeating
    // an action (going back and forth) still reaches the server.
    function runActions(step) {
      if (!bridge || !step.actions || !step.actions.length) return;
      actionCount += 1;
      try {
        bridge.value = step.actions.join(",") + "|" + actionCount + "-" + Date.now();
      } catch (e) {}
    }

    function wantsTab(step) {
      if (!step.tab) return false;
      if (step.highlight_tab === true || step.highlight_tab === false) {
        return step.highlight_tab;
      }
      const first = tabHeaders()[0];
      return !!first && first.textContent.trim() !== step.tab;
    }

    function findTab(label) {
      return tabHeaders().find(
        (t) => t.textContent.trim() === label && isShown(t)) || null;
    }

    // Plotly only shows its toolbar while the mouse is over the plot; keep
    // it visible while one of its buttons is highlighted.
    function reveal(els) {
      els.forEach((t) => {
        const bar = t.closest && t.closest(".modebar");
        if (!bar || revealed.some((r) => r.el === bar)) return;
        [bar, ...bar.querySelectorAll(".modebar-group")].forEach((node) => {
          revealed.push({ el: node, opacity: node.style.opacity });
          node.style.opacity = "1";
        });
      });
    }

    function unreveal() {
      revealed.forEach((r) => { r.el.style.opacity = r.opacity; });
      revealed = [];
    }

    function measureCover() {
      coverTop = 0;
      document.body.querySelectorAll("*").forEach((node) => {
        if (node.closest(".bp-tour")) return;
        const pos = getComputedStyle(node).position;
        if (pos !== "fixed" && pos !== "sticky") return;
        const r = node.getBoundingClientRect();
        // Only bars across the top, not dialogs or zero-size helpers.
        if (r.top <= 1 && r.height > 0 && r.height < window.innerHeight / 3
            && r.width > window.innerWidth / 2) {
          coverTop = Math.max(coverTop, r.bottom);
        }
      });
    }

    function show(i, direction) {
      idx = i;
      measureCover();
      dir = direction || 1;
      const step = steps[i];
      const token = ++stepToken;
      targets = [];
      extras = [];
      tabEl = null;
      lastKey = "";
      unreveal();
      clearInterval(advanceTimer);
      ui.root.classList.toggle("bp-tour-scrollable", !!step.scrollable);
      ui.count.textContent = (i + 1) + " / " + steps.length;
      ui.title.textContent = step.title;
      ui.text.innerHTML = step.text;
      ui.back.disabled = i === 0;
      ui.next.textContent = i === steps.length - 1 ? cfg.labels.finish : cfg.labels.next;
      ui.root.classList.add("bp-tour-animating");
      setTimeout(() => {
        if (ui && token === stepToken) ui.root.classList.remove("bp-tour-animating");
      }, 400);
      // Going to another tab: start at the top of the page, so the tab bar
      // (and which tab the tour went to) is in view.
      const changesTab = !!step.tab && !!findTab(step.tab)
        && step.tab !== activeTabLabel();
      switchTab(step.tab);
      if (changesTab) scrollToTop();
      // After the tab click, so the server builds the tab before it starts
      // on (possibly slower) actions.
      runActions(step);
      search(step, token, changesTab);
      watchAdvance(step, token);
      try { ui.next.focus({ preventScroll: true }); } catch (e) {}
    }

    // ``advance_when``: move on once the visitor did what the step asks.
    // Arriving with it already done moves on right away when going forward;
    // going back, the visitor first has to undo and redo it (otherwise Back
    // could never reach this step).
    function watchAdvance(step, token) {
      if (!step.advance_when) return;
      let wasUndone = false, checks = 0;
      advanceTimer = setInterval(() => {
        if (!active || token !== stepToken) { clearInterval(advanceTimer); return; }
        checks += 1;
        const done = resolve(step.advance_when).length > 0;
        if (!done) { wasUndone = true; return; }
        if (wasUndone || (dir > 0 && checks >= 2)) {
          clearInterval(advanceTimer);
          // A moment to see the click land before the highlight moves.
          setTimeout(() => { if (active && token === stepToken) goNext(); }, 350);
        }
      }, 300);
    }

    // Optional steps whose targets never show up are skipped in the
    // direction the visitor was going.
    function skip() {
      const j = idx + dir;
      if (j >= steps.length) stop();
      else if (j >= 0) show(j, dir);
    }

    // Targets on a freshly opened tab only exist once the server has sent its
    // content, so keep looking (fast at first, then slower) until they appear.
    // The card waits in the middle of the greyed-out page meanwhile.
    // The page may scroll in a container rather than the window (the
    // Bootstrap template scrolls ``#main``): reset every scrolled ancestor of
    // the tab bar, across shadow roots, and the window.
    function scrollToTop() {
      const start = tabHeaders()[0] || document.getElementById("main");
      for (let node = start; node; ) {
        if (node.scrollTop > 0) node.scrollTop = 0;
        node = node.parentElement
          || (node.getRootNode && node.getRootNode().host) || null;
      }
      try { window.scrollTo({ top: 0, left: 0, behavior: "auto" }); }
      catch (e) { window.scrollTo(0, 0); }
    }

    // keepTop: the step just switched tabs; stay at the top unless the target
    // would be out of sight there.
    function search(step, token, keepTop) {
      clearTimeout(searchTimer);
      const started = Date.now();
      const tick = () => {
        if (!active || token !== stepToken) return;
        // The tab header usually exists long before the tab's content.
        if (!tabEl && wantsTab(step)) tabEl = findTab(step.tab);
        const found = findTargets(step.targets);
        if (found.length) {
          targets = found;
          extras = findTargets(step.extra_targets || [])
            .filter((e) => !found.includes(e));
          reveal(targets.concat(extras));
          bringIntoView(keepTop);
          return;
        }
        if (step.optional && Date.now() - started > cfg.optional_wait_ms) {
          skip();
          return;
        }
        // The tab bar may not have existed yet on the first attempt.
        if (Date.now() - started > 800) switchTab(step.tab);
        searchTimer = setTimeout(tick, Date.now() - started < 4000 ? 150 : 700);
      };
      if (step.targets.length) tick();
      else if (wantsTab(step)) tabEl = findTab(step.tab);
    }

    function unionRect(els) {
      let top = Infinity, left = Infinity, bottom = -Infinity, right = -Infinity;
      els.forEach((t) => {
        const r = t.getBoundingClientRect();
        top = Math.min(top, r.top);
        left = Math.min(left, r.left);
        bottom = Math.max(bottom, r.bottom);
        right = Math.max(right, r.right);
      });
      return { top, left, bottom, right };
    }

    function bringIntoView(keepTop) {
      if (!targets.length) return;
      const r = unionRect(targets);
      const vh = window.innerHeight;
      if (r.top >= 0 && r.bottom <= vh) return;
      // Partly visible is enough when the tab bar should stay in view.
      if (keepTop && r.top < vh * 0.75) return;
      const fits = r.bottom - r.top < vh * 0.85;
      targets[0].scrollIntoView({ block: fits ? "center" : "start", behavior: "smooth" });
    }

    function clamp(v, lo, hi) {
      return Math.max(lo, Math.min(hi, v));
    }

    function setBox(node, left, top, width, height) {
      node.style.left = left + "px";
      node.style.top = top + "px";
      node.style.width = Math.max(0, width) + "px";
      node.style.height = Math.max(0, height) + "px";
    }

    // A rectangle grown by padding p ([top, right, bottom, left]) and
    // clipped to the window; null when it is (scrolled) out of view.
    function toHole(r, p, vw, vh) {
      const hole = {
        top: Math.round(clamp(r.top - p[0], coverTop, vh)),
        right: Math.round(clamp(r.right + p[1], 0, vw)),
        bottom: Math.round(clamp(r.bottom + p[2], 0, vh)),
        left: Math.round(clamp(r.left - p[3], 0, vw)),
      };
      if (hole.bottom - hole.top < 4 || hole.right - hole.left < 4) return null;
      return hole;
    }

    function overlaps(a, b) {
      return a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
    }

    // Overlapping cut-outs would cancel each other out: merge them until
    // none overlap.
    function mergeHoles(holes) {
      holes = holes.slice();
      for (let i = 0; i < holes.length; i++) {
        for (let j = i + 1; j < holes.length; j++) {
          if (overlaps(holes[i], holes[j])) {
            const a = holes[i], b = holes[j];
            holes[i] = {
              top: Math.min(a.top, b.top), right: Math.max(a.right, b.right),
              bottom: Math.max(a.bottom, b.bottom), left: Math.min(a.left, b.left),
            };
            holes.splice(j, 1);
            i = -1;
            break;
          }
        }
      }
      return holes;
    }

    // The whole window, with every hole cut out. Each hole is reached by a
    // zero-width "bridge" from the corner and traced the other way round
    // than the outline, which leaves it unfilled. Holes must not overlap.
    function shadeClip(holes, vw, vh) {
      const pts = [[0, 0], [vw, 0], [vw, vh], [0, vh], [0, 0]];
      holes.forEach((h) => {
        pts.push([h.left, h.top], [h.left, h.bottom], [h.right, h.bottom],
                 [h.right, h.top], [h.left, h.top], [0, 0]);
      });
      return "polygon(" + pts.map((p) => p[0] + "px " + p[1] + "px").join(", ") + ")";
    }

    function placeRing(ring, hole) {
      if (!hole) { ring.style.display = "none"; return; }
      ring.style.display = "block";
      setBox(ring, hole.left, hole.top, hole.right - hole.left, hole.bottom - hole.top);
    }

    function holeKey(h) {
      return h ? [h.top, h.right, h.bottom, h.left].join(":") : "none";
    }

    function layout() {
      const vw = window.innerWidth, vh = window.innerHeight;
      if (vw !== lastVw) { lastVw = vw; measureCover(); }
      const step = steps[idx];
      const hole = targets.length ? toHole(unionRect(targets), step.padding, vw, vh) : null;
      const extraHoles = extras.map(
        (e) => toHole(e.getBoundingClientRect(), step.padding, vw, vh));
      const tabRect = tabEl ? tabEl.getBoundingClientRect() : null;
      // A tab header scrolled under the sticky navbar is not highlighted.
      const tabHole = tabRect && tabRect.bottom > coverTop + 4
        ? toHole(tabRect, cfg.tab_padding, vw, vh) : null;
      const cw = ui.card.offsetWidth, ch = ui.card.offsetHeight;
      const key = [idx, vw, vh, cw, ch].concat(
        [hole, tabHole].concat(extraHoles).map(holeKey)).join(",");
      if (key === lastKey) return;
      lastKey = key;

      const holes = mergeHoles([hole, tabHole].concat(extraHoles).filter(Boolean));
      ui.shade.style.clipPath = holes.length ? shadeClip(holes, vw, vh) : "none";

      const ringHoles = [hole].concat(extraHoles);
      while (ui.rings.length < ringHoles.length) {
        ui.rings.push(el("div", "bp-tour-ring", ui.root));
      }
      ui.rings.forEach((ring, i) => placeRing(ring, ringHoles[i] || null));
      placeRing(ui.tabRing, tabHole);
      // The tab header is shown, but clicking it would leave the step.
      if (tabHole) {
        ui.shield.style.display = "block";
        setBox(ui.shield, tabHole.left, tabHole.top,
               tabHole.right - tabHole.left, tabHole.bottom - tabHole.top);
      } else {
        ui.shield.style.display = "none";
      }
      placeCard(hole, vw, vh, cw, ch, step.placement);
    }

    function placeCard(hole, vw, vh, cw, ch, placement) {
      const m = 12, gap = 14;
      let x, y;
      if (!hole) {
        x = (vw - cw) / 2;
        y = (vh - ch) / 2;
      } else {
        const fits = {
          right: vw - hole.right - gap >= cw + m,
          left: hole.left - gap >= cw + m,
          bottom: vh - hole.bottom - gap >= ch + m,
          top: hole.top - gap >= ch + m,
        };
        const order = ["right", "left", "bottom", "top"];
        if (placement && placement !== "auto") order.unshift(placement);
        const side = order.find((s) => fits[s]);
        const alignY = clamp(hole.top, m, vh - ch - m);
        const alignX = clamp(hole.left, m, vw - cw - m);
        if (side === "right") { x = hole.right + gap; y = alignY; }
        else if (side === "left") { x = hole.left - gap - cw; y = alignY; }
        else if (side === "bottom") { x = alignX; y = hole.bottom + gap; }
        else if (side === "top") { x = alignX; y = hole.top - gap - ch; }
        // No free side (e.g. a whole page is highlighted): float the card
        // over the bottom-right corner of the highlight.
        else { x = vw - cw - m; y = vh - ch - m; }
      }
      ui.card.style.left = Math.round(clamp(x, m, Math.max(m, vw - cw - m))) + "px";
      ui.card.style.top = Math.round(clamp(y, m, Math.max(m, vh - ch - m))) + "px";
    }

    // Re-measure every frame: plots resize, tabs render and the page may
    // scroll, and the highlight has to follow all of it.
    function loop() {
      if (!active) return;
      rafId = requestAnimationFrame(loop);
      if (targets.length && !targets.every(isShown)) {
        targets = targets.filter(isShown);
        if (!targets.length) { extras = []; search(steps[idx], stepToken, false); }
      }
      if (extras.length && !extras.every(isShown)) extras = extras.filter(isShown);
      // Panel may re-render the tab bar after a tab switch, which detaches
      // the header found earlier: look it up again (not every frame, the
      // search walks the whole page).
      if (tabEl && !isShown(tabEl)) tabEl = null;
      const now = Date.now();
      if (!tabEl && now - lastTabLookup > 500 && wantsTab(steps[idx])) {
        lastTabLookup = now;
        tabEl = findTab(steps[idx].tab);
      }
      layout();
    }

    function onKey(e) {
      if (!active) return;
      if (e.key === "Escape") { swallow(e); stop(); return; }
      // Arrow keys only while the card has focus, so they still work inside
      // a highlighted dropdown or plot.
      const inCard = ui && ui.card.contains(document.activeElement);
      if (!inCard) return;
      if (e.key === "ArrowRight") { swallow(e); ui.next.click(); }
      else if (e.key === "ArrowLeft" && idx > 0) { swallow(e); ui.back.click(); }
    }

    function start(config, bridgeModel, tourName) {
      const tourSteps = (config.tours || {})[tourName] || [];
      if (!tourSteps.length) return;
      // Switching tours from a hint link keeps the tab the visitor started on.
      const keepTab = active ? startTab : null;
      if (active) stop(true);
      cfg = config;
      bridge = bridgeModel || null;
      steps = tourSteps;
      startTab = keepTab || activeTabLabel();
      active = true;
      build();
      document.addEventListener("keydown", onKey, true);
      show(0, 1);
      loop();
    }

    function stop(silent) {
      active = false;
      stepToken++;
      clearTimeout(searchTimer);
      clearInterval(advanceTimer);
      cancelAnimationFrame(rafId);
      unreveal();
      document.removeEventListener("keydown", onKey, true);
      if (ui) { ui.root.remove(); ui = null; }
      targets = [];
      extras = [];
      tabEl = null;
      if (!silent && cfg && cfg.restore_tab && startTab) switchTab(startTab);
    }

    return { start, stop };
  })();
}

// ``bridge`` is passed in by ``make_tour`` (the hidden text field).
window.__bacpipeTour.start(CONFIG, typeof bridge === "undefined" ? null : bridge, TOUR);
"""
