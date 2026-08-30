import panel as pn
import matplotlib
import sys
import seaborn as sns
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger("bacpipe")

import importlib.resources as pkg_resources
import bacpipe.imgs
from .visualize_embeddings import (
    plot_embeddings,
    plot_comparison,
    EmbedAndLabelLoader,
)
from . import tooltips
from .visualize import (
    plot_clusterings,
    clustering_overview,
    plot_overview_results,
)
from .visualize_spectrograms import SpectrogramPlot
from .visualize_predictions import (
    plot_classification_results,
    plot_classification_heatmap,
    PredictionsLoader,
)

import bacpipe.embedding_evaluation.label_embeddings as le
from .dashboard_utils import DashBoardHelper

### plotting settings
sns.set_theme(style="whitegrid")
matplotlib.use("agg")
pn.extension("plotly")

# ---------------------------------------------------------------------------
# Mobile layout
# ---------------------------------------------------------------------------
# Making the dashboard usable on a phone needs two independent pieces:
#
# 1. A viewport meta tag (set via ``meta_viewport`` on the template further
#    down). Panel does not add one by default, and without it every mobile
#    browser lays the page out in a ~980px wide *virtual* viewport and lets the
#    user pan sideways — the "everything is tiny and scrolls sideways" feel.
#    It also means ``window.innerWidth`` is 980 on a phone, so none of the
#    media queries below would ever match.
#
# 2. Media queries that live *inside* the components' shadow roots. Bokeh 3.x
#    renders every Panel layout as an element with its own shadow root, so
#    page-level CSS cannot reach the flex rules that put panels side by side.
#    Panel's ``stylesheets`` parameter injects CSS into that same shadow root,
#    so ``:host`` rules with ``!important`` win on narrow screens.
#
# ``apply_mobile_styles`` walks the finished component tree and attaches the
# fluid rules everywhere, which is what stops fixed pixel widths (the 180px
# sidebar, the 600px classifier path field, ...) from pushing the page sideways.
#
# 3. Sensible defaults for components that do not exist yet.
#    ``enable_mobile_defaults`` puts the base rules on the ``stylesheets`` class
#    default, so panes that Panel rebuilds later (``pn.bind`` returns fresh
#    objects on every update) cannot reintroduce horizontal scrolling.
MOBILE_BREAKPOINT = 900

# Installed as the *class default* of every component's ``stylesheets`` (see
# ``enable_mobile_defaults``) so it also reaches the components Panel builds
# later on: a ``pn.bind`` callback returns brand new objects on every update and
# those never pass through ``apply_mobile_styles``. Only rules that are safe for
# literally every component belong here.
_MOBILE_BASE_CSS = f"""
@media (max-width: {MOBILE_BREAKPOINT}px) {{
  :host {{
    max-width: 100% !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
  }}
  select, textarea,
  .bk-input:not([type="checkbox"]):not([type="radio"]) {{
    width: 100% !important;
    /* < 16px makes iOS Safari zoom into the field on focus, which leaves the
       page scrolled sideways. */
    font-size: 16px !important;
    min-height: 40px !important;
  }}
  .bk-btn {{
    min-height: 40px !important;
    white-space: normal !important;
  }}
}}
"""

_MOBILE_DIRECTION_CSS = f"""
@media (max-width: {MOBILE_BREAKPOINT}px) {{
  :host {{
    flex-direction: column !important;
    flex-wrap: nowrap !important;
    align-self: stretch !important;
    width: auto !important;
    max-width: 100% !important;
    min-width: 0 !important;
  }}
}}
"""

# ``width: auto`` + ``align-self: stretch`` rather than ``width: 100%``: several
# layouts carry margins, and ``100%`` ignores them, which is exactly how a panel
# ends up a dozen pixels wider than the screen.
_MOBILE_ITEM_CSS = f"""
@media (max-width: {MOBILE_BREAKPOINT}px) {{
  :host {{
    flex: 0 0 auto !important;
    align-self: stretch !important;
    width: auto !important;
    max-width: 100% !important;
    min-width: 0 !important;
  }}
}}
"""

# On a phone the plots are what the visitor came for, so the settings column
# (with the logo and contact block appended to it) is pushed to the bottom of
# the stack. Flex ``order`` does that without touching the desktop layout, where
# the media query does not apply and the sidebar stays on the left.
_MOBILE_LAST_CSS = f"""
@media (max-width: {MOBILE_BREAKPOINT}px) {{
  :host {{
    order: 99 !important;
  }}
}}
"""

# Attached to *every* component: neutralises fixed pixel widths so nothing can
# be wider than the phone screen, and lets flex children actually shrink
# (``min-width: 0`` — flex items default to ``min-width: auto``, which is the
# usual reason a single wide child blows up the whole row).
_MOBILE_FLUID_CSS = f"""
@media (max-width: {MOBILE_BREAKPOINT}px) {{
  :host {{
    max-width: 100% !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
  }}
}}
"""

# The tab bar is the one element that cannot shrink: five labels never fit on
# a phone, so let them wrap onto multiple lines instead of scrolling sideways.
# Bigger hit areas make the tabs actually tappable.
_MOBILE_TABS_CSS = f"""
@media (max-width: {MOBILE_BREAKPOINT}px) {{
  :host {{
    max-width: 100% !important;
    min-width: 0 !important;
  }}
  .bk-header {{
    flex-wrap: wrap !important;
    row-gap: 4px !important;
    overflow-x: visible !important;
  }}
  .bk-header .bk-tab {{
    flex: 1 1 auto !important;
    min-width: 0 !important;
    padding: 8px 10px !important;
    font-size: 0.95rem !important;
    white-space: normal !important;
    text-align: center !important;
  }}
}}
"""

# Plotly figures declare their height in the figure layout itself (700px for the
# embedding, 550px for the spectrogram), so the height must be left alone: a CSS
# clamp only shrinks the wrapper and lets the canvas spill over the buttons
# below it. The width is the part that matters on a phone — the panes are
# ``stretch_width``, and Panel's Plotly view relayouts the figure to the width
# the CSS below gives the pane (see the note on plotly's own ``responsive``
# config in ``spectrogram_panel``).
_MOBILE_PLOT_CSS = f"""
@media (max-width: {MOBILE_BREAKPOINT}px) {{
  :host {{
    align-self: stretch !important;
    width: auto !important;
    max-width: 100% !important;
    min-width: 0 !important;
  }}
}}
"""


# Widgets get a touch-friendly treatment on phones: they fill the width of the
# (now single-column) layout, tap targets are at least ~40px high and inputs use
# a 16px font — anything smaller makes iOS Safari zoom the page in on focus,
# which is a classic way to end up scrolled sideways.
_MOBILE_WIDGET_CSS = f"""
@media (max-width: {MOBILE_BREAKPOINT}px) {{
  :host {{
    width: auto !important;
    max-width: 100% !important;
    min-width: 0 !important;
    align-self: stretch !important;
    flex-shrink: 1 !important;
    box-sizing: border-box !important;
  }}
  select, input, textarea, .bk-input {{
    width: 100% !important;
    font-size: 16px !important;
    min-height: 40px !important;
  }}
  button, .bk-btn {{
    min-height: 40px !important;
    font-size: 1rem !important;
    white-space: normal !important;
  }}
}}
"""

# Chrome of the Bootstrap template itself (header, container, main column).
# These elements live in the normal document — not in a shadow root — so they
# are styled through the template's ``raw_css`` instead of ``stylesheets``.
_TEMPLATE_MOBILE_CSS = f"""
html, body {{
  max-width: 100%;
  /* ``clip`` behaves like ``hidden`` but does not turn the vertical axis into a
     scroll container, so the page keeps scrolling on the document itself.
     ``hidden`` is kept first as a fallback for older browsers. */
  overflow-x: hidden;
  overflow-x: clip;
  -webkit-text-size-adjust: 100%;
  /* Always reserve room for the vertical scrollbar. Without it a plot that
     grows just past the viewport height makes the scrollbar appear, which
     narrows the content, which shrinks the plot, which hides the scrollbar
     again — a width oscillation that shows up as a shaking figure. */
  scrollbar-gutter: stable;
}}
#main {{
  /* Same reasoning as above: on desktop this column is the scroll container. */
  scrollbar-gutter: stable;
  /* Prevent a horizontal scrollbar from appearing/disappearing as the plot
     width settles — that back-and-forth is the other way a figure shakes. */
  overflow-x: hidden;
}}
#container {{
  /* The template ships ``vh-100``; ``dvh`` tracks the *visible* viewport so a
     phone's address bar can not cut off the bottom of the dashboard. */
  height: 100dvh !important;
  max-width: 100%;
  overflow-x: hidden;
}}
@media (max-width: {MOBILE_BREAKPOINT}px) {{
  /* Desktop keeps the dashboard inside a fixed-height, internally scrolling
     column. On a phone that nested scroll area feels broken (no momentum, and
     the header eats a chunk of every scroll), so let the page itself scroll:
     a single scroll container with a sticky header. */
  #container {{
    height: auto !important;
    min-height: 100dvh;
    overflow: visible !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
  }}
  #content.row {{
    margin-left: 0 !important;
    margin-right: 0 !important;
    overflow: visible !important;
  }}
  #main {{
    padding-left: 8px !important;
    padding-right: 8px !important;
    overflow-x: hidden !important;
    overflow-y: visible !important;
    max-height: none !important;
  }}
  #header {{
    padding: 6px 12px !important;
    min-height: 0 !important;
  }}
  #header .app-header {{
    display: flex !important;
    flex-wrap: wrap !important;
    align-items: baseline !important;
    gap: 0 6px !important;
  }}
  #header .title {{
    font-size: 1rem !important;
    line-height: 1.3 !important;
  }}
  /* The website already shows a banner above the iframe, so the long subtitle
     would only eat vertical space on a phone. */
  #header .app-header > span.title,
  #header .app-header > a.title:nth-of-type(2) {{
    display: none !important;
  }}
}}
"""


def _add_stylesheet(obj, css):
    """Attach ``css`` to a Panel object's shadow root, once.

    ``stylesheets`` is a list parameter, so it is replaced (not mutated) to
    avoid touching the parameter default shared by every instance.
    """
    sheets = getattr(obj, "stylesheets", None)
    if sheets is None or css in sheets:
        return
    obj.stylesheets = [*sheets, css]


def enable_mobile_defaults():
    """Give every Panel component the mobile base stylesheet by default.

    ``apply_mobile_styles`` can only reach the components that exist when the
    layout is built. Panes driven by ``pn.bind`` throw their content away and
    build new components on every update, so without a default they would render
    at their desktop size again and push the page sideways.

    ``stylesheets`` is declared once on ``Layoutable`` and shared by every
    subclass, so setting the default here covers panes, layouts and widgets
    alike. It is idempotent, which matters because a new dashboard is built for
    every visitor session.
    """
    default = pn.viewable.Layoutable.param.stylesheets.default
    if _MOBILE_BASE_CSS in default:
        return
    pn.viewable.Layoutable.param.stylesheets.default = [*default, _MOBILE_BASE_CSS]


def _mobile_stack_row(*items, **kwargs):
    """Return a ``pn.Row`` that stacks its children vertically on phones.

    Children keep their natural order when stacked, so pass the panels in the
    order you want top-to-bottom on mobile (e.g. the embedding plot before the
    spectrogram). Extra keyword arguments (``sizing_mode``, ...) are forwarded
    to the ``Row``.
    """
    # Passing ``stylesheets`` explicitly replaces the class default installed by
    # ``enable_mobile_defaults``, so the base sheet is repeated here.
    row = pn.Row(
        *items, stylesheets=[_MOBILE_BASE_CSS, _MOBILE_DIRECTION_CSS], **kwargs
    )
    for item in items:
        _add_stylesheet(item, _MOBILE_ITEM_CSS)
    return row


def _mobile_move_last(obj):
    """Send ``obj`` to the bottom of the stack on phones.

    Only has an effect inside a layout that ``_mobile_stack_row`` turns into a
    column on narrow screens; the desktop order is untouched. Returns ``obj`` so
    it can be used inline in a layout definition.
    """
    _add_stylesheet(obj, _MOBILE_LAST_CSS)
    return obj


def _iter_components(obj, seen=None):
    """Yield ``obj`` and every nested component, depth first.

    ``Viewable.select`` is not usable here: ``Accordion.select`` only descends
    into the internal ``Card`` objects it wraps its children in, and those are
    built lazily on first render. At layout build time the cards do not exist
    yet, so ``select()`` silently skips everything inside an accordion — which
    is most of this dashboard. Walking ``objects`` instead is reliable, and
    ``_panels`` is included so already rendered cards are picked up too.
    """
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    yield obj

    children = list(getattr(obj, "objects", None) or [])
    panels = getattr(obj, "_panels", None)
    if isinstance(panels, dict):
        children += list(panels.values())
    for child in children:
        if isinstance(child, pn.viewable.Viewable):
            yield from _iter_components(child, seen)


def apply_mobile_styles(root):
    """Make an already built component tree fit a phone screen.

    Walks every nested component and injects the fluid media query into its
    shadow root. Fixed widths set in Python (the 180px sidebar, the wide
    classifier widgets, ...) stay in place on desktop but can no longer exceed
    the viewport on a phone, which is what removes the horizontal scrolling.

    Parameters
    ----------
    root : panel.viewable.Viewable
        the root of the dashboard layout (e.g. the ``pn.Tabs`` app)
    """
    try:
        objects = list(_iter_components(root))
    except Exception:  # pragma: no cover - defensive, layout must still render
        logger.warning("Could not walk the layout to apply mobile styles.")
        return

    for obj in objects:
        if isinstance(obj, pn.Tabs):
            _add_stylesheet(obj, _MOBILE_TABS_CSS)
        elif isinstance(obj, pn.pane.Plotly):
            _add_stylesheet(obj, _MOBILE_PLOT_CSS)
        elif isinstance(obj, pn.widgets.TooltipIcon):
            # A full-width tooltip icon would be a huge invisible tap target.
            _add_stylesheet(obj, _MOBILE_FLUID_CSS)
        elif isinstance(obj, pn.widgets.Widget):
            _add_stylesheet(obj, _MOBILE_WIDGET_CSS)
        else:
            _add_stylesheet(obj, _MOBILE_FLUID_CSS)


_EMBEDDING_PLAY_ON_CLICK_JS = """
// Play the clicked segment on *this* device only.
//
// The dashboard gives every device its own bokeh session
// (``dash-<dataset>-<deviceId>``, see dashboard.html), so a server-side
// playback trigger would only reach this device anyway. Playback still has to
// start client side: a websocket round-trip does not count as a browser
// gesture, which mobile browsers require before <audio> can play. So this
// callback runs on the ``plotly_event`` in the browser that actually clicked
// a point, and only that browser unpauses its own <audio> element.
//
// ``cb_obj`` is the PlotlyEvent and ``cb_obj.data`` is
// ``{type: "click", data: {points: [...]}}``. The same event fires for hover
// and selection too, so ignore everything but a click on a real point.
const evt = cb_obj && cb_obj.data;
if (!evt || evt.type !== "click") { return; }
if (!autoplay.value) { return; }
const points = evt.data && evt.data.points;
if (!points || !points.length) { return; }

// The server loads the clicked segment in response to this same click and
// pushes the new source to the shared player a moment later. Wait until the
// player value actually changes, then start playback on this device.
const before = player.value;
const started = Date.now();
const timer = setInterval(() => {
  const value = player.value || "";
  if (value.length > 100 && value !== before) {
    player.time = 0;
    player.paused = false;
    clearInterval(timer);
  } else if (Date.now() - started > 8000) {
    clearInterval(timer);
  }
}, 40);
"""


def _first_model(obj):
    """Return the bokeh model Panel built for ``obj``, if any."""
    models = getattr(obj, "_models", None) or {}
    if not models:
        return None
    return next(iter(models.values()))[0]


def _attach_embedding_autoplay(plot_pane, audio_player, autoplay_select):
    """Start playback on the clicking device when an embedding point is chosen.

    The embedding plot is a ``pn.pane.Plotly`` whose bokeh model emits a
    client-side ``plotly_event`` for every plotly interaction. A JS callback
    attached to that event runs in the browser that made the gesture, so
    unpausing the audio player only ever affects that device. (Each device now
    has its own bokeh session too, but the gesture requirement stays: a
    websocket round-trip does not count as a user gesture on mobile.)
    """

    def _on_load():
        from bokeh.models import CustomJS

        plot_model = _first_model(plot_pane)
        player_model = _first_model(audio_player)
        autoplay_model = _first_model(autoplay_select)
        if not (plot_model and player_model and autoplay_model):
            return
        plot_model.js_on_event(
            "plotly_event",
            CustomJS(
                args={"player": player_model, "autoplay": autoplay_model},
                code=_EMBEDDING_PLAY_ON_CLICK_JS,
            ),
        )

    pn.state.onload(_on_load)


class DashBoard(DashBoardHelper):
    """
    Panel dashboard visualizing embeddings, clustering, probing results and
    classifier predictions for one or multiple models.
    """

    def __init__(
        self,
        model_names,
        audio_dir,
        main_results_dir,
        default_label_keys,
        evaluation_task,
        dim_reduction_model,
        dim_reduc_parent_dir,
        **kwargs,
    ):
        """
        Initialize the dashboard and its widgets.

        Parameters
        ----------
        model_names : list
            names of the models to visualize
        audio_dir : pathlib.Path
            directory containing the audio files
        main_results_dir : pathlib.Path
            directory containing the evaluation results
        default_label_keys : list
            default label keys used for coloring
        evaluation_task : str
            evaluation tasks to display (e.g., clustering, probing)
        dim_reduction_model : str
            dimensionality reduction model used for the embeddings
        dim_reduc_parent_dir : pathlib.Path
            parent directory of the reduced embeddings
        **kwargs
            additional keyword arguments (e.g., plot heights, widths)
        """
        self.models = model_names
        self.default_label_keys = default_label_keys
        self.audio_dir = audio_dir
        self.path_func = le.make_set_paths_func(
            audio_dir, main_results_dir, dim_reduc_parent_dir, **kwargs
        )
        self.label_by = default_label_keys.copy()
        if (
            self.path_func(model_names[0]).preds_path
        ).exists() and not "default_classifier" in self.label_by:
            clfier_paths = list(
                self.path_func(model_names[0]).preds_path.rglob(
                    "*_classifier_annotations.csv"
                )
            )
            if len(clfier_paths) > 0:
                if clfier_paths[0].exists():
                    self.label_by += ["default_classifier"]
        self.plot_path = self.path_func(model_names[0]).plot_path.parent.parent
        self.dim_reduc_parent_dir = dim_reduc_parent_dir

        self.ground_truth = None
        ground_truth_files = list(
            le.get_paths(model_names[0]).labels_path.glob("ground_truth*")
        )
        if len(ground_truth_files) > 0:
            labels = []
            if len(ground_truth_files) > 0:
                for gt_file in ground_truth_files:
                    if gt_file.suffix == ".csv":
                        ground_truth_df = le.get_ground_truth(
                            model_names[0],
                            file_path=gt_file,
                            return_type="dataframe",
                        )
                    elif gt_file.suffix == ".npy":
                        ground_truth_df = le.get_ground_truth(
                            model_names[0],
                            file_path=gt_file,
                            return_type="array",
                        )
                    labels.append(gt_file.stem.replace("ground_truth_", ""))
            self.ground_truth = True
            self.label_by += labels

        if (
            len(list(le.get_paths(model_names[0]).clust_path.glob("*.npy")))
            > 0
        ):
            self.label_by += [
                clustering['name'] for clustering in bacpipe.settings.clust_configs.values()
                if clustering['bool'] is True
            ]

        self.evaluation_task = evaluation_task
        self.dim_reduction_model = dim_reduction_model
        self.widget_width = 100
        self.vis_loader = EmbedAndLabelLoader(
            dim_reduction_model=dim_reduction_model,
            default_label_keys=default_label_keys,
            **kwargs,
        )

        self.interactive_embedding_plot = True

        self.model_select = dict()
        self.label_select = dict()
        self.noise_select = dict()
        self.clfier_select = dict()
        self.species_select = dict()
        self.accumulate_select = dict()
        self.class_select = dict()
        self.embed_plot = dict()

        self.embed_save_button = dict()
        self.embed_notification = dict()

        self.interactive_embed_plot = dict()
        self.spectrogram_plot_panel = dict()
        self.spec_plot_obj = dict()
        self._trigger_spec_obj_update = dict()
        # Client-side audio player per widget, plus the radio button that
        # decides whether clicking a point in the embedding plot plays the
        # corresponding segment right away.
        self.audio_player = dict()
        self.autoplay_select = dict()

        self.class_options = dict()
        self.preds_data = dict()
        self.clfier_path = dict()
        self.clfier_thresh = dict()
        self.btn_run_clfier = dict()
        self.progress_bar = dict()
        self.trigger_classification = dict()
        self.loading_test_placeholder = dict()

        self.heatmap_plot = dict()
        self.kwargs = kwargs

    @staticmethod
    def get_audio_dir():
        """
        Resolve the audio directory for the current visitor.

        The website embeds the dashboard in an iframe and lets each visitor
        pick a dataset through the ``?audio_dir=`` query parameter. That value
        is read from the Panel session args; when it is absent the configured
        default (``bacpipe.config.audio_dir``) is used instead. The returned
        path is relative to the website's submodule checkout
        (``sites/bacpipe/bacpipe_for_website``).
        """
        audio_dir = pn.state.session_args.get("audio_dir", [None])[0]
        if audio_dir:
            clean_string = (
                audio_dir.decode("utf-8")
                if isinstance(audio_dir, bytes)
                else audio_dir
            )
            return "../public/assets/audio/" + clean_string
        return bacpipe.config.audio_dir

    def embedding_panel(self, widget_idx=0):
        """
        Build the 2D embedding plot panel for a widget.

        Parameters
        ----------
        widget_idx : int
            index of the widget

        Returns
        -------
        tuple of (str, pn.Column)
            panel title and the column containing the plot
        """
        if not self.interactive_embedding_plot:
            embedding_plot = self.init_plot(
                # self.init_interactive_plot(
                "embed",
                plot_embeddings,
                widget_idx,
                loader=self.vis_loader,
                model_name=self.model_select[widget_idx],
                label_by=self.label_select[widget_idx],
                ground_truth=self.ground_truth,
                dim_reduction_model=self.dim_reduction_model,
                remove_noise=(
                    self.noise_select[widget_idx]
                    if len(self.noise_select.keys()) > 0
                    else False
                ),
                dashboard=True,
                dashboard_idx=widget_idx,
            )
        else:

            self.init_interactive_embed_plot(widget_idx)

            # Callback to update plot when any selector changes, while preserving accordion state.
            def update_plot_on_change(event):
                """
                Redraw the embedding plot when a selector value changes.

                Parameters
                ----------
                event : object or None
                    panel parameter change event, or None on first render
                """
                self.update_main_plot(
                    "interactive_embed",
                    plot_embeddings,
                    widget_idx,
                    loader=self.vis_loader,
                    model_name=self.model_select[widget_idx].value,
                    label_by=self.label_select[widget_idx].value,
                    ground_truth=self.ground_truth,
                    dim_reduction_model=self.dim_reduction_model,
                    remove_noise=(
                        self.noise_select[widget_idx].value
                        if widget_idx in self.noise_select
                        and self.noise_select[widget_idx] is not None
                        else False
                    ),
                    dashboard=True,
                    dashboard_idx=widget_idx,
                )

            # Only attach watchers once per widget (check if already attached)
            if not hasattr(
                self.model_select[widget_idx], "_embedding_watchers_attached"
            ):
                self.model_select[widget_idx].param.watch(
                    update_plot_on_change, "value"
                )
                self.label_select[widget_idx].param.watch(
                    update_plot_on_change, "value"
                )
                if (
                    widget_idx in self.noise_select
                    and self.noise_select[widget_idx] is not None
                ):
                    self.noise_select[widget_idx].param.watch(
                        update_plot_on_change, "value"
                    )
                # Mark that watchers have been attached
                self.model_select[widget_idx]._embedding_watchers_attached = (
                    True
                )

            # Render plot with current widget values (every time, to refresh display when navigating tabs)
            update_plot_on_change(None)

            # Embed plot reference (no longer using pn.bind to avoid accordion collapse).
            embedding_plot = self.interactive_embed_plot[widget_idx]
        return (
            "2D Embedding Plot",
            pn.Column(
                embedding_plot,
                self.embed_save_button[widget_idx],
                self.embed_notification[widget_idx],
            ),
        )

    def spectrogram_panel(self, widget_idx=0):
        """
        Build the spectrogram plot panel for a widget.

        Parameters
        ----------
        widget_idx : int
            index of the widget

        Returns
        -------
        tuple of (str, pn.Column)
            panel title and the column containing the plot
        """
        self.spectrogram_plot_panel[widget_idx] = pn.pane.Plotly(
            SpectrogramPlot.dummy_image(title=""),
            height=self.kwargs.get("spectrogram_plot_height"),
            # Responsive width + fixed height, with the figure on
            # ``autosize=True``. Panel's Plotly view relayouts the figure to the
            # pane's clientWidth on every layout pass (``after_layout`` ->
            # ``Plotly.relayout({width, height})``); with ``autosize=True`` that
            # relayout is a no-op for the rendered size, so Bokeh's layout does
            # not feed back into itself and the plot does not oscillate (the old
            # ``autosize=False`` + ``stretch_width`` combo "shivered").
            # Do *not* use ``styles={"display": "contents"}`` here: it removes
            # the pane's own box so the plot overflows into the accordions and
            # buttons below. And do *not* pass ``config={"responsive": True}``:
            # it installs a second ResizeObserver that fights the pane the same
            # way.
            sizing_mode="stretch_width",
        )

        embedding_info_dialogue = pn.widgets.StaticText(
            value="",
            sizing_mode="stretch_width",
        )

        self.spec_plot_obj[widget_idx] = SpectrogramPlot(
            self.audio_dir,
            self.vis_loader,
            self.model_select[widget_idx],
            embedding_info_dialogue,
            **self.kwargs,
        )

        self._trigger_spec_obj_update[widget_idx] = pn.bind(
            (self.spec_plot_obj[widget_idx]._update_spec_obj),
            self.model_select[widget_idx],
        )

        # Client-side audio player. The site runs the dashboard on a headless
        # server (no sounddevice device), so audio is streamed to the browser
        # instead. It starts out empty (and hidden) and appears once a segment
        # has been loaded, because the native controls are the one playback
        # trigger every phone browser allows. Playback is never started from
        # the server (see the ``autoplay=False`` note below) — it is always
        # triggered by a gesture in the visitor's own browser.
        audio_player = pn.pane.Audio(
            name="Audio playback",
            visible=False,
            sizing_mode="stretch_width",
            # Deliberately NOT autoplay. Each device now has its own bokeh
            # session (``dash-<dataset>-<deviceId>``), so nothing is broadcast
            # between visitors anymore — but ``autoplay=True`` or a server-side
            # ``paused=False`` still would not start playback: mobile browsers
            # only allow audio that is triggered by a real user gesture, and a
            # websocket round-trip does not count as one. Playback is therefore
            # always started by the visitor's own tap (the "Play audio" button
            # or the native <audio> controls), which only ever affects the
            # device that made the gesture.
            autoplay=False,
        )
        self.spec_plot_obj[widget_idx].audio_player = audio_player
        self.audio_player[widget_idx] = audio_player

        # Clicking a point in the embedding plot loads its spectrogram and, by
        # default, also loads the matching segment and starts it playing on the
        # device that clicked (see _attach_embedding_autoplay — the playback
        # trigger is client side, never broadcast from the server). Anyone
        # browsing in a quiet room can switch that off here; the "Play audio"
        # button and the player's own controls keep working either way.
        autoplay_select = pn.widgets.RadioBoxGroup(
            name="Audio on click",
            options={"play segment": True, "stay silent": False},
            value=True,
            inline=True,
            sizing_mode="stretch_width",
        )
        self.autoplay_select[widget_idx] = autoplay_select
        # When enabled, clicking an embedding point also starts playback on the
        # device that clicked (per-client; see _attach_embedding_autoplay).
        _attach_embedding_autoplay(
            self.interactive_embed_plot.get(widget_idx),
            audio_player,
            autoplay_select,
        )
        # On a phone this label + radio pair stacks vertically instead of
        # sitting side by side, so the "Audio on click:" text never overlaps
        # the radio buttons.
        autoplay_setting = _mobile_stack_row(
            pn.pane.Markdown(
                "**Audio on click:**",
                margin=(0, 5, 0, 10),
            ),
            autoplay_select,
            sizing_mode="stretch_width",
        )

        def play_current_audio(event):
            spec_plot = self.spec_plot_obj[widget_idx]
            if not spec_plot.update_audio_player():
                embedding_info_dialogue.visible = True
                embedding_info_dialogue.value = (
                    "Click a point in the embedding plot first, "
                    "then press play."
                )
                return
            # Playback itself is started client side by this button's
            # ``js_on_click`` (a real user gesture, which is what mobile
            # browsers require). We intentionally do NOT set
            # ``audio_player.paused`` here: a server-side unpause would not
            # count as a user gesture, so mobile browsers would still block it.

        play_audio_button = pn.widgets.Button(
            name="Play audio", button_type="primary"
        )
        play_audio_button.on_click(play_current_audio)
        # Runs in the browser during the tap itself, which is what mobile
        # browsers require to start playback - a websocket round-trip does not
        # count as a user gesture. Guarded, so an empty player is not asked to
        # play (the value is a WAV data URI, empty until a segment is loaded).
        play_audio_button.js_on_click(
            args={"player": audio_player},
            code=(
                "if ((player.value || '').length > 100) {"
                " player.time = 0; player.paused = false; }"
            ),
        )
        save_selection_dialogue = pn.widgets.StaticText(
            value="",
            # A fixed pixel width here used to bubble up as a ``min-width`` on
            # the surrounding accordion card (bokeh derives a container's
            # minimum size from its children), which made the whole dashboard
            # wider than a phone screen.
            sizing_mode="stretch_width",
        )


        save_selection_button = pn.widgets.Button(
            name="Save selection to file", button_type="primary"
        )
        save_selection_button.on_click(
            lambda x: self.save_selected_points(
                x, save_selection_dialogue, widget_idx
            )
        )
        save_selection_dialogue.visible = False

        return (
            "Spectrogram",
            pn.Column(
                embedding_info_dialogue,
                self.spectrogram_plot_panel[widget_idx],
                save_selection_dialogue,
                _mobile_stack_row(play_audio_button, save_selection_button),
                autoplay_setting,
                audio_player,
            ),
        )

    def clustering_panel(self, widget_idx):
        """
        Build the clustering results panel for a widget.

        Parameters
        ----------
        widget_idx : int
            index of the widget

        Returns
        -------
        tuple of (str, pn.Column)
            panel title and the column containing the clustering plot
        """
        return (
            "Clustering Results",
            (
                pn.Column(
                    pn.widgets.TooltipIcon(value=tooltips.clustering),
                    (
                        self.plot_widget(
                            plot_clusterings,
                            path_func=self.path_func,
                            model_name=self.model_select[widget_idx],
                            label_by=self.label_select[widget_idx],
                            no_noise=(
                                self.noise_select[widget_idx]
                                if len(self.noise_select.keys()) > 0
                                else False
                            ),
                        )
                        if "clustering" in self.evaluation_task
                        else pn.pane.Markdown(
                            "No clustering task specified. "
                            "Please check the config file."
                        )
                    ),
                )
            ),
        )

    def probing_panel(self, widget_idx):
        """
        Build the probing performance panel for a widget.

        Parameters
        ----------
        widget_idx : int
            index of the widget

        Returns
        -------
        tuple of (str, pn.Column)
            panel title and the column containing the probing plot
        """
        return (
            "Probing Performance",
            (
                pn.Column(
                    pn.widgets.TooltipIcon(value=tooltips.probing),
                    (
                        self.plot_widget(
                            plot_classification_results,
                            path_func=self.path_func,
                            task_name=self.class_select[widget_idx],
                            model_name=self.model_select[widget_idx],
                            return_fig=True,
                        )
                        if "probing" in self.evaluation_task
                        else pn.pane.Markdown(
                            "No probing task specified. "
                            "Please check the config file."
                        )
                    ),
                )
            ),
        )

    def model_page(self, widget_idx, single_model=False):
        """
        Build the single model dashboard page.

        Parameters
        ----------
        widget_idx : int
            index of the widget
        single_model : bool
            if True, panels are laid out for a single model

        Returns
        -------
        pn.Row
            row containing the sidebar and the model content
        """
        sidebar = self.make_sidebar(widget_idx, model=True)
        title_string = "Model Dashboard for {}".format
        accordion_title = pn.bind(title_string, self.model_select[widget_idx])
        if single_model:
            # The embedding plot sits next to the spectrogram on wide screens.
            # ``_mobile_stack_row`` stacks them on phones with the embedding
            # plot first (on top) and the spectrograms below it.
            data_panels = _mobile_stack_row(
                pn.Accordion(
                    self.embedding_panel(widget_idx),
                    active=[0],
                    sizing_mode="stretch_width",
                ),
                pn.Accordion(
                    self.spectrogram_panel(widget_idx),
                    self.clustering_panel(widget_idx),
                    self.probing_panel(widget_idx),
                    active=[0, 1, 2],
                    sizing_mode="stretch_width",
                ),
            )
        else:
            data_panels = pn.Accordion(
                self.embedding_panel(widget_idx),
                self.spectrogram_panel(widget_idx),
                self.clustering_panel(widget_idx),
                self.probing_panel(widget_idx),
                active=[0, 1, 2, 3],
                sizing_mode="stretch_width",
            )

        main_content = pn.Column(
            pn.widgets.StaticText(
                value=accordion_title,
                styles={
                    "font-size": "1.5em",  # Equivalent to a standard H2
                    "font-weight": "bold",
                    "margin-top": "0px",
                    "margin-bottom": "15px",
                },
            ),
            data_panels,
            sizing_mode="stretch_width",
        )

        # Side by side on desktop. On a phone the plots come first and the
        # settings (plus the logo and contact block) sit at the bottom.
        return _mobile_stack_row(
            _mobile_move_last(sidebar),
            main_content,
            sizing_mode="stretch_width",
        )

    def all_models_page(self, widget_idx):
        """
        Build the dashboard page comparing all models.

        Parameters
        ----------
        widget_idx : int
            index of the widget

        Returns
        -------
        pn.Row
            row containing the sidebar and the all-models content
        """
        sidebar = self.make_sidebar(widget_idx, model=False, all_models=True)

        main_content = pn.Column(
            pn.pane.Markdown("## All Models Dashboard"),
            pn.Accordion(
                (
                    "Embedding Comparison",
                    self.init_plot(
                        "embed",
                        plot_comparison,
                        widget_idx,
                        loader=self.vis_loader,
                        plot_path=self.plot_path,
                        models=self.models,
                        dim_reduction_model=self.dim_reduction_model,
                        label_by=self.label_select[widget_idx],
                        remove_noise=(
                            self.noise_select[widget_idx]
                            if len(self.noise_select.keys()) > 0
                            else False
                        ),
                        default_label_keys=self.default_label_keys,
                        dashboard=True,
                    ),
                ),
                (
                    "Clustering Overview",
                    (
                        pn.Column(
                            pn.widgets.TooltipIcon(value=tooltips.clustering),
                            (
                                self.plot_widget(
                                    clustering_overview,
                                    path_func=self.path_func,
                                    model_list=self.models,
                                    label_by=self.label_select[widget_idx],
                                    no_noise=(
                                        self.noise_select[widget_idx]
                                        if len(self.noise_select.keys()) > 0
                                        else False
                                    ),
                                    **self.kwargs,
                                )
                                if "clustering" in self.evaluation_task
                                else pn.pane.Markdown(
                                    "No clustering task specified. "
                                    "Please check the config file."
                                )
                            ),
                        )
                    ),
                ),
                (
                    "Probing Metrics",
                    (
                        self.plot_widget(
                            plot_overview_results,
                            plot_path=None,
                            metrics=None,
                            task_name=self.class_select[widget_idx],
                            path_func=self.path_func,
                            model_list=self.models,
                            return_fig=True,
                        )
                        if "probing" in self.evaluation_task
                        else pn.pane.Markdown(
                            "No probing task specified. "
                            "Please check the config file."
                        )
                    ),
                ),
                active=[0, 1, 2],
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )

        # Side by side on desktop. On a phone the plots come first and the
        # settings (plus the logo and contact block) sit at the bottom.
        return _mobile_stack_row(
            _mobile_move_last(sidebar),
            main_content,
            sizing_mode="stretch_width",
        )

    def apply_clfier_page(self, widget_idx):
        """
        Build the page for applying a classifier to model predictions.

        Parameters
        ----------
        widget_idx : int
            index of the widget

        Returns
        -------
        pn.Row
            row containing the sidebar and the classification content
        """
        self.class_options[widget_idx] = []
        sidebar = self.make_sidebar(
            widget_idx, model=True, classifier_page=True
        )

        # input box where i can input the path to the linear classifier
        self.clfier_path[widget_idx] = pn.widgets.TextInput(
            name="Path to Linear Probe",
            placeholder=(
                self.path_func(self.models[0]).probe_path / "linear_probe.pt"
            ).as_posix(),
            # Fluid instead of a fixed 600px: a fixed width propagates as a
            # ``min-width`` onto the enclosing card and forces the page to be
            # wider than a phone screen.
            sizing_mode="stretch_width",
            max_width=600,
            max_length=800,
            visible=False,
        )

        self.clfier_thresh[widget_idx] = pn.widgets.TextInput(
            name="Threshold for classification",
            placeholder="0.5",
            width=80,
        )

        self.btn_run_clfier[widget_idx] = pn.widgets.Button(
            # name='Apply linear classifier',
            name="Load predictions from integrated classifier",
            # ``max_width`` (rather than ``width``) keeps the desktop size but
            # lets the button shrink to the phone's width.
            sizing_mode="stretch_width",
            max_width=300,
            height=30,
        )

        self.progress_bar[widget_idx] = pn.indicators.Progress(
            value=0,
            max=100,
            bar_color="primary",
            sizing_mode="stretch_width",
            max_width=500,
        )

        self.loading_test_placeholder[widget_idx] = pn.widgets.StaticText(
            name="Preparing classification", value=""
        )

        self.clfier_select[widget_idx].param.watch(
            lambda x: self.change_input_options(x, widget_idx=widget_idx),
            "value",
        )

        self.preds_data[widget_idx] = PredictionsLoader(
            self.vis_loader,
            self.path_func,
            self.models,
            panel_selection=self.species_select[widget_idx],
            progress_bar=self.progress_bar[widget_idx],
            loading_pane=self.loading_test_placeholder[widget_idx],
        )
        self.btn_run_clfier[widget_idx].on_click(
            lambda x: self.update_main_plot(
                "heatmap",
                plot_classification_heatmap,
                widget_idx=widget_idx,
                event=x,
                predictions_loader=self.preds_data[widget_idx],
                model=self.model_select[widget_idx],
                accumulate_by=self.accumulate_select[widget_idx],
                species=self.species_select[widget_idx],
                threshold=self.clfier_thresh[widget_idx],
                clfier_path=self.clfier_path[widget_idx],
                clfier_type=self.clfier_select[widget_idx],
                **self.kwargs,
            )
        )

        main_content = pn.Column(
            pn.pane.Markdown("## Classifier Predictions"),
            pn.Accordion(
                (
                    "Classification settings",
                    pn.Column(
                        # trigger_input_options,
                        self.clfier_path[widget_idx],
                        # after that show me the classes that this
                        # linear classifier will classify
                        pn.widgets.StaticText(
                            name="Classes",
                            value=pn.bind(
                                self.preds_data[widget_idx].get_classes,
                                self.clfier_path[widget_idx],
                            ),
                        ),
                        # input section to give a threshold for classification
                        self.clfier_thresh[widget_idx],
                        # button to click run
                        self.btn_run_clfier[widget_idx],
                        # placeholder textbox to show that something
                        # is happening while waiting on embeddings to load
                        self.loading_test_placeholder[widget_idx],
                        # progbar
                        self.progress_bar[widget_idx],
                    ),
                ),
                (
                    "Classification heatmap",
                    self.init_plot(
                        "heatmap",
                        plot_classification_heatmap,
                        widget_idx=widget_idx,
                        event=None,
                        predictions_loader=self.preds_data[widget_idx],
                        model=self.model_select[widget_idx],
                        accumulate_by=self.accumulate_select[widget_idx],
                        species=self.species_select[widget_idx],
                        threshold=self.clfier_thresh[widget_idx],
                        clfier_type=self.clfier_select[widget_idx],
                        **self.kwargs,
                    ),
                ),
                active=[0, 1, 2],
                sizing_mode="stretch_width",
                # by default create all annotations as one big annotations file
                # # add button to save as raven annotations
            ),
            sizing_mode="stretch_width",
        )
        # Side by side on desktop. On a phone the plots come first and the
        # settings (plus the logo and contact block) sit at the bottom.
        return _mobile_stack_row(
            _mobile_move_last(sidebar),
            main_content,
            sizing_mode="stretch_width",
        )

    def make_sidebar(
        self, widget_idx, model=True, classifier_page=False, all_models=False
    ):
        """
        Build the sidebar widgets for a dashboard page.

        Parameters
        ----------
        widget_idx : int
            index of the widget
        model : bool
            whether to include the model selector
        classifier_page : bool
            whether the sidebar belongs to the classifier page
        all_models : bool
            whether the sidebar belongs to the all-models page

        Returns
        -------
        pn.Column
            column of sidebar widgets
        """
        widgets = [pn.pane.Markdown("## Settings")]

        if model:
            widgets.append(
                self.init_widget(
                    widget_idx, "model", name="Model", options=self.models
                )
            )

        if not classifier_page:
            widgets.extend(
                [
                    self.init_widget(
                        widget_idx,
                        "label",
                        name="Label by",
                        options=self.label_by,
                    ),
                    (
                        pn.widgets.StaticText(
                            name="", value="View only annotated?"
                        )
                        if not self.ground_truth is None
                        else None
                    ),
                    (
                        self.init_widget(
                            widget_idx,
                            "noise",
                            name="remove_noise",
                            options=[True, False],
                            attr="RadioBoxGroup",
                            value=False,
                            inline=True,
                        )
                        if not self.ground_truth is None
                        else None
                    ),
                    (
                        self.init_widget(
                            widget_idx,
                            "class",
                            name="Classification Type",
                            options=["knn", "linear"],
                        )
                        if "probing" in self.evaluation_task
                        else None
                    ),
                ]
            )
        else:
            widgets.extend(
                [
                    self.init_widget(
                        widget_idx,
                        w_type="clfier",
                        name="Integrated or linear classifier",
                        options=["Integrated", "Linear"],
                        attr="RadioBoxGroup",
                        inline=True,
                        value="Integrated",
                    ),
                    self.init_widget(
                        widget_idx,
                        w_type="species",
                        name="Select species",
                        options=self.class_options[widget_idx],
                    ),
                    self.init_widget(
                        widget_idx,
                        w_type="accumulate",
                        name="Select what to aggregate by",
                        options=["day", "week", "month"],
                    ),
                ]
            )

        return pn.Column(*widgets, width=180, margin=(10, 10))

    @staticmethod
    def _build_page_safely(name, builder, *args, **kwargs):
        """Build one dashboard page, falling back to an error panel on failure.

        A single broken model or incomplete evaluation for one dataset must not
        take down the whole dashboard session (previously any exception here
        bubbled up through ``build_layout`` and produced an HTTP 500 — i.e. a
        dead iframe on the website, and a failed session during pre-warming).
        The fallback is a Column with *two* children so the
        ``sidebar, content = page.objects`` unpacking in ``build_layout`` keeps
        working even for pages that failed to build.
        """
        try:
            return builder(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Failed to build dashboard page '{name}': {e}")
            error = pn.Column(
                pn.pane.Markdown(
                    f"## ⚠️ {name} could not be built\n\n"
                    f"**{type(e).__name__}: {e}**\n\n"
                    "This usually means the pipeline results for this dataset "
                    "are incomplete or out of date — rerun the pipeline with "
                    "`overwrite=True`, or check that every configured model "
                    "was actually evaluated. The other dashboard tabs are "
                    "unaffected."
                ),
                sizing_mode="stretch_both",
            )
            return pn.Column(error, error)

    def build_layout(self):
        """
        Builds the layout for the dashboard with two models and a single model page.
        The layout consists of a single model page, a two-models comparison page,
        and a page showing all models. Each page contains sidebars with model-specific
        information and content areas for visualizations.
        """

        # Build both model pages to initialize widgets. Each page is guarded
        # individually so one broken model/evaluation can't kill the session.
        model0_page = self._build_page_safely(
            "Single model", self.model_page, 0, single_model=True
        )
        model1_page = self._build_page_safely(
            "Two models (model 1)", self.model_page, 1
        )
        model2_page = self._build_page_safely(
            "Two models (model 2)", self.model_page, 2
        )
        model_all_page = self._build_page_safely(
            "All models", self.all_models_page, 3
        )
        apply_classifier0_page = self._build_page_safely(
            "Single model predictions", self.apply_clfier_page, 4
        )
        apply_classifier1_page = self._build_page_safely(
            "Two model predictions (model 1)", self.apply_clfier_page, 5
        )
        apply_classifier2_page = self._build_page_safely(
            "Two model predictions (model 2)", self.apply_clfier_page, 6
        )

        # Extract sidebars and content
        sidebar0, content0 = model0_page.objects
        sidebar1, content1 = model1_page.objects
        sidebar2, content2 = model2_page.objects
        sidebar4, content4 = apply_classifier1_page.objects
        sidebar5, content5 = apply_classifier2_page.objects

        # Wrap sidebars with titles
        sidebar0 = pn.Column(
            pn.pane.Markdown("## Model 1"),
            sidebar0,  # , sizing_mode="stretch_height"
        )
        sidebar1 = pn.Column(
            pn.pane.Markdown("## Model 2"),
            sidebar1,  # , sizing_mode="stretch_height"
        )

        self.app = pn.Tabs(
            ("Single model", model0_page),
            (
                "Two models",
                _mobile_stack_row(
                    _mobile_move_last(_mobile_stack_row(sidebar1, sidebar2)),
                    _mobile_stack_row(content1, content2),
                    sizing_mode="stretch_both",
                ),
            ),
            ("All models", model_all_page),
            ("Single Model Predictions", apply_classifier0_page),
            (
                "Two Model Predictions",
                _mobile_stack_row(
                    _mobile_move_last(_mobile_stack_row(sidebar4, sidebar5)),
                    _mobile_stack_row(content4, content5),
                    sizing_mode="stretch_both",
                ),
            ),
            dynamic=True,
        )

        self.add_styling(
            model0_page, model2_page, model_all_page, apply_classifier0_page
        )

        # Last step: make every nested component fluid on narrow screens so the
        # dashboard fits the width of a phone instead of scrolling sideways.
        apply_mobile_styles(self.app)

    def add_styling(self, *pages):
        """
        Add the logo, contact info, and close button to each page sidebar.

        Parameters
        ----------
        *pages
            dashboard pages whose sidebars should be styled
        """

        
        logo = pkg_resources.files("bacpipe") / "imgs" / "bacpipe_unlabelled.png"
            
        logo_path = Path(str(logo))

        for page in pages:
            sidebar = page.objects[0]
            # Add logo to the sidebar
            sidebar.append(pn.pane.PNG(logo_path, sizing_mode="scale_width"))

            # Add a spacer + contact info below the logo
            sidebar.append(pn.Spacer(height=20))
            sidebar.append(pn.pane.Markdown("""
                    **Contact**
                    
                    If you run into problems, please raise issues on github
                    
                    Please collaborate and help make bacpipe as convenient for many as possible
                    
                    🌍 [github](https://github.com/bioacoustic-ai/bacpipe)  
                    
                    To stay updated with new releases, subscribe to the [newsletter](https://buttondown.com/vskode)
                    """))
            # Add close button to the header
            close_button = pn.widgets.Button(name="❌ close dashboard")

            def shutdown_callback(event):
                """
                Shut down the dashboard server.

                Parameters
                ----------
                event : object
                    panel button click event
                """
                logger.info("Shutting down dashboard server...")
                sys.exit(0)

            close_button.on_click(shutdown_callback)

            sidebar.append(close_button)


def visualize_using_dashboard(
    models,
    dashboard_port=5006,
    dashboard_address="localhost",
    dashboard_websocket_origin=False,
    extra_patterns=None,
    **kwargs,
):
    """
    Create and serve the dashboard for visualization. To colorcode embeddings
    by other labels than the default ones, create an annotations file with timestamps.
    An example file can be found in 'bacpipe/tests/test_data/annotations.csv'.
    Multiple dashboards can be opened, the port will simply increment.

    Parameters
    ----------
    models : list
        embedding models
    dashboard_port : int, optional
        port the dashboard is served on, by default 5006
    dashboard_address : str, optional
        address the dashboard is served on, by default "localhost"
    dashboard_websocket_origin : bool, optional
        whether to allow cross origin websocket connections,
        by default False
    extra_patterns : list, optional
        list of (url pattern, tornado RequestHandler) tuples to register on
        the Panel server, e.g. the website's ``/api/datasets`` endpoint that
        feeds the dataset dropdown, by default None
    kwargs : dict
        Dictionary with parameters for dashboard creation
    """
    # Server options that must NOT flow into the per-session dashboard kwargs.
    # Tuned for the website: keep the bokeh session token valid for 24h (the
    # token is re-sent on every websocket reconnect, so it must outlive the
    # 12h session lifetime — a 1h token caused recurring "Token is expired"
    # errors for visitors who kept the dashboard open) and keep sessions alive
    # for 12h so warm sessions stay reusable and a visitor's own session does
    # not disappear mid-browse.
    server_options = {
        "session_token_expiration": kwargs.pop("session_token_expiration", 24 * 60 * 60),
        "unused_session_lifetime_milliseconds": kwargs.pop(
            "unused_session_lifetime_milliseconds", 12 * 60 * 60 * 1000
        ),
        "check_unused_sessions_milliseconds": kwargs.pop(
            "check_unused_sessions_milliseconds", 5 * 60 * 1000
        ),
    }

    models = [bacpipe.confirm_model_name(model, **kwargs) for model in models]
    import panel as pn

    favicon_logo = pkg_resources.files("bacpipe") / "imgs" / "bacpipe_favicon_white.png"

    favicon_path = Path(str(favicon_logo))

    def create_dashboard():
        # Build a fresh dashboard per session. The per-user ``audio_dir`` is
        # read from the ``?audio_dir=`` query parameter (see
        # ``DashBoard.get_audio_dir``), which is how the website lets each
        # visitor pick a dataset without sharing state.
        # Must run before any component is created, so late/dynamically built
        # components are mobile friendly too.
        enable_mobile_defaults()
        audio_dir = DashBoard.get_audio_dir()

        session_kwargs = {**kwargs, "audio_dir": audio_dir}
        dashboard = DashBoard(models, **session_kwargs)

        # Build the dashboard layout
        try:
            dashboard.build_layout()
        except Exception as e:
            logger.exception(
                f"\nError building dashboard layout: {str(e)}\n \n "
                "Are you sure all the evaluations have been performed? "
                "If not, rerun the pipeline with `overwrite=True`.\n \n "
            )
            raise e

        template = pn.template.BootstrapTemplate(
            site="bacpipe dashboard",
            title="Explore embeddings of audio data",
            favicon=str(favicon_path),  # must be a path ending in .ico, .png, etc.
            # Without this, mobile browsers lay the page out in a ~980px wide
            # virtual viewport and let the visitor pan sideways. With it the
            # page is exactly as wide as the screen and the dashboard's mobile
            # media queries (see MOBILE_BREAKPOINT) actually match.
            meta_viewport="width=device-width, initial-scale=1, viewport-fit=cover",
            main=[dashboard.app],
        )
        template.config.raw_css = [_TEMPLATE_MOBILE_CSS]
        return template

    websocket_origin = dashboard_websocket_origin if dashboard_websocket_origin else None

    port_not_available = True
    while port_not_available:
        try:
            pn.serve(
                create_dashboard,  # callable — Panel invokes it per session
                port=dashboard_port,
                address=dashboard_address,
                websocket_origin=websocket_origin,
                show=False,
                extra_patterns=extra_patterns or [],
                **server_options,
            )
            port_not_available = False
        except OSError:
            logger.warning(
                f"The port {dashboard_port} is already in use. This "
                "is most likely the case because you already have a "
                "dashboard open. There is a exit button in the bottom "
                "left of the dashboard. If this was intentional and you "
                "want to open multiple dashboards at once, ignore this message."
            )
            dashboard_port += 1
