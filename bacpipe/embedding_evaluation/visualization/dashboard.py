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
# Bokeh 3.x renders each Panel layout as a custom element whose flex styles
# (``flex-direction``) live in the element's *own* shadow-DOM stylesheet, so
# page-level CSS can not reach them. Instead the media queries below are
# pushed onto the layout models themselves via ``stylesheets=``: they are
# injected into the same shadow root as the default rules and win on narrow
# screens thanks to ``!important``.
MOBILE_BREAKPOINT = 900

_MOBILE_DIRECTION_CSS = f"""
@media (max-width: {MOBILE_BREAKPOINT}px) {{
  :host {{
    flex-direction: column !important;
  }}
}}
"""

_MOBILE_ITEM_CSS = f"""
@media (max-width: {MOBILE_BREAKPOINT}px) {{
  :host {{
    flex: 0 0 auto !important;
    align-self: stretch !important;
    width: 100% !important;
  }}
}}
"""


def _mobile_stack_row(*items, **kwargs):
    """Return a ``pn.Row`` that stacks its children vertically on phones.

    Children keep their natural order when stacked, so pass the panels in the
    order you want top-to-bottom on mobile (e.g. the embedding plot before the
    spectrogram). Extra keyword arguments (``sizing_mode``, ...) are forwarded
    to the ``Row``.
    """
    row = pn.Row(*items, stylesheets=[_MOBILE_DIRECTION_CSS], **kwargs)
    for item in items:
        if getattr(item, "stylesheets", None) is not None:
            item.stylesheets = [*item.stylesheets, _MOBILE_ITEM_CSS]
    return row


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
            sizing_mode="stretch_width",
            height=self.kwargs.get("spectrogram_plot_height"),
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
        # as WAV bytes instead. Each session gets its own dashboard instance,
        # hence its own player, so playback never leaks between visitors.
        audio_player = pn.pane.Audio(
            np.zeros(8000, dtype=np.float32),
            name="Audio playback",
            sample_rate=8000,
            visible=False,
        )
        self.spec_plot_obj[widget_idx].audio_player = audio_player

        def play_current_audio(event):
            self.spec_plot_obj[widget_idx].update_audio_player()

        play_audio_button = pn.widgets.Button(
            name="Play audio", button_type="primary"
        )
        play_audio_button.on_click(play_current_audio)
        save_selection_dialogue = pn.widgets.StaticText(value="", width=400)

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
                pn.Row(play_audio_button, save_selection_button),
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

        # Sidebar above content on phones, side by side on desktop.
        return _mobile_stack_row(sidebar, main_content, sizing_mode="stretch_width")

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

        # Sidebar above content on phones, side by side on desktop.
        return _mobile_stack_row(sidebar, main_content, sizing_mode="stretch_width")

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
            width=600,
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
            width=100,
            height=30,
        )

        self.progress_bar[widget_idx] = pn.indicators.Progress(
            value=0, max=100, bar_color="primary", width=500
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
        # Sidebar above content on phones, side by side on desktop.
        return _mobile_stack_row(sidebar, main_content, sizing_mode="stretch_width")

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
                    _mobile_stack_row(sidebar1, sidebar2),
                    _mobile_stack_row(content1, content2),
                    sizing_mode="stretch_both",
                ),
            ),
            ("All models", model_all_page),
            ("Single Model Predictions", apply_classifier0_page),
            (
                "Two Model Predictions",
                _mobile_stack_row(
                    _mobile_stack_row(sidebar4, sidebar5),
                    _mobile_stack_row(content4, content5),
                    sizing_mode="stretch_both",
                ),
            ),
            dynamic=True,
        )

        self.add_styling(
            model0_page, model2_page, model_all_page, apply_classifier0_page
        )

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
    # Tuned for the website: the document-build token stays valid for an hour
    # (slow builds on big datasets no longer die with "Token is expired") and
    # sessions are kept alive for 12h so pre-warmed sessions stay reusable.
    server_options = {
        "session_token_expiration": kwargs.pop("session_token_expiration", 3600),
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
            main=[dashboard.app],
        )
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
