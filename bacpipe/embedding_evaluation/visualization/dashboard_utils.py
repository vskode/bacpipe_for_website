import io
import json
import uuid
from pathlib import Path

import panel as pn
import param
import matplotlib
import seaborn as sns
import pandas as pd
import datetime
import logging
from bacpipe import settings
from .tour import mark_target

logger = logging.getLogger("bacpipe")

sns.set_theme(style="whitegrid")

matplotlib.use("agg")

_SAVE_FIGURE_CHROME_HINT = (
    "Saving figures requires Google Chrome or Microsoft Edge (kaleido uses it for "
    "static image export) but none could be found on this system. Install Chrome or "
    "Edge from your browser's official website, or run `kaleido_get_chrome` in the "
    "terminal to install a compatible Chrome automatically."
)

_SAVE_FIGURE_KALEIDO_HINT = (
    "kaleido is required to save figures but is not installed. Install it with "
    "`pip install 'kaleido>=1.0.0'` and try again."
)


def _friendly_export_error(exc):
    """
    Map kaleido static-export failures to a friendly, actionable message.

    Runs when the "Save Figure" button is clicked: kaleido only needs a browser
    at export time, so on machines without Chrome/Edge (or without kaleido) the
    raw library exception would otherwise leak into the dashboard notification.

    Parameters
    ----------
    exc : Exception
        the exception raised while exporting the figure

    Returns
    -------
    str or None
        a friendly hint when ``exc`` is a known kaleido export failure, else
        ``None`` so callers can fall back to the raw error text
    """
    try:
        import kaleido.errors as kaleido_errors
    except Exception:  # pragma: no cover - kaleido is a hard dependency
        kaleido_errors = None

    # kaleido/choreographer cannot find or launch a browser
    if kaleido_errors is not None:
        browser_errors = tuple(
            err
            for err in (
                getattr(kaleido_errors, name, None)
                for name in (
                    "ChromeNotFoundError",
                    "BrowserFailedError",
                    "BrowserDepsError",
                )
            )
            if err is not None
        )
        if browser_errors and isinstance(exc, browser_errors):
            return _SAVE_FIGURE_CHROME_HINT

    # plotly wraps ChromeNotFoundError into a RuntimeError mentioning Chrome,
    # and choreographer's BrowserDepsError also mentions Chrome in its message
    if "chrome" in str(exc).lower():
        return _SAVE_FIGURE_CHROME_HINT

    # plotly raises ValueError when the kaleido package itself is missing
    if isinstance(exc, ValueError) and "kaleido package" in str(exc).lower():
        return _SAVE_FIGURE_KALEIDO_HINT

    # defensive: raw ModuleNotFoundError for the kaleido package
    if isinstance(exc, ModuleNotFoundError) and "kaleido" in str(exc).lower():
        return _SAVE_FIGURE_KALEIDO_HINT

    return None


def _capture_view_ranges(relayout_data):
    """
    Extract 2D axis ranges from a plotly ``relayout_data`` event.

    Plotly reports zoom/pan through ``relayout_data`` using either the list
    form ``{"xaxis.range": [lo, hi], ...}`` or the indexed form
    ``{"xaxis.range[0]": lo, "xaxis.range[1]": hi, ...}``. Both are handled
    here so the current view can be re-applied before a static export.

    Parameters
    ----------
    relayout_data : dict or None
        the ``relayout_data`` payload emitted by the Plotly pane

    Returns
    -------
    dict
        maps ``"xaxis"`` and/or ``"yaxis"`` to ``(lo, hi)`` tuples, or is
        empty when no ranges were reported (e.g. a double-click reset)
    """
    if not relayout_data:
        return {}
    ranges = {}
    for axis in ("xaxis", "yaxis"):
        value = relayout_data.get(f"{axis}.range")
        if isinstance(value, (list, tuple)) and len(value) == 2:
            ranges[axis] = (value[0], value[1])
            continue
        lo = relayout_data.get(f"{axis}.range[0]")
        hi = relayout_data.get(f"{axis}.range[1]")
        if lo is not None and hi is not None:
            ranges[axis] = (lo, hi)
    return ranges


def _apply_view_ranges(fig, ranges):
    """
    Apply previously captured axis ranges to a figure so a static export
    reflects the zoomed-in view the user is currently looking at.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        figure to update
    ranges : dict
        output of :func:`_capture_view_ranges`

    Returns
    -------
    plotly.graph_objects.Figure
        the same figure with the ranges applied
    """
    if not ranges:
        return fig
    if "xaxis" in ranges:
        fig.update_xaxes(range=list(ranges["xaxis"]), autorange=False)
    if "yaxis" in ranges:
        fig.update_yaxes(range=list(ranges["yaxis"]), autorange=False)
    return fig


def _static_export_figure(fig):
    """
    Return a copy of ``fig`` with webgl traces converted to regular SVG
    scatter traces.

    ``plot_embeddings_px`` renders 2D embeddings with ``render_mode="webgl"``.
    Kaleido's headless-Chrome export is unreliable for webgl traces (markers
    and especially the continuous colorbar can be missing), so export a
    plain-SVG copy instead.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        figure to prepare for export

    Returns
    -------
    plotly.graph_objects.Figure
        copy of ``fig`` without webgl traces
    """
    import json

    import plotly.graph_objects as go

    fig_dict = json.loads(fig.to_json())
    for trace in fig_dict.get("data", []):
        if trace.get("type") == "scattergl":
            trace["type"] = "scatter"
    return go.Figure(fig_dict)


def downloads_only():
    """
    Whether saved files go to the visitor's browser instead of the disk.

    ``dashboard_downloads`` in ``settings.yaml``; the website turns it on in
    ``run_pipeline.py`` so visitors cannot fill up the server's disk.
    """
    return bool(getattr(settings, "dashboard_downloads", False))


def display_path(path):
    """
    A results path as shown to website visitors.

    Starts at the dataset's results folder (e.g.
    ``test_data/evaluations/birdnet/plots``) instead of the server's own
    directory layout.
    """
    parts = Path(path).parts
    if "evaluations" in parts:
        start = max(parts.index("evaluations") - 1, 0)
        return Path(*parts[start:]).as_posix()
    return Path(path).as_posix()


class OptionalDownload(pn.widgets.FileDownload):
    """
    A download button whose callback may decline to send a file.

    The callback returns a file-like object to download, or ``None`` when
    there is nothing to send (e.g. no points selected); the plain
    ``FileDownload`` raises an error for ``None``. The callback may also set
    ``filename`` before returning.
    """

    # Panel's FileDownload never sends its colour to the browser (``color``
    # is mapped to nothing and ``button_type`` is folded into ``color``), so
    # every download button came out grey. Map it like ``Button`` does.
    _rename = {**pn.widgets.FileDownload._rename, "color": "button_type"}

    @param.depends("_clicks", watch=True)
    def _transfer(self):
        fileobj = self.callback() if self.callback is not None else None
        if fileobj is not None:
            self._sync_data(fileobj)


# Browser-side "Save Figure" for the website: downloads the figure exactly as
# displayed, so the server never renders or stores anything. Plotly figures
# use plotly's own PNG export; matplotlib figures are already PNG images in
# the page. ``container`` is a CSS class on the Column holding the figure.
_DOWNLOAD_FIGURE_JS = r"""
function dq(sel, root, out) {
  if (root.shadowRoot) dq(sel, root.shadowRoot, out);
  root.querySelectorAll(sel).forEach((e) => out.push(e));
  root.querySelectorAll("*").forEach((e) => { if (e.shadowRoot) dq(sel, e.shadowRoot, out); });
  return out;
}
const shown = (e) => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
const box = dq("." + container, document, []).find(shown);
if (box) {
  const parts = [base];
  Object.values(widgets).forEach((w) => {
    if (w && w.value !== undefined && w.value !== null && w.value !== "") parts.push(String(w.value));
  });
  const name = parts.join("_").replace(/[^A-Za-z0-9._-]+/g, "-");
  const plot = dq(".js-plotly-plot", box, []).find(shown);
  if (plot && window.Plotly) {
    window.Plotly.downloadImage(plot, { format: "png", width: 1200, height: 800, filename: name });
  } else {
    const img = dq("img", box, []).find((i) => shown(i) && /^data:image|^blob:/.test(i.src));
    if (img) {
      const a = document.createElement("a");
      a.href = img.src;
      a.download = name + ".png";
      document.body.appendChild(a);
      a.click();
      a.remove();
    }
  }
}
"""


def attach_figure_download(button, container, base, widgets=None):
    """
    Make ``button`` download the figure shown inside ``container``.

    Parameters
    ----------
    button : pn.widgets.Button
        the "Save Figure" button
    container : panel layout
        layout holding the figure; gets a unique CSS class to be found by
    base : str
        start of the file name; the current ``widgets`` values are appended
    widgets : dict or None
        name -> widget whose value goes into the file name
    """
    css_class = "bp-download-" + uuid.uuid4().hex[:12]
    container.css_classes = [*container.css_classes, css_class]
    # Panel only turns widgets into browser models when they are top-level
    # ``args``, so they are passed one by one and collected in the script.
    args = {
        f"w{i}": w
        for i, w in enumerate((widgets or {}).values())
        if isinstance(w, pn.widgets.Widget)
    }
    code = (
        f"const container = {json.dumps(css_class)};\n"
        f"const base = {json.dumps(base)};\n"
        f"const widgets = [{', '.join(args)}];\n" + _DOWNLOAD_FIGURE_JS
    )
    button.js_on_click(args=args, code=code)


class DashBoardHelper:
    """
    Helper class providing shared widget event handlers and figure update
    logic used by the dashboard pages.
    """

    def handle_selection(self, event, widget_idx=None):
        """
        Triggered when the user uses the Lasso or Box select tool.

        Parameters
        ----------
        event : panel param event
            the selection event triggered by the user
        widget_idx : int, optional
            index of the widget the selection belongs to, by default None
        """
        if not event.new:
            return

        try:
            selected_points = event.new.get("points", [])

            if not selected_points:
                logger.info("Selection cleared")
                return

            logger.info(f"Selected {len(selected_points)} points")

            # Extract data from the selected points
            points = {}
            for idx, keys in enumerate(
                ["audiofilename", "start", "end", "index", "label"]
            ):
                points[keys] = [p["customdata"][idx] for p in selected_points]

            self.spec_plot_obj[widget_idx]._cache_selected_points(points)
            logger.info(f"First 5 files: {points['audiofilename'][:5]}")

        except Exception as e:
            logger.info(f"Error handling selection: {str(e)}")

    def _selected_points_table(self, dialogue_panel, widget_idx):
        """
        The selected points as a table, where they would be saved, and the
        file name; ``None`` (with a message) if nothing is selected.
        """
        if not hasattr(self.spec_plot_obj[widget_idx], "selected_points"):
            dialogue_panel.visible = True
            dialogue_panel.value = (
                "No points have been selected. Use the Box Select or Lasso "
                "Select tool of the embedding plot first."
            )
            return None
        df = pd.DataFrame(self.spec_plot_obj[widget_idx].selected_points)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = timestamp + "_selected_points.csv"
        self._trigger_spec_obj_update[widget_idx]()
        model_name = self.spec_plot_obj[widget_idx].model_name
        save_path = self.path_func(model_name).plot_path
        return df, save_path, file_name

    def download_selected_points(self, dialogue_panel, download, widget_idx):
        """
        Website version of "Save selection to file": send the CSV to the
        visitor's browser instead of writing it on the server.

        Parameters
        ----------
        dialogue_panel : panel widget
            panel used to show the confirmation message
        download : OptionalDownload
            the button, whose ``filename`` is set to the CSV's name
        widget_idx : int
            index of the widget the selected points belong to

        Returns
        -------
        io.StringIO or None
            the CSV, or ``None`` if no points are selected
        """
        table = self._selected_points_table(dialogue_panel, widget_idx)
        if table is None:
            return None
        df, save_path, file_name = table
        download.filename = file_name
        dialogue_panel.visible = True
        dialogue_panel.value = (
            f"{len(df)} selected points were downloaded as {file_name}. "
            "When you run bacpipe yourself, the file is saved to "
            f"{display_path(save_path)} instead."
        )
        return io.StringIO(df.to_csv())

    def save_selected_points(self, event, dialogue_panel, widget_idx):
        """
        Save the currently selected points to a csv file in the plot path.

        Parameters
        ----------
        event : panel event
            the event triggering the save
        dialogue_panel : panel widget
            panel used to show the save confirmation message
        widget_idx : int
            index of the widget the selected points belong to
        """
        table = self._selected_points_table(dialogue_panel, widget_idx)
        if table is None:
            return
        df, save_path, file_name = table

        df.to_csv(save_path / file_name)

        dialogue_panel.visible = True
        dialogue_panel.value = (
            f"{len(df)} selected points were save to "
            + str(save_path / file_name)
        )

    def handle_click(self, event, widget_idx=0):
        """
        Triggered when the user clicks on a point in the embedding plot.

        Parameters
        ----------
        event : panel param event
            the click event triggered by the user
        widget_idx : int, optional
            index of the widget the click belongs to, by default 0
        """
        if not event.new:
            return
        try:
            point_data = event.new["points"][0]
            logger.info(f"DEBUG CLICK: {point_data}")
            self.show_point(point_data, widget_idx)
        except Exception as e:
            logger.info(f"Error handling click: {str(e)}")
            # Without this the spectrogram just stays blank, which looks like
            # the click was ignored (e.g. the dataset's audio is missing).
            spec_plot = self.spec_plot_obj.get(widget_idx)
            if spec_plot is not None:
                spec_plot.panel_static_text.visible = True
                spec_plot.panel_static_text.value = (
                    "Could not load the audio of this point: " + str(e)
                )

    def show_point(self, point_data, widget_idx=0):
        """
        Show an embedding point in the spectrogram panel, as a click does.

        Also used by the guided tour (``tour.py``) to show an example point
        when the visitor has not clicked one yet.

        Parameters
        ----------
        point_data : dict
            one point of a plotly click event; only ``"customdata"`` is used
        widget_idx : int, optional
            index of the widget the point belongs to, by default 0
        """
        # this ensures that the sample rate and
        # input segment length are set specific to the
        # currently used model
        self._trigger_spec_obj_update[widget_idx]()

        # Generate the new figure
        new_fig = self.spec_plot_obj[widget_idx].update_spectrogram(
            clickData=point_data
        )

        self.spectrogram_plot_panel[widget_idx].object = new_fig

        # Clicking a point loads (caches) the audio segment. Unless the
        # visitor switched the "Audio on click" radio button to "stay
        # silent", the segment is also pushed straight into the client-side
        # player so it is ready to play on that device.
        if self.autoplay_on_click(widget_idx):
            self.play_clicked_audio(widget_idx)

    def autoplay_on_click(self, widget_idx=0):
        """
        Whether clicking a point should start playback right away.

        Parameters
        ----------
        widget_idx : int, optional
            index of the widget the click belongs to, by default 0

        Returns
        -------
        bool
            True when the "Audio on click" radio button is set to play (the
            default, also used when no such widget exists).
        """
        widget = getattr(self, "autoplay_select", {}).get(widget_idx)
        if widget is None:
            return True
        return bool(widget.value)

    def play_clicked_audio(self, widget_idx=0):
        """
        Push the clicked segment into the browser player (paused, ready to play).

        The segment is loaded and the player becomes visible, but playback is
        *not* started from the server: the dashboard shares one bokeh session
        between every visitor, so unpausing the shared player would start sound
        on every connected device. The visitor starts the sound with the
        "Play audio" button or the native <audio> controls on their own device.

        Parameters
        ----------
        widget_idx : int, optional
            index of the widget the click belongs to, by default 0
        """
        spec_plot = self.spec_plot_obj[widget_idx]
        # Load the clicked segment into the (now visible) client-side player so
        # the visitor can start it with the "Play audio" button or the native
        # <audio> controls. We intentionally do NOT unpause the player here:
        # the website shares a single bokeh session between all visitors, so a
        # server-side ``paused=False`` would start playback in every connected
        # browser instead of only on the device that clicked the point.
        spec_plot.update_audio_player()

    def init_interactive_embed_plot(self, widget_idx):
        """
        Initialize interactive embedding plot with dummy figure.

        Parameters
        ----------
        widget_idx : int
            index of the widget to initialize
        """
        from .visualize_spectrograms import SpectrogramPlot

        # Create Plotly pane with dummy figure and reserved height to prevent accordion collapse
        self.interactive_embed_plot[widget_idx] = pn.pane.Plotly(
            SpectrogramPlot.dummy_image(
                title="Loading...",
                height=self.kwargs.get(
                    "embed_fig_height", settings.embed_fig_height
                ),
            ),
            height=self.kwargs.get(
                "embed_fig_height", settings.embed_fig_height
            ),
            # Responsive width + fixed height, with the figure on
            # ``autosize=True`` (see ``dashboard.spectrogram_panel``).
            # ``min_width=0`` lets the pane shrink below its content width; the
            # desktop flex rows are pinned by ``dashboard._DESKTOP_ITEM_CSS``
            # (applied in ``dashboard._mobile_stack_row``) so the plot stops
            # reporting its content width back into the layout.
            config={"responsive": True},
            sizing_mode="stretch_width",
            # See ``dashboard.model_page``: ``min_width=0`` stops the plot's
            # intrinsic width from blowing up the desktop flex row.
            min_width=0,
        )

        # Add event handlers
        self.interactive_embed_plot[widget_idx].param.watch(
            lambda x: self.handle_click(x, widget_idx), "click_data"
        )
        self.interactive_embed_plot[widget_idx].param.watch(
            lambda x: self.handle_selection(x, widget_idx), "selected_data"
        )
        # Track zoom/pan so the "Save Figure" button can export the currently
        # displayed view rather than the full (unzoomed) figure.
        self.interactive_embed_plot[widget_idx].param.watch(
            lambda x: self._store_view_ranges(x, widget_idx), "relayout_data"
        )
        button = pn.widgets.Button(name="Save Figure", button_type="primary")
        notification = pn.pane.Markdown("")

        # Attach save button handler that gets current values from widgets at click time
        button.on_click(lambda e: self._on_save_button_click(e, widget_idx))
        if downloads_only():
            # The browser saves the plot as displayed (zoom and pan included).
            attach_figure_download(
                button,
                self.interactive_embed_plot[widget_idx],
                "embedding",
                {
                    "model": self.model_select.get(widget_idx),
                    "label": self.label_select.get(widget_idx),
                },
            )

        self.embed_save_button[widget_idx] = button
        self.embed_notification[widget_idx] = notification

    def _store_view_ranges(self, event, widget_idx):
        """
        Store the current zoom/pan ranges of the embedding plot.

        Parameters
        ----------
        event : panel event
            the ``relayout_data`` event triggered by the user zooming/panning
        widget_idx : int
            index of the widget whose view changed
        """
        ranges = _capture_view_ranges(event.new if event else None)
        if ranges:
            self._embed_view_ranges[widget_idx] = ranges
        else:
            # e.g. a double-click reset returned to the full view
            self._embed_view_ranges.pop(widget_idx, None)

    def _on_save_button_click(self, event, widget_idx):
        """
        Button click handler that saves the current embedding plot with
        preserved zoom/pan.

        Parameters
        ----------
        event : panel event
            the event triggering the save
        widget_idx : int
            index of the widget belonging to the save button
        """
        model_name = self.model_select[widget_idx].value
        label_by = self.label_select[widget_idx].value
        displayed_fig = self.interactive_embed_plot[widget_idx].object

        filename = f"{model_name}_embedding_{label_by}.png"
        save_path = self.path_func(model_name).plot_path / filename
        if downloads_only():
            # The download itself happens in the browser (see
            # ``init_interactive_embed_plot``); nothing is written here.
            self.embed_notification[widget_idx].object = (
                "✓ Downloaded. When you run bacpipe yourself, the figure is "
                f"saved to {display_path(save_path)} instead."
            )
            return

        try:
            # Rebuild a non-webgl copy of the figure (so kaleido renders the
            # continuous colorbar reliably) and re-apply the current zoom/pan
            # so the exported PNG matches what the user is looking at. The
            # canvas is fixed so the plot dimensions never depend on the
            # width of the legend/colorbar.
            export_fig = _static_export_figure(displayed_fig)
            export_fig = _apply_view_ranges(
                export_fig, self._embed_view_ranges.get(widget_idx, {})
            )
            export_fig.write_image(save_path, width=1200, height=800)
            self.embed_notification[widget_idx].object = (
                f"✓ Saved to: {save_path}"
            )
        except Exception as e:
            message = _friendly_export_error(e) or f"Error: {str(e)}"
            self.embed_notification[widget_idx].object = f"✗ {message}"

    def update_main_plot(self, p_type, plot_func, widget_idx, **kwargs):
        """
        Update existing plot by just updating the .object

        Parameters
        ----------
        p_type : str
            type of the plot to update
        plot_func : callable
            function that creates the plot
        widget_idx : int
            index of the widget to update

        Returns
        -------
        panel widget
            the updated plot panel
        """
        plots_dict = getattr(self, f"{p_type}_plot")

        # Just update the figure object (no recreation!)
        if p_type == "interactive_embed":
            self.interactive_embed_plot[widget_idx].object = plot_func(
                widget_idx=widget_idx, **kwargs
            )

        else:
            # Other plot types
            new_panel = self.add_save_button(plot_func, **kwargs)
            plots_dict[widget_idx] = new_panel

            if isinstance(new_panel[0], pn.pane.Plotly):
                new_panel[0].object = plot_func(**kwargs)

        return plots_dict[widget_idx]

    def init_plot(self, p_type, plot_func, widget_idx, **kwargs):
        """
        Initialize a plot panel and store it in the corresponding plot dict.

        Parameters
        ----------
        p_type : str
            type of the plot to initialize
        plot_func : callable
            function that creates the plot
        widget_idx : int
            index of the widget to initialize

        Returns
        -------
        panel widget
            the initialized plot panel
        """
        getattr(self, f"{p_type}_plot")[widget_idx] = pn.panel(
            self.plot_widget(plot_func, widget_idx=widget_idx, **kwargs),
            tight=False,
        )
        return getattr(self, f"{p_type}_plot")[widget_idx]

    def plot_widget(self, plot_func, **kwargs):
        """
        Wrap the plot function in a panel widget, either bound to the
        kwargs or with an added save button.

        Parameters
        ----------
        plot_func : callable
            function that creates the plot

        Returns
        -------
        panel widget
            panel object containing the plot
        """
        if kwargs.get("return_fig", False):
            fig_panel = pn.panel(pn.bind(plot_func, **kwargs))
            # Make the plot fill the available accordion width
            fig_panel.sizing_mode = "stretch_width"
            return fig_panel
        else:
            return self.add_save_button(plot_func, **kwargs)

    def widget(self, name, options, attr="Select", width=120, **kwargs):
        """
        Create a panel widget of the requested type.

        Parameters
        ----------
        name : str
            label of the widget
        options : list
            options for the widget
        attr : str, optional
            name of the panel widget class to use, by default "Select"
        width : int, optional
            width of the widget, by default 120

        Returns
        -------
        panel widget
            the created widget
        """
        return getattr(pn.widgets, attr)(
            name=name, options=options, width=self.widget_width, **kwargs
        )

    def init_widget(self, idx, w_type, **kwargs):
        """
        Initialize a widget and store it in the corresponding select dict.

        Parameters
        ----------
        idx : int
            index of the widget to initialize
        w_type : str
            type of the widget to initialize

        Returns
        -------
        panel widget
            the initialized widget
        """
        getattr(self, f"{w_type}_select")[idx] = self.widget(**kwargs)
        return getattr(self, f"{w_type}_select")[idx]

    def change_input_options(self, clfier_selection, widget_idx):
        """
        Update the classifier widget labels based on the selected
        classifier type.

        Parameters
        ----------
        clfier_selection : panel event
            event containing the newly selected classifier type
        widget_idx : int
            index of the widget to update
        """
        if clfier_selection.new == "Linear":
            self.btn_run_clfier[widget_idx].name = "Apply linear classifier"
            # Never offered on the website (see ``apply_clfier_page``).
            self.clfier_path[widget_idx].visible = not downloads_only()
        else:
            self.btn_run_clfier[widget_idx].name = (
                "Load predictions from integrated classifier"
            )
            self.clfier_path[widget_idx].visible = False

    def add_save_button(self, plot_func, **kwargs):
        """
        Adds a save button to the plot panel.

        Parameters
        ----------
        plot_func : callable
            function that creates the plot

        Returns
        -------
        panel column
            panel containing the plot and the save button
        """

        # Check if this is for a Plotly plot by checking if any widgets are passed
        has_widgets = any(hasattr(v, "value") for v in kwargs.values())

        if has_widgets:
            # Create bound figure panel (will auto-update)
            fig_panel = pn.panel(pn.bind(plot_func, **kwargs))
        else:
            # No widgets, just call the function once
            fig_panel = pn.panel(plot_func(**kwargs))

        # Make the plot fill the available container width so the dashboard
        # stays responsive when the browser window is resized. Plotly panes are
        # deliberately skipped: with ``sizing_mode="stretch_width"`` Panel's
        # Plotly view relayouts the figure on every layout pass, which feeds
        # back into Bokeh's layout and makes the plot oscillate in width (the
        # "shivering" dashboard). Plotly panes stay on the default sizing mode
        # so Plotly sizes itself (see ``dashboard.spectrogram_panel``).
        if not isinstance(fig_panel, pn.pane.Plotly):
            fig_panel.sizing_mode = "stretch_width"

        def _save_dir(plot_kwargs):
            """Folder the figure is saved to (when saving to disk)."""
            model_name = plot_kwargs.get(
                "model_name", plot_kwargs.get("model", "all_models")
            )
            if model_name == "all_models":
                # Any model's results folder leads to the shared overview.
                return (
                    self.path_func(self.models[0]).plot_path.parent.parent
                    / "overview"
                )
            return self.path_func(model_name).plot_path

        def save_figure(event):
            """
            Save the displayed figure to the plot path.

            Parameters
            ----------
            event : panel event
                the event triggering the save
            """
            # Extract values from widgets
            plot_kwargs = {}
            for key, value in kwargs.items():
                if hasattr(value, "value"):
                    plot_kwargs[key] = value.value
                else:
                    plot_kwargs[key] = value

            if downloads_only():
                # The browser downloads the displayed figure (see
                # ``attach_figure_download`` below); nothing is rendered or
                # written on the server.
                notification.object = (
                    "✓ Downloaded. When you run bacpipe yourself, the figure "
                    f"is saved to {display_path(_save_dir(plot_kwargs))} "
                    "instead."
                )
                return

            # Generate the figure
            fig = plot_func(**plot_kwargs)

            # Generate filename
            if "model_name" in plot_kwargs:
                model_name = plot_kwargs["model_name"]
            elif "model" in plot_kwargs:
                model_name = plot_kwargs["model"]
            else:
                model_name = "all_models"

            plot_type = plot_func.__name__.replace("plot_", "")

            if "predictions_loader" in plot_kwargs:
                label_part = f"{plot_kwargs.get('species', 'unknown')}_{plot_kwargs.get('accumulate_by', 'unknown')}"
            elif "label_by" in plot_kwargs:
                label_part = plot_kwargs["label_by"]
            else:
                label_part = "plot"

            default_filename = f"{model_name}_{plot_type}_{label_part}.png"

            # Determine save path
            save_dir = _save_dir(plot_kwargs)
            save_dir.mkdir(exist_ok=True, parents=True)
            save_path = save_dir / default_filename

            # Save the figure (handle both Plotly and matplotlib)
            try:
                import plotly.graph_objs as go

                if isinstance(fig, go.Figure):
                    fig.write_image(save_path, width=1200, height=800)
                else:
                    fig.savefig(save_path, dpi=300, bbox_inches="tight")
            except Exception as e:
                logger.error(f"Error saving figure: {str(e)}")
                message = _friendly_export_error(e) or f"Error: {str(e)}"
                notification.object = f"✗ {message}"
                return

            notification.object = f"✓ Figure saved to: {save_path}"

        # Create button and notification
        button = pn.widgets.Button(name="Save Figure", button_type="primary")
        # The tour points at the heatmap's one via
        # "predictions-heatmap >> .bp-tour-save-figure" (tour.py).
        mark_target(button, "save-figure")
        button.on_click(save_figure)
        notification = pn.pane.Markdown("")

        layout = pn.Column(
            fig_panel,
            pn.Row(button),
            notification,
            sizing_mode="stretch_width",
        )
        if downloads_only():
            attach_figure_download(
                button,
                layout,
                plot_func.__name__.replace("plot_", ""),
                {
                    key: kwargs.get(key)
                    for key in (
                        "model_name", "model", "label_by", "species",
                        "accumulate_by", "task_name",
                    )
                },
            )
        return layout
