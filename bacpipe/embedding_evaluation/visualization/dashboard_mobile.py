"""Mobile-only bacpipe dashboard layout.

The desktop dashboard (``dashboard.DashBoard``) carries a tab bar and desktop
plot sizes. On a phone that tab bar wastes the top of the screen and the
desktop plot heights (700px embedding) are far too tall. This module reuses all
of ``DashBoard``'s data loading and widget logic but builds a dedicated phone
layout:

* only the **single-model** page — no tab bar, so the top of the screen goes
  straight to the embedding plot;
* plot heights chosen for a phone screen and set **server-side** (the figures
  are ``autosize=True``, so they follow the pane height). There is no
  client-side resize loop here: that loop was the source of both the desktop
  "shivering" and the mobile "plot vanishes while scrolling" bug.

Keeping this in its own module means mobile tweaks can never destabilise the
desktop dashboard.
"""

import functools

from .dashboard import DashBoard, apply_mobile_styles

# Phone-friendly heights for the two Plotly panes that dominate the single-model
# page. Kept tall enough to stay readable but short enough to leave room for the
# spectrogram and the rest of the sidebar below on a phone screen.
_EMBED_HEIGHT = 430
_SPEC_HEIGHT = 400


def _without_drag(fig):
    """
    Let swipes over a plot scroll the page instead of zooming the plot.

    With plotly's default ``dragmode`` ("zoom") every swipe that starts on a
    plot draws a zoom box, so a visitor scrolling down to the spectrogram
    zooms into the embedding plot instead. ``dragmode=False`` hands swipes
    back to the browser; taps still click dots, and zooming is still one tap
    away in the plot's toolbar.
    """
    if hasattr(fig, "update_layout"):
        fig.update_layout(dragmode=False)
    elif isinstance(fig, dict):
        fig.setdefault("layout", {})["dragmode"] = False
    return fig


def _no_drag(make_figure):
    """Wrap a figure-making function so its figures do not drag (see above)."""

    @functools.wraps(make_figure)
    def wrapper(*args, **kwargs):
        return _without_drag(make_figure(*args, **kwargs))

    return wrapper


class DashBoardMobile(DashBoard):
    """Single-model, mobile-first bacpipe dashboard."""

    def update_main_plot(self, p_type, plot_func, widget_idx, **kwargs):
        # Every embedding figure comes through here (first draw included).
        if p_type == "interactive_embed":
            plot_func = _no_drag(plot_func)
        return super().update_main_plot(p_type, plot_func, widget_idx, **kwargs)

    def build_layout(self):
        # Only the single-model page: no tab bar. ``single_model=True`` also
        # applies the mobile reordering (Model + Label-by selectors above the
        # embedding plot, the rest of the sidebar below).
        page = self._build_page_safely(
            "Single model", self.model_page, 0, single_model=True
        )

        # Give the two main Plotly panes phone-sized heights. This is done in
        # Python (before first render) so Panel's Plotly view sizes the inner
        # container correctly from the start; with the figures on ``autosize``
        # the plot follows the pane. No client-side JS is needed, so nothing
        # re-resizes while the visitor scrolls (the old glitch).
        if self.interactive_embed_plot.get(0) is not None:
            self.interactive_embed_plot[0].height = _EMBED_HEIGHT
        if self.spectrogram_plot_panel.get(0) is not None:
            self.spectrogram_plot_panel[0].height = _SPEC_HEIGHT

        # Swipes over the plots scroll the page (see ``_without_drag``): the
        # figures already on the page and every spectrogram a click makes.
        for pane in (
            self.interactive_embed_plot.get(0),
            self.spectrogram_plot_panel.get(0),
        ):
            if pane is not None and pane.object is not None:
                _without_drag(pane.object)
        spec_plot = self.spec_plot_obj.get(0)
        if spec_plot is not None:
            spec_plot.update_spectrogram = _no_drag(spec_plot.update_spectrogram)

        # The mobile page needs the same logo + contact/newsletter footer the
        # desktop pages get. ``add_styling`` appends it to the sidebar, which
        # the single-model reorder already dissolves into the page's flex
        # column, so on a phone the footer lands at the bottom instead of being
        # missing altogether.
        self.add_styling(page)

        self.app = page
        # The one page this layout shows (there is no tab bar). The tour
        # (tour.py) only keeps steps on this page, so phones get a short tour
        # instead of steps on the All models / Predictions tabs.
        self.tab_names = ["Single model"]

        # Fluid width on narrow screens (media-query CSS; harmless everywhere).
        apply_mobile_styles(self.app)
