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

from .dashboard import DashBoard, apply_mobile_styles

# Phone-friendly heights for the two Plotly panes that dominate the single-model
# page. Kept tall enough to stay readable but short enough to leave room for the
# spectrogram and the rest of the sidebar below on a phone screen.
_EMBED_HEIGHT = 430
_SPEC_HEIGHT = 400


class DashBoardMobile(DashBoard):
    """Single-model, mobile-first bacpipe dashboard."""

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

        # The mobile page needs the same logo + contact/newsletter footer the
        # desktop pages get. ``add_styling`` appends it to the sidebar, which
        # the single-model reorder already dissolves into the page's flex
        # column, so on a phone the footer lands at the bottom instead of being
        # missing altogether.
        self.add_styling(page)

        self.app = page

        # Fluid width on narrow screens (media-query CSS; harmless everywhere).
        apply_mobile_styles(self.app)
