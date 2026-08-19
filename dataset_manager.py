"""
Dataset manager for Bacpipe.

Scans the website's dataset directories and manages dataset selection.

Datasets live next to the submodule checkout under ``sites/bacpipe/public/assets/``:

- ``audio/<dataset>/``      — the raw audio files
- ``embeddings/<dataset>/`` — the pipeline results (embeddings,
  dimensionality-reduced embeddings and evaluations)

Only datasets that have been fully processed (audio + results present) are
reported as available, so the website's dataset dropdown never points at a
dataset that the dashboard can't visualize.

How a dataset actually gets read (there is no ``set_dataset`` call anywhere —
by design):

- The static page (``public/dashboard.html``) loads this module's
  ``get_available_datasets()`` through the Panel server's ``/api/datasets``
  endpoint and drops the choices into its dropdown.
- When a visitor picks a dataset, the page embeds the Panel app in an iframe
  with ``?audio_dir=<dataset>`` in the URL.
- The bacpipe package reads that per-visitor query parameter in
  ``DashBoard.get_audio_dir()`` and builds a fresh dashboard for exactly that
  dataset. Each visitor gets their own session, so there is no global
  "active dataset" to switch.
"""

from pathlib import Path


def get_assets_dir():
    """Get the website's data directory (``sites/bacpipe/public/assets``).

    Mirrors the ``../public/assets/audio/`` base used by
    ``DashBoard.get_audio_dir`` in the bacpipe package (both resolve from the
    submodule checkout), so the datasets listed here are exactly the ones the
    dashboard can load.
    """
    return Path(__file__).parent.parent / "public" / "assets"


def get_datasets_dir():
    """Get the path to the audio datasets directory."""
    return get_assets_dir() / "audio"


def get_results_dir():
    """Get the path to the per-dataset pipeline results (embeddings)."""
    return get_assets_dir() / "embeddings"


def _has_results(dataset_name):
    """Return True if the pipeline results for a dataset can be plotted.

    The dashboard needs both the raw embeddings and the dimensionality
    reduced embeddings, so both folders must exist and be non-empty.
    """
    results_dir = get_results_dir() / dataset_name
    for folder in ("embeddings", "dim_reduced_embeddings"):
        path = results_dir / folder
        if not path.is_dir() or not any(path.iterdir()):
            return False
    return True


def get_available_datasets():
    """
    Get the list of datasets that are ready to visualize.

    A dataset is listed when its audio directory exists and the pipeline has
    produced plottable results for it. New datasets only need to be dropped
    into the audio/ and embeddings/ folders — no code changes or restarts.

    Returns:
        list: List of dicts with 'name' and 'display_name' keys
    """
    audio_dir = get_datasets_dir()
    if not audio_dir.exists():
        return []

    datasets = []
    for item in sorted(audio_dir.iterdir()):
        if item.is_dir() and _has_results(item.name):
            datasets.append(
                {
                    "name": item.name,
                    "display_name": item.name.replace("_", " ").title(),
                }
            )
    return datasets


