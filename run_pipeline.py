import bacpipe
import os
os.chdir('sites/bacpipe/bacpipe_for_website')
import json
import panel as pn
from tornado.web import RequestHandler
from dataset_manager import get_available_datasets

bacpipe.settings.dashboard_port = 5006
bacpipe.settings.dashboard_websocket_origin = [
    'localhost:5006',
    'localhost:5177',
    'localhost:8000',
    'bacpipe.siriusly.me',
    'bacpipe.siriusly.me:80',
    'siriusly.me',
    'siriusly.me:80',
    'null',  # for iframes served from file:// or cross-origin contexts
]


class DatasetListHandler(RequestHandler):
    """Serve the available datasets as JSON for the site's dataset dropdown.

    The list is scanned from the filesystem on every request, so adding a
    dataset is just dropping its audio folder under ``public/assets/audio``
    (and its pipeline results under ``public/assets/embeddings``) — no code
    changes or server restarts needed.
    """

    def get(self):
        datasets = get_available_datasets()
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps({'success': True, 'datasets': datasets}))


from bacpipe.embedding_evaluation.visualization.dashboard import DashBoard
audio_dir = DashBoard.get_audio_dir()

# `main_results_dir` keeps the pipeline results next to the audio files, so a
# dataset is fully described by its two folders under public/assets/:
#   audio/<dataset>/       -> audio_dir
#   embeddings/<dataset>/  -> main_results_dir
bacpipe.play(
    audio_dir=audio_dir,
    main_results_dir='../public/assets/embeddings',
    extra_patterns=[(r"/api/datasets", DatasetListHandler)],
)
