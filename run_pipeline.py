import bacpipe
import os
os.chdir('sites/bacpipe/bacpipe_for_website')
import panel as pn
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

def datasets_endpoint(request):
    """Serve available datasets as JSON."""
    datasets = get_available_datasets()
    return {'success': True, 'datasets': datasets}

from bacpipe.embedding_evaluation.visualization.dashboard import DashBoard
audio_dir = DashBoard.get_audio_dir()

# Register endpoint on Panel's Tornado server
# pn.serve(
#     ...,  # your create_dashboard callable from bacpipe.play()
#     endpoints={'/api/datasets': datasets_endpoint}
# )

bacpipe.play(audio_dir=audio_dir)
