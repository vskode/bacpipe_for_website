import bacpipe
import os
from pathlib import Path

# Always run from the submodule checkout: the dashboard resolves relative
# paths (../public/assets/...) against the CWD. This also lets launchers
# (serve_local.py, PM2, systemd, ...) start us from any working directory.
os.chdir(Path(__file__).resolve().parent)

import json
import threading
import time
import urllib.request
import urllib.parse

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


def start_prewarm(port=5006):
    """Warm a dashboard session for every available dataset at startup.

    The website embeds the dashboard in an iframe and reuses the *exact*
    session id (``bokeh-session-id=dash-<dataset>``) the pre-warm created.
    Bokeh builds a session's document synchronously while handling the first
    request for that session id, so when this function's GET returns, the
    dashboard for that dataset is fully built and cached in memory. The first
    (and every later) visitor for each dataset therefore gets an already-built
    dashboard instead of waiting minutes for it to be constructed.

    Sessions are re-opened every 4h to keep them alive: bokeh expires sessions
    that are not re-accessed within ``unused_session_lifetime_milliseconds``
    (set to 12h in ``bacpipe.play`` below). Builds are guarded by a free-RAM
    check plus a crash-backoff marker, so the heaviest dataset (AnuranSet, with
    a ~6GB embeddings folder) can never knock the small server over with an OOM
    during boot and then restart-loop forever. Overrides:
      BACPIPE_PREWARM=0        disable prewarming entirely
      BACPIPE_PREWARM_ALL=1    attempt every dataset regardless of free RAM
    """

    # Only attempt a build while the machine has this much free RAM. The
    # crash-backoff marker additionally skips a dataset for 30 minutes after a
    # build that ended in a process death (OOM), so PM2 restart loops are
    # impossible.
    MIN_FREE_MB_FOR_PREWARM = 2500
    CRASH_BACKOFF_SECONDS = 30 * 60
    STATE_FILE = Path("/tmp/bacpipe_prewarm_state.json")

    if os.environ.get("BACPIPE_PREWARM") == "0":
        print("[prewarm] disabled via BACPIPE_PREWARM=0", flush=True)
        return

    def _prewarm_all():
        return os.environ.get("BACPIPE_PREWARM_ALL") == "1"

    def _available_mem_mb():
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) // 1024
        except OSError:
            return 0

    def _crashed_recently(name):
        """True if the last prewarm of this dataset died mid-build recently.

        The marker is written just before a build starts and removed on
        success. If the process is killed (e.g. OOM) the marker survives, so a
        PM2 restart within the backoff window skips this dataset instead of
        crashing again.
        """
        try:
            if STATE_FILE.exists():
                state = json.loads(STATE_FILE.read_text())
                ts = state.get("started", {}).get(name)
                if ts is not None and time.time() - ts < CRASH_BACKOFF_SECONDS:
                    return True
        except Exception:
            pass
        return False

    def _mark_started(name):
        try:
            state = (
                json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
            )
            state.setdefault("started", {})[name] = time.time()
            STATE_FILE.write_text(json.dumps(state))
        except Exception:
            pass

    def _mark_done(name):
        try:
            state = (
                json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
            )
            state.get("started", {}).pop(name, None)
            STATE_FILE.write_text(json.dumps(state))
        except Exception:
            pass

    def _server_ready():
        url = f"http://127.0.0.1:{port}/api/datasets"
        for _ in range(90):  # ~3 minutes of retries
            try:
                with urllib.request.urlopen(url, timeout=3) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                time.sleep(2)
        return False

    def _warm(name):
        session_id = f"dash-{name}"
        url = (
            f"http://127.0.0.1:{port}/"
            f"?audio_dir={urllib.parse.quote(name)}"
            f"&bokeh-session-id={session_id}"
        )
        started = time.time()
        # A successful GET means the document build completed (bokeh builds it
        # synchronously while serving the first request for the session id). An
        # exception means the build failed — logged below so a broken dataset
        # is visible at deploy time instead of on the live site.
        with urllib.request.urlopen(url, timeout=3600) as resp:
            resp.read()
        print(
            f"[prewarm] {name}: session '{session_id}' ready "
            f"({time.time() - started:.1f}s)",
            flush=True,
        )

    def _loop():
        if not _server_ready():
            print("[prewarm] server did not become ready, skipping warm-up", flush=True)
            return
        print("[prewarm] warming sessions for all available datasets…", flush=True)
        while True:
            for ds in get_available_datasets():
                name = ds["name"]
                if _crashed_recently(name):
                    print(
                        f"[prewarm] skipping {name}: previous build crashed "
                        f"<{CRASH_BACKOFF_SECONDS // 60}min ago",
                        flush=True,
                    )
                    continue
                if not _prewarm_all() and _available_mem_mb() < MIN_FREE_MB_FOR_PREWARM:
                    print(
                        f"[prewarm] skipping {name}: only "
                        f"{_available_mem_mb()}MB free RAM "
                        f"(need {MIN_FREE_MB_FOR_PREWARM}MB)",
                        flush=True,
                    )
                    continue
                try:
                    _mark_started(name)
                    _warm(name)
                    _mark_done(name)
                except Exception as e:
                    print(f"[prewarm] {name}: warming failed: {e}", flush=True)
            time.sleep(4 * 60 * 60)  # re-open sessions to keep them alive

    threading.Thread(target=_loop, daemon=True, name="bacpipe-prewarm").start()


# Keep pre-warmed sessions alive for ~12h and give the bokeh session token a
# 24h lifetime. The token is embedded in the page a browser loads and is
# re-sent on every websocket (re)connect, so it must outlive the session
# itself: with a 1h token a visitor who keeps the dashboard open for longer
# than an hour gets "Token is expired" on reconnect (the recurring cron
# error). 24h comfortably covers the daily restart, so tokens never expire
# before their session is torn down.
start_prewarm()

bacpipe.play(
    audio_dir=audio_dir,
    main_results_dir='../public/assets/embeddings',
    extra_patterns=[(r"/api/datasets", DatasetListHandler)],
    session_token_expiration=24 * 60 * 60,
    unused_session_lifetime_milliseconds=12 * 60 * 60 * 1000,
    check_unused_sessions_milliseconds=5 * 60 * 1000,
)
