"""The fleet smoke suite's machinery (#143): real PCC and music, on scratch state.

Session evals (``session_harness.py``) run conductor over ``fake_world.py``,
whose apps read conductor's words with rules. This harness puts the real seam
back: a second 12B in PCC or music reading conductor's ``ask``. It starts each
app's backend as a subprocess from its sibling checkout, with that checkout's
own ``.venv`` (code used read-only), and gives each one nothing real to touch:

- **its own scratch state**: a free loopback port (never a deployed or dev
  one, :data:`FORBIDDEN_PORTS`), a SQLite file migrated with the app's own
  ``alembic upgrade head``, a library and trace files, all in one temp folder.
  The subprocess runs from that folder with a minimal environment, so neither
  a checkout's ``.env`` nor this process's variables reach it;
- **no network**: music's ways out are yt-dlp and MusicBrainz. ``YT_DLP_BIN``
  (music's own test seam, ``music/backend/tests/conftest.py``) points it at
  :data:`FAKE_YT_DLP`, which writes a silent mp3 and logs its argv, and
  ``MUSICBRAINZ_BASE_URL`` at :class:`FakeMusicBrainz` on loopback, which
  knows ``fake_world``'s Caribou catalogue, so ``download_album`` resolves
  albums by name as it does in use (music#96, #102). The likes poller,
  Genius, Spotify and SoundCloud stay off or unconfigured, and every proxy
  variable points at a dead port with only loopback exempt, so a stray HTTP
  client fails instead of leaving. Music's download worker is off: a queued
  album stays queued, as it does for the minutes a real one takes, so the
  next turn finds it held;
- **a watch on both**: :class:`Watch` samples the processes' sockets (``ss``)
  and open files (``/proc``) the whole time; a peer off loopback, or a
  database, mp3 or trace file outside the scratch folder, is a finding.

Conductor finds the apps the way it always does: ``app.yaml`` manifests in
``fleet_manifest_dir``. :func:`write_fleet` copies the siblings' manifests
there with ``upstream`` rewritten to the scratch ports (chess's to a port
nothing listens on, so the planner's prompt is the production one and a
stray look fails). Every model call goes through :class:`Meter`, a recording
pass-through to llama-swap, which is how music's calls get cache numbers its
trace doesn't keep (#155).

Measured first, gated later: nothing here is a CI gate.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import yaml

from app import clock
from tests.fake_world import Album, Receipt, World
from tests.session_harness import TurnRecord

FLAG = "CONDUCTOR_FLEET_SMOKE"
ENABLED = os.environ.get(FLAG) == "1"

# Deployed stacks (8000 chess, 8100 PCC, 8300 conductor, 8500 music), their
# dev ports and llama-swap: a scratch app never binds one.
FORBIDDEN_PORTS = frozenset({8000, 8100, 8101, 8200, 8300, 8301, 8500, 8501})
# Where chess's manifest points: a privileged port nothing listens on.
DEAD_UPSTREAM = "127.0.0.1:1"
LLAMA_SWAP = os.environ.get("CONDUCTOR_FLEET_LLAMA", "http://127.0.0.1:8200")
_LOOPBACK = re.compile(r"^(127\.|::1$|::ffff:127\.|\*$|0\.0\.0\.0$|::$)")


def workspace_root() -> Path | None:
    """The folder holding the sibling checkouts: the nearest parent of this
    file with both apps in it (a worktree sits deeper than the checkout)."""
    override = os.environ.get("CONDUCTOR_FLEET_WORKSPACE")
    candidates = [Path(override)] if override else list(Path(__file__).resolve().parents)
    for parent in candidates:
        if (parent / "project-command-center" / "backend").is_dir() and (
            parent / "music" / "backend"
        ).is_dir():
            return parent
    return None


def missing_siblings() -> str | None:
    """Why the suite can't run here, or ``None`` when it can."""
    root = workspace_root()
    if root is None:
        return "no sibling project-command-center/ and music/ checkouts"
    for name in ("project-command-center", "music"):
        python = root / name / "backend" / ".venv" / "bin" / "python"
        if not python.exists():
            return f"{name} has no backend/.venv"
    return None


def free_port() -> int:
    """A loopback port nothing holds, never one of :data:`FORBIDDEN_PORTS`."""
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port not in FORBIDDEN_PORTS:
            return port


# --- the fake yt-dlp ------------------------------------------------------------------

# A real silent mp3 (216 bytes; music's tests/audio_fixtures.py, made with
# ffmpeg), so the file music tags is one mutagen can open.
TINY_MP3 = base64.b64decode(
    "/+MYxAAAAANIAAAAAExBTUUzLjEwMFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
    "VVVVVVVVVVVV/+MYxDsAAANIAAAAAFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
    "VVVVVVVVVVVVVVVVVVVVVVVV/+MYxHYAAANIAAAAAFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
    "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
)

# yt-dlp's wire shape as music's own fake has it (argv in, one JSON line per
# file out): a search finds nothing, a download writes the silent mp3 under
# --paths and names it from the words searched, a collection is three tracks.
FAKE_YT_DLP = r"""
import json, os, re, sys
from pathlib import Path

args = sys.argv[1:]
with open(os.environ["FAKE_YTDLP_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(args) + "\n")
if "--flat-playlist" in args:
    sys.exit(0)
target = args[-1]
words = re.sub(r"^[a-z]+search\d*:", "", target).strip() or "Track"
paths = Path(args[args.index("--paths") + 1])
paths.mkdir(parents=True, exist_ok=True)
audio = Path(os.environ["FAKE_YTDLP_AUDIO"]).read_bytes()
artist, _, title = words.partition(" - ")
if not title:
    artist, title = "", words


def emit(index, name, count=None):
    ident = "fake%d" % index
    path = paths / ("%s [%s].mp3" % (re.sub(r"[/\\]", " ", name), ident))
    path.write_bytes(audio)
    line = {"id": ident, "title": name, "filepath": str(path), "duration": 200 + index}
    if artist:
        line.update(artist=artist, track=name)
    if count is not None:
        line.update(playlist_index=index, playlist_count=count)
    print(json.dumps(line), flush=True)


if "--yes-playlist" in args:
    for index in range(1, 4):
        emit(index, "%s (track %d)" % (title, index), 3)
else:
    emit(1, title)
"""


# --- the fake MusicBrainz -------------------------------------------------------------------

# Every fake track is this long, so a matcher comparing lengths sees one.
FAKE_TRACK_MS = 200_000


class FakeMusicBrainz:
    """MusicBrainz's web service at the wire shape music reads (its
    ``musicbrainz.py``), on loopback, over a catalogue: ``fake_world``'s
    albums by default, so the smoke and the session evals name the same
    ones. A release-group search by artist (and title), an album's editions,
    an edition's tracks; a recording search (music's tagging) finds nothing.
    Records every path it was asked for."""

    def __init__(self, catalog: Iterable[Album] | None = None) -> None:
        self.catalog = list(catalog if catalog is not None else World().catalog)
        self.asked: list[str] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                fake.asked.append(self.path)
                body = fake.answer(self.path)
                data = json.dumps(body).encode()
                self.send_response(200 if body is not None else 404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/ws/2"

    def answer(self, path: str) -> dict[str, Any] | None:
        from urllib.parse import parse_qs, urlparse

        url = urlparse(path)
        params = {key: values[0] for key, values in parse_qs(url.query).items()}
        route = url.path.removeprefix("/ws/2/")
        if route == "release-group":
            query = params.get("query", "")
            artist = re.search(r'artist:"([^"]*)"', query)
            title = re.search(r'releasegroup:"([^"]*)"', query)
            found = [
                a
                for a in self.catalog
                if artist is not None
                and a.artist.casefold() == artist.group(1).replace("\\", "").casefold()
                and (title is None or a.title.casefold() == title.group(1).casefold())
            ]
            return {"release-groups": [self._group(a) for a in found]}
        if route == "release":
            album = self._by_id(params.get("release-group", ""))
            if album is None:
                return {"releases": []}
            return {
                "releases": [
                    {
                        "id": f"{album.mbid}-release",
                        "date": f"{album.year}-01-01",
                        "status": "Official",
                        "media": [{"track-count": len(album.tracks)}],
                    }
                ]
            }
        if route.startswith("release/"):
            album = self._by_id(route.removeprefix("release/").removesuffix("-release"))
            if album is None:
                return None
            tracks = [
                {"title": t, "length": FAKE_TRACK_MS, "artist-credit": [{"name": album.artist}]}
                for t in album.tracks
            ]
            return {"media": [{"tracks": tracks}]}
        if route == "recording":
            return {"recordings": []}
        return None

    def _by_id(self, mbid: str) -> Album | None:
        return next((a for a in self.catalog if a.mbid == mbid), None)

    @staticmethod
    def _group(album: Album) -> dict[str, Any]:
        artist_id = f"artist-{album.artist.casefold().replace(' ', '-')}"
        return {
            "id": album.mbid,
            "score": 100,
            "title": album.title,
            "primary-type": "Album",
            "secondary-types": [],
            "first-release-date": f"{album.year}-01-01",
            "artist-credit": [
                {"name": album.artist, "artist": {"id": artist_id, "name": album.artist}}
            ],
        }

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


# --- the call meter ----------------------------------------------------------------------


@dataclass
class MeteredCall:
    """One model call as llama-server answered it: whose (``tag``), when, how
    long, and the prompt cache's share of the prompt."""

    tag: str
    started: float
    ended: float
    ms: int
    status: int
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None
    prompt_ms: float | None = None
    predicted_ms: float | None = None


def read_usage(tag: str, started: float, ended: float, status: int, body: bytes) -> MeteredCall:
    call = MeteredCall(tag, started, ended, round((ended - started) * 1000), status)
    try:
        answer = json.loads(body)
    except ValueError:
        return call
    if not isinstance(answer, dict):
        return call
    usage = answer.get("usage") or {}
    timings = answer.get("timings") or {}
    call.prompt_tokens = usage.get("prompt_tokens")
    call.completion_tokens = usage.get("completion_tokens")
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    call.cached_tokens = cached if cached is not None else timings.get("cache_n")
    call.prompt_ms = timings.get("prompt_ms")
    call.predicted_ms = timings.get("predicted_ms")
    return call


class Meter:
    """A recording pass-through to llama-swap, one port per caller. It
    forwards each request and keeps what the answer said about tokens and the
    cache; the callers see llama-swap's answer byte for byte, **as it
    streams** (#200: a buffering meter handed a peek its first words only
    after the whole reply, so the fleet couldn't see the peek save anything).
    A caller that stops reading (a peek) closes the upstream stream too, as it
    would without the meter."""

    def __init__(self, upstream: str = LLAMA_SWAP) -> None:
        self.upstream = upstream.rstrip("/")
        self.calls: list[MeteredCall] = []
        self._lock = threading.Lock()
        self._servers: list[ThreadingHTTPServer] = []
        self._client = httpx.Client(timeout=httpx.Timeout(600.0, connect=5.0), trust_env=False)

    def url(self, tag: str) -> str:
        """A base URL (``…/v1``) whose calls are recorded under ``tag``."""
        meter = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _forward(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                headers = {
                    k: v
                    for k, v in self.headers.items()
                    if k.lower() not in ("host", "content-length", "connection")
                }
                started = time.time()
                request = meter._client.build_request(
                    self.command, meter.upstream + self.path, content=body, headers=headers
                )
                answer = meter._client.send(request, stream=True)
                kept = bytearray()
                try:
                    self.send_response(answer.status_code)
                    for key, value in answer.headers.items():
                        if key.lower() not in ("content-length", "transfer-encoding", "connection"):
                            self.send_header(key, value)
                    # HTTP/1.0 with no length: the body ends when the
                    # connection closes, so each chunk goes out as it comes.
                    self.send_header("Connection", "close")
                    self.end_headers()
                    for chunk in answer.iter_raw():
                        kept += chunk
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the caller stopped reading (a peek): so do we
                finally:
                    answer.close()
                ended = time.time()
                if self.command == "POST" and self.path.endswith("/chat/completions"):
                    with meter._lock:
                        meter.calls.append(
                            read_usage(tag, started, ended, answer.status_code, bytes(kept))
                        )

            do_GET = _forward
            do_POST = _forward

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self._servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}/v1"

    def between(self, started: float, ended: float, tag: str | None = None) -> list[MeteredCall]:
        with self._lock:
            calls = list(self.calls)
        return sorted(
            (c for c in calls if started <= c.started <= ended and (tag is None or c.tag == tag)),
            key=lambda c: c.started,
        )

    def close(self) -> None:
        for server in self._servers:
            server.shutdown()
            server.server_close()
        self._client.close()


# --- the watch ---------------------------------------------------------------------------

_WATCHED_SUFFIXES = (".db", ".sqlite", ".sqlite3", ".mp3", ".m4a", ".jsonl")


@dataclass
class Watch:
    """Samples the scratch processes' sockets and open files until stopped.

    ``egress`` collects any socket a scratch process (or a child it spawned)
    holds with a peer off loopback; ``strays`` any database, audio or trace
    file it holds outside ``root``. Both empty is the evidence that the run
    stayed inside its sandbox."""

    root: Path
    pgids: Callable[[], set[int]]
    interval: float = 0.25
    egress: set[str] = field(default_factory=set)
    strays: set[str] = field(default_factory=set)
    samples: int = 0
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def sample(self) -> None:
        groups = self.pgids()
        if not groups:
            return
        pids = _pids_in(groups)
        self.samples += 1
        for line in _ss_lines():
            owners = {int(pid) for pid in re.findall(r"pid=(\d+)", line)}
            if not owners & pids:
                continue
            peer = _peer(line)
            if peer is not None and not _LOOPBACK.match(peer):
                self.egress.add(line.strip())
        root = str(self.root.resolve())
        for pid in pids:
            for target in _open_files(pid):
                if target.endswith(_WATCHED_SUFFIXES) and not target.startswith(root):
                    self.strays.add(target)


def _pids_in(groups: set[int]) -> set[int]:
    found: set[int] = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.getpgid(int(entry.name)) in groups:
                found.add(int(entry.name))
        except (ProcessLookupError, PermissionError):
            continue
    return found


def _ss_lines() -> list[str]:
    try:
        done = subprocess.run(
            ["ss", "-tunapH"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return done.stdout.splitlines()


def _peer(line: str) -> str | None:
    """The peer host of one ``ss -tunapH`` line (``None`` for a listener)."""
    columns = line.split()
    if len(columns) < 6 or columns[1] == "LISTEN":
        return None
    host, _, _ = columns[5].rpartition(":")
    return host.strip("[]").split("%")[0] or None


def _open_files(pid: int) -> list[str]:
    fds = Path(f"/proc/{pid}/fd")
    found = []
    try:
        for fd in fds.iterdir():
            try:
                found.append(os.readlink(fd))
            except OSError:
                continue
    except OSError:
        return []
    return found


# --- one scratch app -----------------------------------------------------------------------


@dataclass
class ScratchApp:
    """One sibling backend on scratch state. ``template`` is its migrated
    database; every :meth:`start` runs on a fresh copy of it."""

    name: str
    manifest_dir: str
    backend: Path
    port: int
    root: Path
    env: dict[str, str]
    template: Path | None = None
    process: subprocess.Popen[bytes] | None = None
    started: int = 0

    @property
    def python(self) -> Path:
        return self.backend / ".venv" / "bin" / "python"

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def db(self) -> Path:
        return self.root / f"{self.name}.db"

    def migrate(self) -> None:
        """``alembic upgrade head``, as the app's main.sh runs it (from
        backend/, so its alembic.ini resolves), into the template database.
        DATABASE_URL in the environment wins over the checkout's .env."""
        self.template = self.root / f"{self.name}-template.db"
        env = {**self.env, "DATABASE_URL": f"sqlite:///{self.template}"}
        done = subprocess.run(
            [str(self.backend / ".venv" / "bin" / "alembic"), "upgrade", "head"],
            cwd=self.backend,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if done.returncode != 0 or not self.template.exists():
            raise RuntimeError(f"{self.name} migration failed:\n{done.stdout}\n{done.stderr}")

    def start(self) -> None:
        assert self.template is not None, "migrate first"
        assert self.port not in FORBIDDEN_PORTS
        shutil.copyfile(self.template, self.db)
        self.started += 1
        log = (self.root / f"{self.name}.log").open("ab")
        self.process = subprocess.Popen(
            [
                str(self.python),
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--app-dir",
                str(self.backend),
            ],
            # A scratch cwd: no checkout's .env is read.
            cwd=self.root,
            env={**self.env, "DATABASE_URL": f"sqlite:///{self.db}"},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.close()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.name} exited at boot:\n{self.log_tail()}")
            try:
                if httpx.get(f"{self.base}/health", timeout=1.0).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        raise RuntimeError(f"{self.name} never answered /health:\n{self.log_tail()}")

    def stop(self) -> None:
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        except ProcessLookupError:
            pass

    def log_tail(self, lines: int = 40) -> str:
        path = self.root / f"{self.name}.log"
        if not path.exists():
            return "(no log)"
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])

    def get(self, path: str, **params: Any) -> Any:
        response = httpx.get(f"{self.base}{path}", params=params, timeout=10.0)
        response.raise_for_status()
        return response.json()

    def post(self, path: str, body: Mapping[str, Any]) -> Any:
        response = httpx.post(f"{self.base}{path}", json=dict(body), timeout=10.0)
        response.raise_for_status()
        return response.json()


def _base_env(root: Path, tag_url: str) -> dict[str, str]:
    """What every scratch process gets: a minimal environment, the household's
    zone, the metered model, and proxies that lead nowhere."""
    dead = "http://127.0.0.1:9"
    env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL") if key in os.environ}
    env.update(
        HOME=str(root / "home"),
        TZ=clock.zone().key,
        APP_ENV="development",
        API_HOST="127.0.0.1",
        LLAMACPP_BASE_URL=tag_url,
        HTTP_PROXY=dead,
        HTTPS_PROXY=dead,
        ALL_PROXY=dead,
        http_proxy=dead,
        https_proxy=dead,
        all_proxy=dead,
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
    )
    (root / "home").mkdir(parents=True, exist_ok=True)
    return env


def pcc_app(backend: Path, root: Path, meter: Meter) -> ScratchApp:
    root = root / "pcc"
    root.mkdir(parents=True)
    env = _base_env(root, meter.url("tasks"))
    env.update(PCC_TRACE_PATH=str(root / "trace.jsonl"), TRASH_RETENTION_DAYS="0")
    return ScratchApp("tasks", "project-command-center", backend, free_port(), root, env)


def music_app(backend: Path, root: Path, meter: Meter, musicbrainz: str) -> ScratchApp:
    root = root / "music"
    root.mkdir(parents=True)
    fake = root / "fake-yt-dlp"
    fake.write_text(f"#!{sys.executable}\n{FAKE_YT_DLP}", encoding="utf-8")
    fake.chmod(0o755)
    audio = root / "silent.mp3"
    audio.write_bytes(TINY_MP3)
    env = _base_env(root, meter.url("music"))
    env.update(
        MUSIC_TRACE_PATH=str(root / "trace.jsonl"),
        MUSIC_DOWNLOAD_DIR=str(root / "library"),
        YT_DLP_BIN=str(fake),
        YT_DLP_CACHE_DIR=str(root / "yt-dlp-cache"),
        FAKE_YTDLP_LOG=str(root / "yt-dlp-argv.jsonl"),
        FAKE_YTDLP_AUDIO=str(audio),
        MUSICBRAINZ_ENABLED="true",
        MUSICBRAINZ_BASE_URL=musicbrainz,
        MUSICBRAINZ_MIN_INTERVAL_SECONDS="0",
        LIKES_WORKER_ENABLED="false",
        # An album stays queued, held, as it is for the minutes a real one
        # takes: the fake yt-dlp's searches find nothing, so a running album
        # would fail at once and hold nothing (music#102).
        DOWNLOAD_WORKER_ENABLED="false",
    )
    return ScratchApp("music", "music", backend, free_port(), root, env)


# --- the manifests conductor discovers -----------------------------------------------------


def rewrite_manifest(text: str, upstream: str) -> str:
    """A sibling's ``app.yaml`` with its ``upstream`` pointed elsewhere; every
    other field (the descriptions and examples the planner reads) as is."""
    manifest = yaml.safe_load(text)
    if not isinstance(manifest, dict) or "upstream" not in manifest:
        raise ValueError("not an app manifest")
    manifest["upstream"] = upstream
    return yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)


def write_fleet(
    target: Path, workspace: Path, upstreams: Mapping[str, str], dead: Iterable[str] = ("chess",)
) -> Path:
    """``target/<dir>/app.yaml`` for each scratch app (``upstreams``: sibling
    folder → host:port) and for each ``dead`` sibling, pointed at nothing."""
    pointed = {**{name: DEAD_UPSTREAM for name in dead}, **upstreams}
    for folder, upstream in pointed.items():
        source = workspace / folder / "app.yaml"
        if not source.exists():
            continue
        (target / folder).mkdir(parents=True, exist_ok=True)
        (target / folder / "app.yaml").write_text(
            rewrite_manifest(source.read_text(encoding="utf-8"), upstream), encoding="utf-8"
        )
    return target


# --- seeds ---------------------------------------------------------------------------------

# fake_world's seed, minus the parts no smoke scenario reads: three projects,
# Kitchen's three tasks among them (S9 wipes it).
SEED_TASKS: tuple[tuple[str, str, int | None], ...] = (
    ("groceries", "Household", 0),
    ("call mom", "Household", 0),
    ("renew passport", "Household", -4),
    ("pay water bill", "Household", -2),
    ("buy new sink", "Kitchen", 10),
    ("paint cabinets", "Kitchen", None),
    ("fix the drawer", "Kitchen", None),
    ("garage shelves", "Garage", None),
    ("clear out old bikes", "Garage", None),
    ("sweep the floor", "Garage", None),
)


def seed_pcc(pcc: ScratchApp, today: date) -> set[int]:
    """The seed through PCC's own REST API; the seeded task ids."""
    projects: dict[str, int] = {}
    ids: set[int] = set()
    for title, project, offset in SEED_TASKS:
        if project not in projects:
            projects[project] = int(pcc.post("/api/projects", {"name": project})["id"])
        body: dict[str, Any] = {"title": title, "project_id": projects[project]}
        if offset is not None:
            body["due_date"] = (today + timedelta(days=offset)).isoformat()
        ids.add(int(pcc.post("/api/tasks", body)["id"]))
    return ids


# --- the fleet -----------------------------------------------------------------------------


class ScratchFleet:
    """PCC and music on scratch state behind one meter and one watch.
    :meth:`reset` is a fresh sample: both restarted on fresh databases, on
    the same ports (conductor discovered them once), PCC seeded."""

    def __init__(self, root: Path) -> None:
        workspace = workspace_root()
        assert workspace is not None, missing_siblings()
        self.workspace = workspace
        self.root = root
        self.meter = Meter()
        self.musicbrainz = FakeMusicBrainz()
        self.pcc = pcc_app(workspace / "project-command-center" / "backend", root, self.meter)
        self.music = music_app(
            workspace / "music" / "backend", root, self.meter, self.musicbrainz.url
        )
        self.apps = (self.pcc, self.music)
        self.conductor_llm = self.meter.url("conductor")
        self.conductor_trace = root / "conductor-trace.jsonl"
        self.seeded: set[int] = set()
        self.today = clock.local_today()
        self.watch = Watch(root, self._groups)

    def _groups(self) -> set[int]:
        return {app.process.pid for app in self.apps if app.process is not None}

    def upstreams(self) -> dict[str, str]:
        return {app.manifest_dir: f"127.0.0.1:{app.port}" for app in self.apps}

    @property
    def allowed(self) -> frozenset[str]:
        """The upstreams conductor may dial: the scratch apps' and nothing else."""
        return frozenset(self.upstreams().values())

    def start(self) -> None:
        for app in self.apps:
            app.migrate()
        self.watch.start()
        self.reset()

    def reset(self) -> None:
        for app in self.apps:
            app.stop()
            shutil.rmtree(app.root / "library", ignore_errors=True)
            app.start()
        self.today = clock.local_today()
        self.seeded = seed_pcc(self.pcc, self.today)

    def close(self) -> None:
        for app in self.apps:
            app.stop()
        self.watch.sample()
        self.watch.stop()
        self.meter.close()
        self.musicbrainz.close()

    # -- reads of the apps' end state, through their own APIs --

    def tasks(self) -> list[dict[str, Any]]:
        return list(self.pcc.get("/api/tasks", limit=1000))

    def new_tasks(self) -> list[dict[str, Any]]:
        return [t for t in self.tasks() if t["id"] not in self.seeded]

    def projects(self) -> list[dict[str, Any]]:
        return list(self.pcc.get("/api/projects", include_closed=True))

    def trash(self) -> dict[str, Any]:
        return dict(self.pcc.get("/api/trash"))

    def jobs(self) -> list[dict[str, Any]]:
        return list(self.music.get("/api/downloads/jobs")["recent"])

    def yt_dlp_calls(self) -> list[list[str]]:
        log = self.music.root / "yt-dlp-argv.jsonl"
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines() if line]

    def library(self) -> list[str]:
        folder = self.music.root / "library"
        return sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())


# --- receipts from the real apps' records ----------------------------------------------------

_FAMILIES: tuple[tuple[str, str], ...] = (
    ("download_", "queue"),
    ("find_song_by_lyrics", "queue"),
    ("sort_music", "file"),
    ("create_", "create"),
    ("add_", "create"),
    ("trash_", "trash"),
    ("remove_", "trash"),
    ("complete_", "complete"),
    ("cancel", "cancel"),
)


def family(tool: str) -> str:
    """The receipt family (``receipt_claims``' kinds) of an app tool's write."""
    return next((kind for prefix, kind in _FAMILIES if tool.startswith(prefix)), "update")


def receipts_from(record: TurnRecord) -> list[Receipt]:
    """A turn's writes as world receipts, read off the apps' typed records
    (delegate v1.1) the way conductor's data flow reads them: each ``done``
    entry's titled refs, job refs skipped unless the job is all it names (a
    download job's title is the request it was given)."""
    found: list[Receipt] = []
    for call in record.tools:
        tool = str(call.get("tool", ""))
        # Conductor's own memory app is the session harness's (memory_writes).
        if tool == "ask_memory" or not (tool.startswith("ask_") or tool == "confirm"):
            continue
        for entry in call.get("app_record") or []:
            if entry.get("outcome") != "done":
                continue
            refs = [r for r in entry.get("touched") or [] if r.get("title")]
            things = [r for r in refs if not str(r.get("kind", "")).endswith("job")] or refs
            for ref in things:
                found.append(
                    Receipt(
                        str(ref.get("app") or call.get("tool", "")).removeprefix("ask_"),
                        family(str(entry.get("tool", ""))),
                        f"{ref.get('kind')}:{ref.get('id')}",
                        str(ref["title"]),
                        record.index,
                        {"tool": entry.get("tool"), "ref_kind": ref.get("kind")},
                    )
                )
    return found


def requests_sent(record: TurnRecord) -> list[str]:
    """What conductor sent each app this turn, as ``app: request``."""
    sent = []
    for call in record.tools:
        tool = str(call.get("tool", ""))
        args = call.get("arguments") or {}
        if tool == "confirm":
            sent.append(f"confirm: the yes, to {args.get('app')}")
        elif tool.startswith("ask_"):
            # A step that didn't run still shows its words, ``{sN}`` unfilled.
            request = args.get("message") or args.get("request") or ""
            sent.append(f"{tool}: {request!r}" + (" (not run)" if call.get("error") else ""))
        elif tool.startswith(("look_", "open_")):
            sent.append(tool)
    return sent


# --- #155: one delegated write, joined across the three traces -----------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@dataclass
class CallRow:
    who: str  # "conductor" | "tasks" | "music"
    role: str
    ms: int
    prompt_tokens: int | None
    cached_tokens: int | None
    prompt_ms: float | None


@dataclass
class TurnCost:
    run_id: str
    utterance: str
    total_ms: int
    calls: list[CallRow]
    # Each hop's app run, as its own trace has it: (app, model calls, ms).
    app_runs: list[tuple[str, int, int]]


def _epoch(stamp: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(stamp).timestamp()


def turn_costs(
    conductor: list[dict[str, Any]],
    apps: Mapping[str, list[dict[str, Any]]],
    meter: Meter,
) -> list[TurnCost]:
    """Each conductor turn's model calls in order, conductor's and the apps'.

    The turn is conductor's trace record (``run_id``, sent to the app as
    ``X-Agent-Turn``). Neither app records that header, so a hop's app run is
    the app trace record on the hop's thread (``conversation_id`` =
    ``thread_id``) that started inside the turn. Every call's numbers are the
    meter's (llama-server's own answer), named by the trace that made it:
    conductor's ``role``, PCC's ``phase``; music's trace has neither a phase
    nor cache numbers, so its calls are numbered."""
    costs = []
    for turn in conductor:
        started, ended = _epoch(turn["started"]), _epoch(turn["ts"])
        roles = [str(c.get("role", "?")) for c in turn.get("calls") or []]
        rows: list[CallRow] = []
        metered = meter.between(started - 0.5, ended + 0.5)
        mine = iter(roles)
        runs: list[tuple[str, int, int]] = []
        phases: dict[str, list[str]] = {}
        for hop in turn.get("hops") or []:
            app = str(hop.get("app"))
            for run in apps.get(app, []):
                if run.get("conversation_id") == hop.get("thread_id") and (
                    started - 0.5 <= _epoch(run["started"]) <= ended + 0.5
                ):
                    calls = run.get("calls") or []
                    runs.append((app, len(calls), int(run.get("total_ms") or 0)))
                    phases.setdefault(app, []).extend(
                        str(c.get("phase") or f"call {n + 1}") for n, c in enumerate(calls)
                    )
        app_turn = {app: iter(names) for app, names in phases.items()}
        for call in metered:
            if call.tag == "conductor":
                role = next(mine, "?")
            else:
                role = next(app_turn.get(call.tag, iter(())), "?")
            rows.append(
                CallRow(
                    call.tag, role, call.ms, call.prompt_tokens, call.cached_tokens, call.prompt_ms
                )
            )
        costs.append(
            TurnCost(
                str(turn.get("run_id")),
                str(turn.get("utterance")),
                int(turn.get("total_ms") or round((ended - started) * 1000)),
                rows,
                runs,
            )
        )
    return costs


def cost_lines(cost: TurnCost) -> list[str]:
    def num(value: int | float | None) -> str:
        return "-" if value is None else f"{value:.0f}"

    lines = [
        f"[fleet-cost] {cost.utterance!r} total={cost.total_ms}ms calls={len(cost.calls)} "
        f"app_runs={cost.app_runs}"
    ]
    for row in cost.calls:
        lines.append(
            f"[fleet-cost]   {row.who:<9} {row.role:<10} {row.ms:>6}ms "
            f"prompt={num(row.prompt_tokens)} cached={num(row.cached_tokens)} "
            f"prefill={num(row.prompt_ms)}ms"
        )
    return lines


__all__ = [
    "ENABLED",
    "FLAG",
    "Meter",
    "ScratchFleet",
    "Watch",
    "cost_lines",
    "family",
    "missing_siblings",
    "receipts_from",
    "requests_sent",
    "rewrite_manifest",
    "turn_costs",
    "write_fleet",
]
