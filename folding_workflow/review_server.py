"""Local-only HTTP server for reviewing and relabeling finalized episodes."""

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html
import mimetypes
from pathlib import Path
import secrets
import subprocess
from typing import Callable
from urllib.parse import parse_qs, quote, urlparse

from .visualize import build_episode_player


def serve_review(
    *,
    root: Path,
    raw_root: Path,
    report_root: Path,
    player_links: dict[int, str],
    render_page: Callable,
    relabel_episode: Callable,
    port: int,
    open_browser: bool,
) -> None:
    """Serve review assets and guarded relabel POSTs on loopback only."""

    csrf_token = secrets.token_urlsafe(32)
    last_message = {"value": None}

    class ReviewHandler(BaseHTTPRequestHandler):
        def _send_bytes(self, payload: bytes, content_type: str, status=HTTPStatus.OK):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _send_error_page(self, message: str, status=HTTPStatus.CONFLICT):
            payload = (
                "<!doctype html><meta charset=utf-8><title>Relabel blocked</title>"
                f"<h1>Relabel blocked</h1><p>{html.escape(message)}</p>"
                '<p><a href="/">Return to review</a></p>'
            ).encode()
            self._send_bytes(payload, "text/html; charset=utf-8", status)

        def do_GET(self):
            request_path = urlparse(self.path).path
            if request_path == "/":
                payload = render_page(
                    raw_root,
                    player_links,
                    csrf_token=csrf_token,
                    message=last_message["value"],
                ).encode()
                last_message["value"] = None
                self._send_bytes(payload, "text/html; charset=utf-8")
                return

            if request_path.startswith("/episode-") and request_path.endswith(
                "-player.html"
            ):
                candidate = (report_root / request_path.removeprefix("/")).resolve()
                allowed = candidate.is_relative_to(report_root.resolve())
            elif request_path.startswith("/folding_data/"):
                candidate = (root / request_path.removeprefix("/")).resolve()
                allowed = candidate.is_relative_to((root / "folding_data").resolve()) and (
                    candidate.suffix.lower() in {".jpeg", ".jpg"}
                )
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            if not allowed or not candidate.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            self._send_bytes(candidate.read_bytes(), content_type)

        def do_POST(self):
            if urlparse(self.path).path != "/relabel":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = 0
            if content_length <= 0 or content_length > 8192:
                self._send_error_page("Invalid request size", HTTPStatus.BAD_REQUEST)
                return
            fields = parse_qs(self.rfile.read(content_length).decode())
            if fields.get("csrf_token", [""])[0] != csrf_token:
                self._send_error_page("Invalid review-session token", HTTPStatus.FORBIDDEN)
                return
            try:
                episode_number = int(fields.get("episode", [""])[0])
                status = fields.get("status", [""])[0]
                failure_reason = fields.get("failure_reason", [None])[0]
                if status == "accepted":
                    failure_reason = None
                message = relabel_episode(
                    raw_root, episode_number, status, failure_reason
                )
                player, _, _ = build_episode_player(
                    raw_root, report_root, episode_number
                )
                player_links[episode_number] = player.name
            except (OSError, ValueError, SystemExit) as error:
                self._send_error_page(str(error))
                return
            last_message["value"] = message + "; raw files preserved"
            location = f"/?updated={quote(str(episode_number))}#episode-{episode_number}"
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", location)
            self.end_headers()

        def log_message(self, format, *args):
            print(f"review-ui: {format % args}")

    server = ThreadingHTTPServer(("127.0.0.1", port), ReviewHandler)
    url = f"http://127.0.0.1:{port}"
    print(f"Interactive review: {url}")
    print("Press Ctrl+C to stop the review server")
    if open_browser:
        subprocess.Popen(
            ["xdg-open", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
