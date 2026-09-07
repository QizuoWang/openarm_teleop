"""Build a local, synchronized browser player for one raw episode."""

from bisect import bisect_left
import html
import json
import os
from pathlib import Path
import statistics

from .raw_dataset import load_records


CAMERAS = ("wrist_left", "ceiling", "wrist_right")


def _camera_frames(episode_path: Path, camera: str) -> list[tuple[int, Path]]:
    camera_path = episode_path / "cameras" / camera
    paths = list(camera_path.glob("*.jpeg")) + list(camera_path.glob("*.jpg"))
    frames = []
    for path in paths:
        try:
            timestamp_ns = int(path.stem)
        except ValueError:
            continue
        frames.append((timestamp_ns, path))
    return sorted(frames)


def _nearest_path(
    frames: list[tuple[int, Path]], timestamps: list[int], target_ns: int
) -> Path | None:
    if not frames:
        return None
    index = bisect_left(timestamps, target_ns)
    candidates = []
    if index < len(frames):
        candidates.append(frames[index])
    if index:
        candidates.append(frames[index - 1])
    return min(candidates, key=lambda item: abs(item[0] - target_ns))[1]


def _estimate_fps(timestamps: list[int]) -> float:
    intervals = [
        later - earlier
        for earlier, later in zip(timestamps, timestamps[1:])
        if later > earlier
    ]
    if not intervals:
        return 30.0
    fps = 1e9 / statistics.median(intervals)
    return min(120.0, max(1.0, fps))


def build_episode_player(
    raw_root: Path, report_root: Path, episode_id: int
) -> tuple[Path, int, float]:
    records = {record.episode_id: record for record in load_records(raw_root)}
    record = records.get(episode_id)
    if record is None:
        available = ", ".join(str(value) for value in sorted(records)) or "none"
        raise ValueError(f"episode {episode_id} not found; available episodes: {available}")

    camera_frames = {
        camera: _camera_frames(record.path, camera) for camera in CAMERAS
    }
    master_camera = "ceiling"
    if not camera_frames[master_camera]:
        master_camera = next(
            (camera for camera in CAMERAS if camera_frames[camera]), ""
        )
    if not master_camera:
        raise ValueError(f"episode {episode_id} has no camera frames")

    master_timestamps = [item[0] for item in camera_frames[master_camera]]
    timestamps_by_camera = {
        camera: [item[0] for item in frames]
        for camera, frames in camera_frames.items()
    }
    synchronized = []
    for target_ns in master_timestamps:
        frame = {}
        for camera in CAMERAS:
            path = _nearest_path(
                camera_frames[camera], timestamps_by_camera[camera], target_ns
            )
            frame[camera] = (
                None
                if path is None
                else Path(os.path.relpath(path, report_root)).as_posix()
            )
        synchronized.append(frame)

    fps = _estimate_fps(master_timestamps)
    duration_s = len(synchronized) / fps
    status = "accepted" if record.success else record.failure_reason or "failed"
    title = f"Episode {episode_id} · {status} · {duration_s:.1f}s"
    payload = json.dumps(synchronized, separators=(",", ":"))
    labels = json.dumps(list(CAMERAS))
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; background: #071019; color: #edf7ff; }}
    header {{ display:flex; align-items:center; gap:18px; padding:14px 18px;
              position:sticky; top:0; background:#0d1822; z-index:2; }}
    h1 {{ font-size:18px; margin:0; min-width:310px; }}
    button, select {{ font:inherit; padding:7px 12px; background:#183044;
                      color:#edf7ff; border:1px solid #36556d; border-radius:6px; }}
    input[type=range] {{ flex:1; }}
    #time {{ min-width:150px; text-align:right; font-variant-numeric:tabular-nums; }}
    main {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; padding:8px; }}
    figure {{ margin:0; background:#101d29; border:1px solid #294052; }}
    img {{ display:block; width:100%; aspect-ratio:16/9; object-fit:contain; background:#000; }}
    figcaption {{ padding:8px 10px; color:#a9bed0; }}
    footer {{ padding:8px 18px 18px; color:#91a6b7; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <button id="toggle">Pause</button>
    <select id="rate" aria-label="Playback speed">
      <option value="0.25">0.25×</option><option value="0.5">0.5×</option>
      <option value="1" selected>1×</option><option value="2">2×</option>
    </select>
    <input id="seek" type="range" min="0" max="{len(synchronized) - 1}" value="0">
    <span id="time"></span>
  </header>
  <main id="streams"></main>
  <footer>Space: play/pause · Left/Right: seek one second · Home/End: first/last frame</footer>
  <script>
    const frames = {payload};
    const cameras = {labels};
    const fps = {fps:.8f};
    const streams = document.getElementById('streams');
    const images = {{}};
    for (const camera of cameras) {{
      const figure = document.createElement('figure');
      const image = document.createElement('img');
      image.alt = camera.replaceAll('_', ' ');
      const caption = document.createElement('figcaption');
      caption.textContent = camera.replaceAll('_', ' ');
      figure.append(image, caption);
      streams.append(figure);
      images[camera] = image;
    }}
    const toggle = document.getElementById('toggle');
    const rate = document.getElementById('rate');
    const seek = document.getElementById('seek');
    const time = document.getElementById('time');
    let index = 0;
    let playing = true;
    let anchorIndex = 0;
    let anchorTime = performance.now();

    function clock(value) {{
      const seconds = value / fps;
      const minutes = Math.floor(seconds / 60);
      return `${{minutes}}:${{(seconds % 60).toFixed(1).padStart(4, '0')}}`;
    }}
    function render() {{
      const frame = frames[index];
      for (const camera of cameras) {{
        if (frame[camera] && images[camera].getAttribute('src') !== frame[camera]) {{
          images[camera].src = frame[camera];
        }}
      }}
      seek.value = index;
      time.textContent = `${{clock(index)}} / ${{clock(frames.length - 1)}} · frame ${{index + 1}}/${{frames.length}}`;
    }}
    function setIndex(value) {{
      index = Math.max(0, Math.min(frames.length - 1, value));
      anchorIndex = index;
      anchorTime = performance.now();
      render();
    }}
    function setPlaying(value) {{
      playing = value;
      anchorIndex = index;
      anchorTime = performance.now();
      toggle.textContent = playing ? 'Pause' : 'Play';
    }}
    function animate(now) {{
      if (playing) {{
        const next = anchorIndex + Math.floor((now - anchorTime) * fps * Number(rate.value) / 1000);
        if (next >= frames.length) {{
          setIndex(frames.length - 1);
          setPlaying(false);
        }} else if (next !== index) {{
          index = next;
          render();
        }}
      }}
      requestAnimationFrame(animate);
    }}
    toggle.addEventListener('click', () => setPlaying(!playing));
    rate.addEventListener('change', () => setIndex(index));
    seek.addEventListener('input', () => setIndex(Number(seek.value)));
    document.addEventListener('keydown', event => {{
      if (event.code === 'Space') {{ event.preventDefault(); setPlaying(!playing); }}
      else if (event.code === 'ArrowLeft') setIndex(index - Math.round(fps));
      else if (event.code === 'ArrowRight') setIndex(index + Math.round(fps));
      else if (event.code === 'Home') setIndex(0);
      else if (event.code === 'End') setIndex(frames.length - 1);
    }});
    render();
    requestAnimationFrame(animate);
  </script>
</body>
</html>
"""

    report_root.mkdir(parents=True, exist_ok=True)
    output = report_root / f"episode-{episode_id}-player.html"
    temporary = output.with_suffix(".html.tmp")
    temporary.write_text(page, encoding="utf-8")
    os.replace(temporary, output)
    return output, len(synchronized), fps
