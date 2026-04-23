from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import tempfile
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from PIL.ImageQt import ImageQt

os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

from PySide6.QtCore import QObject, QSignalBlocker, Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QPlainTextEdit,
    QSizePolicy,
    QSplitter,
    QStyle,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
    QSpinBox,
)

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma"}

VIDEO_RESOLUTIONS = {
    "HD 1280x720": (1280, 720),
    "Full HD 1920x1080": (1920, 1080),
    "2K 2560x1440": (2560, 1440),
    "UHD 3840x2160": (3840, 2160),
    "Square 1080x1080": (1080, 1080),
    "Vertical 1080x1920": (1080, 1920),
    "Vertical 720x1280": (720, 1280),
}

TRANSITIONS = ["Crossfade", "Slide Left", "Slide Right", "Dip to Black"]
BACKGROUND_MODES = ["Black", "White", "Blurred"]
FPS_OPTIONS = [24, 25, 30, 50, 60]


@dataclass
class SlideSettings:
    fps: int
    resolution: Tuple[int, int]
    transition_mode: str
    background_mode: str
    output_path: str

    target_duration: float
    min_hold_seconds: float
    max_hold_seconds: float
    transition_seconds: float
    motion_intensity: float

    audio_paths: List[str]
    audio_mode: str  # none, single_loop, playlist_once, playlist_loop


@dataclass
class CandidateImage:
    path: str
    clarity_score: float
    hash_bits: np.ndarray
    novelty_score: float = 1.0
    final_score: float = 0.0


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def ease_in_out(t: float) -> float:
    return 0.5 * (1.0 - math.cos(math.pi * clamp(t, 0.0, 1.0)))


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS


def is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTS


def list_images_in_folder(folder: str, recursive: bool = False) -> List[str]:
    p = Path(folder)
    if not p.exists():
        return []

    if recursive:
        files = [f for f in p.rglob("*") if is_image_file(f)]
    else:
        files = [f for f in p.iterdir() if is_image_file(f)]

    return [str(f) for f in sorted(files)]


def list_audio_in_folder(folder: str, recursive: bool = False) -> List[str]:
    p = Path(folder)
    if not p.exists():
        return []

    if recursive:
        files = [f for f in p.rglob("*") if is_audio_file(f)]
    else:
        files = [f for f in p.iterdir() if is_audio_file(f)]

    return [str(f) for f in sorted(files)]


def seconds_to_hms(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def load_and_normalise_image(path: str) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def pil_to_bgr_bytes(img: Image.Image) -> bytes:
    arr = np.asarray(img.convert("RGB"))
    arr = arr[:, :, ::-1]
    return arr.tobytes()


def average_hash_bits(img: Image.Image, hash_size: int = 8) -> np.ndarray:
    g = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(g, dtype=np.float32)
    mean = arr.mean()
    return (arr >= mean).astype(np.uint8).flatten()


def hamming_distance_bits(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def clarity_score(img: Image.Image) -> float:
    g = np.asarray(
        img.convert("L").resize((512, 512), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    edges = (dx.mean() + dy.mean()) / 2.0
    contrast = g.std()
    return float(edges * 0.7 + contrast * 0.3)


def make_background_canvas(img: Image.Image, target_size: Tuple[int, int], mode: str) -> Image.Image:
    tw, th = target_size

    if mode == "White":
        return Image.new("RGB", (tw, th), (255, 255, 255))

    if mode == "Blurred":
        bg = ImageOps.fit(img, (tw, th), method=Image.Resampling.LANCZOS)
        bg = bg.filter(ImageFilter.GaussianBlur(radius=24))
        arr = np.asarray(bg).astype(np.float32) * 0.72
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    return Image.new("RGB", (tw, th), (0, 0, 0))


def fit_image_with_padding(
    img: Image.Image,
    target_size: Tuple[int, int],
    bg_canvas: Image.Image,
    zoom_factor: float = 1.0,
    pan_x: float = 0.0,
    pan_y: float = 0.0,
) -> Image.Image:
    tw, th = target_size
    iw, ih = img.size
    scale = min(tw / iw, th / ih)
    
    current_scale = scale * zoom_factor
    a = 1.0 / current_scale
    e = 1.0 / current_scale

    new_w = iw * current_scale
    new_h = ih * current_scale

    overflow_x = max(0.0, new_w - tw)
    overflow_y = max(0.0, new_w * (ih / iw) - th)

    shift_target_x = pan_x * (overflow_x / 2.0)
    shift_target_y = pan_y * (overflow_y / 2.0)

    center_x = tw / 2.0 - shift_target_x
    center_y = th / 2.0 - shift_target_y

    c = iw / 2.0 - center_x * a
    f = ih / 2.0 - center_y * e

    # SSAA 2.0x: Render at double resolution then downsample for perfect anti-aliasing
    ss_tw, ss_th = tw * 2, th * 2
    ss_a, ss_c, ss_e, ss_f = a / 2.0, c, e / 2.0, f

    img_rgba = img.convert("RGBA")
    ss_transformed = img_rgba.transform(
        (ss_tw, ss_th),
        Image.AFFINE,
        data=(ss_a, 0.0, ss_c, 0.0, ss_e, ss_f),
        resample=Image.Resampling.BICUBIC
    )
    
    transformed = ss_transformed.resize((tw, th), Image.Resampling.LANCZOS)
    bg = bg_canvas.copy()
    bg.paste(transformed, (0, 0), transformed)
    return bg


class MotionPlan:
    def __init__(
        self,
        start_zoom: float,
        end_zoom: float,
        start_pan_x: float,
        end_pan_x: float,
        start_pan_y: float,
        end_pan_y: float,
    ):
        self.start_zoom = start_zoom
        self.end_zoom = end_zoom
        self.start_pan_x = start_pan_x
        self.end_pan_x = end_pan_x
        self.start_pan_y = start_pan_y
        self.end_pan_y = end_pan_y

    def state_at(self, t: float) -> Tuple[float, float, float]:
        e = ease_in_out(t)
        zoom = self.start_zoom + (self.end_zoom - self.start_zoom) * e
        pan_x = self.start_pan_x + (self.end_pan_x - self.start_pan_x) * e
        pan_y = self.start_pan_y + (self.end_pan_y - self.start_pan_y) * e
        return zoom, pan_x, pan_y


def build_motion_plans_for_selected(selected_paths: List[str], intensity: float = 1.0) -> List[MotionPlan]:
    plans: List[MotionPlan] = []
    prev_hash: Optional[np.ndarray] = None
    prev_plan: Optional[MotionPlan] = None

    for i, path in enumerate(selected_paths):
        img = load_and_normalise_image(path)
        hb = average_hash_bits(img)
        rng = random.Random(1000 + i)

        if prev_hash is not None:
            dist = hamming_distance_bits(prev_hash, hb)
        else:
            dist = 999

        similar_to_previous = dist <= 10

        if similar_to_previous and prev_plan is not None:
            start_zoom = prev_plan.end_zoom
            end_zoom = clamp(start_zoom + rng.uniform(0.004, 0.012) * intensity, 1.0, 2.0)

            start_pan_x = prev_plan.end_pan_x
            start_pan_y = prev_plan.end_pan_y
            end_pan_x = clamp(start_pan_x + rng.uniform(-0.08, 0.08) * intensity, -1.0, 1.0)
            end_pan_y = clamp(start_pan_y + rng.uniform(-0.08, 0.08) * intensity, -1.0, 1.0)
        else:
            start_zoom = 1.0 + rng.uniform(0.0, 0.015) * intensity
            end_zoom = clamp(start_zoom + rng.uniform(0.01, 0.04) * intensity, 1.0, 2.0)

            start_pan_x = rng.uniform(-0.20, 0.20) * intensity
            start_pan_y = rng.uniform(-0.20, 0.20) * intensity
            end_pan_x = clamp(start_pan_x + rng.uniform(-0.15, 0.15) * intensity, -1.0, 1.0)
            end_pan_y = clamp(start_pan_y + rng.uniform(-0.15, 0.15) * intensity, -1.0, 1.0)

        plan = MotionPlan(
            start_zoom=start_zoom,
            end_zoom=end_zoom,
            start_pan_x=start_pan_x,
            end_pan_x=end_pan_x,
            start_pan_y=start_pan_y,
            end_pan_y=end_pan_y,
        )
        plans.append(plan)
        prev_hash = hb
        prev_plan = plan

    return plans


def blend_images(a: Image.Image, b: Image.Image, alpha: float) -> Image.Image:
    return Image.blend(a, b, clamp(alpha, 0.0, 1.0))


def slide_transition(a: Image.Image, b: Image.Image, alpha: float, direction: str) -> Image.Image:
    alpha = clamp(alpha, 0.0, 1.0)
    w, h = a.size
    frame = Image.new("RGB", (w, h), (0, 0, 0))
    offset = int(w * ease_in_out(alpha))

    if direction == "Slide Right":
        a_x = offset
        b_x = -w + offset
    else:
        a_x = -offset
        b_x = w - offset

    frame.paste(a, (a_x, 0))
    frame.paste(b, (b_x, 0))
    return frame


def dip_to_black_transition(a: Image.Image, b: Image.Image, alpha: float) -> Image.Image:
    alpha = clamp(alpha, 0.0, 1.0)
    black = Image.new("RGB", a.size, (0, 0, 0))
    if alpha < 0.5:
        local = alpha / 0.5
        return Image.blend(a, black, ease_in_out(local))
    local = (alpha - 0.5) / 0.5
    return Image.blend(black, b, ease_in_out(local))


def make_transition_frame(a: Image.Image, b: Image.Image, alpha: float, mode: str) -> Image.Image:
    if mode == "Slide Left":
        return slide_transition(a, b, alpha, "Slide Left")
    if mode == "Slide Right":
        return slide_transition(a, b, alpha, "Slide Right")
    if mode == "Dip to Black":
        return dip_to_black_transition(a, b, alpha)
    return blend_images(a, b, ease_in_out(alpha))


class SlideshowExporter:
    def __init__(self, image_paths: List[str], settings: SlideSettings, progress_cb, log_cb):
        self.image_paths = image_paths
        self.settings = settings
        self.progress_cb = progress_cb
        self.log_cb = log_cb
        self.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        self.ffprobe = str(Path(self.ffmpeg).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe"))

    def log(self, msg: str) -> None:
        if self.log_cb:
            self.log_cb(msg)

    def progress(self, value: float, status: str) -> None:
        if self.progress_cb:
            self.progress_cb(value, status)

    def run_ffmpeg(self, cmd: List[str], label: str) -> None:
        self.log(label)
        self.log(" ".join(f'"{c}"' if " " in c else c for c in cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"{label} failed.\n\nSTDERR:\n{result.stderr}")

    def probe_audio_duration(self, path: str) -> float:
        cmd = [
            self.ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Could not probe audio duration for:\n{path}\n\n{result.stderr}")

        data = json.loads(result.stdout)
        duration = float(data["format"]["duration"])
        return max(0.0, duration)

    def analyse_images(self) -> List[CandidateImage]:
        candidates: List[CandidateImage] = []
        total = len(self.image_paths)

        for idx, path in enumerate(self.image_paths, start=1):
            self.progress(0.02 + 0.18 * (idx / max(1, total)), f"Scoring image {idx}/{total}")
            try:
                img = load_and_normalise_image(path)
                c = clarity_score(img)
                hb = average_hash_bits(img)
                candidates.append(CandidateImage(path=path, clarity_score=c, hash_bits=hb))
            except Exception as exc:
                self.log(f"Skipped unreadable image: {path} ({exc})")

        if not candidates:
            raise RuntimeError("No valid images could be loaded.")

        clarity_values = np.array([c.clarity_score for c in candidates], dtype=np.float32)
        cmin = float(clarity_values.min())
        cmax = float(clarity_values.max())
        denom = max(1e-6, cmax - cmin)

        for i, cand in enumerate(candidates):
            norm_clarity = (cand.clarity_score - cmin) / denom

            nearest = 64
            if len(candidates) > 1:
                for j, other in enumerate(candidates):
                    if i == j:
                        continue
                    d = hamming_distance_bits(cand.hash_bits, other.hash_bits)
                    if d < nearest:
                        nearest = d

            novelty = clamp(nearest / 24.0, 0.0, 1.0)
            cand.novelty_score = novelty
            cand.final_score = norm_clarity * 0.65 + novelty * 0.35

        return candidates

    def choose_images_to_fit(self, candidates: List[CandidateImage]) -> Tuple[List[str], float]:
        s = self.settings
        total = s.target_duration
        tr = s.transition_seconds
        min_hold = s.min_hold_seconds
        max_hold = s.max_hold_seconds

        if total <= 0:
            raise ValueError("Target duration must be greater than zero.")

        max_images = int(math.floor((total + tr) / (min_hold + tr)))
        max_images = max(1, max_images)

        usable_count = min(len(candidates), max_images)
        ranked = sorted(enumerate(candidates), key=lambda x: x[1].final_score, reverse=True)
        keep_indices = sorted([idx for idx, _ in ranked[:usable_count]])

        chosen = [candidates[i].path for i in keep_indices]
        n = len(chosen)

        hold = (total - max(0, n - 1) * tr) / max(1, n)
        if hold <= 0:
            raise RuntimeError("Target duration is too short for the chosen transition settings.")

        if hold > max_hold and len(chosen) < len(candidates):
            while len(chosen) < len(candidates):
                remaining = [i for i in range(len(candidates)) if i not in keep_indices]
                if not remaining:
                    break

                best_extra = max(remaining, key=lambda i: candidates[i].final_score)
                keep_indices = sorted(keep_indices + [best_extra])

                trial_n = len(keep_indices)
                trial_hold = (total - max(0, trial_n - 1) * tr) / trial_n

                if trial_hold < min_hold:
                    keep_indices.remove(best_extra)
                    break

                chosen = [candidates[i].path for i in keep_indices]
                hold = trial_hold

                if hold <= max_hold:
                    break

        self.log(f"Selected {len(chosen)} image(s) out of {len(candidates)}.")
        self.log(f"Computed hold time per image: {hold:.2f}s")
        return chosen, hold

    def render_video_only(self, selected_paths: List[str], hold_seconds: float, temp_video_path: str) -> float:
        s = self.settings
        width, height = s.resolution
        fps = s.fps
        hold_frames = max(1, int(round(hold_seconds * fps)))
        transition_frames = max(1, int(round(s.transition_seconds * fps)))

        images = [load_and_normalise_image(p) for p in selected_paths]
        count = len(images)
        motion_plans = build_motion_plans_for_selected(selected_paths, s.motion_intensity)

        total_frames = count * hold_frames + max(0, count - 1) * transition_frames

        start_frames = []
        total_active = []
        for j in range(count):
            start = max(0, j * (hold_frames + transition_frames) - transition_frames) if j > 0 else 0
            end = (j + 1) * (hold_frames + transition_frames) if j < count - 1 else j * (hold_frames + transition_frames) + hold_frames
            start_frames.append(start)
            total_active.append(end - start)

        writer = imageio_ffmpeg.write_frames(
            temp_video_path,
            (width, height),
            fps=fps,
            codec="libx264",
            pix_fmt_in="bgr24",
            output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "18"],
        )
        next(writer)

        frame_index = 0

        try:
            for i in range(count):
                img_a = images[i]
                plan_a = motion_plans[i]
                bg_a = make_background_canvas(img_a, s.resolution, s.background_mode)

                for f in range(hold_frames):
                    t_a = (frame_index - start_frames[i]) / max(1, total_active[i] - 1)
                    zoom, pan_x, pan_y = plan_a.state_at(t_a)
                    frame = fit_image_with_padding(
                        img_a,
                        s.resolution,
                        bg_a,
                        zoom_factor=zoom,
                        pan_x=pan_x,
                        pan_y=pan_y,
                    )
                    writer.send(pil_to_bgr_bytes(frame))
                    frame_index += 1
                    self.progress(
                        0.25 + 0.55 * (frame_index / max(1, total_frames)),
                        f"Rendering frame {frame_index}/{total_frames}",
                    )

                if i < count - 1:
                    img_b = images[i + 1]
                    plan_b = motion_plans[i + 1]
                    bg_b = make_background_canvas(img_b, s.resolution, s.background_mode)

                    for f in range(transition_frames):
                        t_trans = f / max(1, transition_frames - 1)

                        t_a = (frame_index - start_frames[i]) / max(1, total_active[i] - 1)
                        zoom_a, pan_ax, pan_ay = plan_a.state_at(t_a)
                        frame_a = fit_image_with_padding(
                            img_a,
                            s.resolution,
                            bg_a,
                            zoom_factor=zoom_a,
                            pan_x=pan_ax,
                            pan_y=pan_ay,
                        )

                        t_b = (frame_index - start_frames[i + 1]) / max(1, total_active[i + 1] - 1)
                        zoom_b, pan_bx, pan_by = plan_b.state_at(t_b)
                        frame_b = fit_image_with_padding(
                            img_b,
                            s.resolution,
                            bg_b,
                            zoom_factor=zoom_b,
                            pan_x=pan_bx,
                            pan_y=pan_by,
                        )

                        frame = make_transition_frame(frame_a, frame_b, t_trans, s.transition_mode)
                        writer.send(pil_to_bgr_bytes(frame))
                        frame_index += 1
                        self.progress(
                            0.25 + 0.55 * (frame_index / max(1, total_frames)),
                            f"Rendering frame {frame_index}/{total_frames}",
                        )
        finally:
            writer.close()

        return total_frames / fps

    def build_audio_track(self, target_audio_path: str, actual_duration: float) -> bool:
        s = self.settings
        if s.audio_mode == "none" or not s.audio_paths:
            return False

        target = actual_duration
        fade_dur = min(3.0, target)
        fade_start = max(0.0, target - fade_dur)
        af_filter = f"afade=t=out:st={fade_start:.3f}:d={fade_dur:.3f}"

        if s.audio_mode == "single_loop":
            src = s.audio_paths[0]
            cmd = [
                self.ffmpeg,
                "-y",
                "-stream_loop",
                "-1",
                "-i",
                src,
                "-map",
                "0:a:0",
                "-vn",
                "-t",
                str(target),
                "-af",
                af_filter,
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                target_audio_path,
            ]
            self.run_ffmpeg(cmd, "Building looping audio")
            return True

        if s.audio_mode == "playlist_once":
            playlist_sources = s.audio_paths.copy()
        else:
            total_playlist_duration = 0.0
            for p in s.audio_paths:
                total_playlist_duration += self.probe_audio_duration(p)

            if total_playlist_duration <= 0:
                raise RuntimeError("Could not determine playlist duration.")

            repeats = max(1, math.ceil(target / total_playlist_duration))
            playlist_sources = s.audio_paths * repeats

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
            list_path = tf.name
            for p in playlist_sources:
                safe_path = Path(p).resolve().as_posix().replace("'", r"'\''")
                tf.write(f"file '{safe_path}'\n")

        try:
            cmd = [
                self.ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-map",
                "0:a:0",
                "-vn",
                "-t",
                str(target),
                "-af",
                af_filter,
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                target_audio_path,
            ]
            self.run_ffmpeg(cmd, "Building playlist audio")
        finally:
            try:
                os.remove(list_path)
            except Exception:
                pass

        return True

    def mux_video_audio(self, video_path: str, audio_path: Optional[str], out_path: str) -> None:
        if not audio_path:
            if os.path.abspath(video_path) != os.path.abspath(out_path):
                os.replace(video_path, out_path)
            return

        cmd = [
            self.ffmpeg,
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            out_path,
        ]
        self.run_ffmpeg(cmd, "Muxing video and audio")

    def export(self) -> None:
        out_path = self.settings.output_path
        os.makedirs(str(Path(out_path).resolve().parent), exist_ok=True)

        temp_video = str(Path(out_path).with_suffix(".video_only_temp.mp4"))
        temp_audio = str(Path(out_path).with_suffix(".audio_temp.m4a"))

        self.progress(0.01, "Analysing images")
        candidates = self.analyse_images()
        selected_paths, hold_seconds = self.choose_images_to_fit(candidates)

        self.progress(0.22, "Rendering video")
        actual_duration = self.render_video_only(selected_paths, hold_seconds, temp_video)

        built_audio = False
        if self.settings.audio_mode != "none" and self.settings.audio_paths:
            self.progress(0.83, "Preparing audio")
            built_audio = self.build_audio_track(temp_audio, actual_duration)

        self.progress(0.93, "Finalising")
        self.mux_video_audio(temp_video, temp_audio if built_audio else None, out_path)

        for tmp in [temp_video, temp_audio]:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

        self.progress(1.0, "Done")
        self.log(f"Finished: {out_path}")


class ExportWorker(QObject):
    progress_changed = Signal(float, str)
    log_message = Signal(str)
    finished = Signal(str)
    failed = Signal(str, str)

    def __init__(self, image_paths: List[str], settings: SlideSettings):
        super().__init__()
        self.image_paths = image_paths
        self.settings = settings

    def run(self) -> None:
        try:
            exporter = SlideshowExporter(
                image_paths=self.image_paths.copy(),
                settings=self.settings,
                progress_cb=lambda value, status: self.progress_changed.emit(float(value), str(status)),
                log_cb=lambda msg: self.log_message.emit(str(msg)),
            )
            exporter.export()
            self.finished.emit(self.settings.output_path)
        except Exception as exc:
            err = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.failed.emit(str(exc), err)


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    qimage = ImageQt(img.convert("RGBA"))
    return QPixmap.fromImage(qimage)


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Slideshow Maker")
        self.resize(1280, 840)
        self.setMinimumSize(1100, 720)

        self.image_paths: List[str] = []
        self.audio_paths: List[str] = []
        self.preview_pixmap: Optional[QPixmap] = None
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[ExportWorker] = None

        self._build_ui()
        self.update_summary()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        right = QWidget()
        right.setFixedWidth(390)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        root.addWidget(left, 1)
        root.addWidget(right, 0)

        img_actions = QGroupBox("Images")
        img_layout = QVBoxLayout(img_actions)

        top_row = QHBoxLayout()
        self.btn_add_images = QPushButton("Add Images")
        self.btn_add_images.clicked.connect(self.add_images)
        top_row.addWidget(self.btn_add_images)
        self.btn_add_folder = QPushButton("Add Folder")
        self.btn_add_folder.clicked.connect(self.add_folder)
        top_row.addWidget(self.btn_add_folder)
        self.btn_remove_images = QPushButton("Remove Selected")
        self.btn_remove_images.clicked.connect(self.remove_selected_images)
        top_row.addWidget(self.btn_remove_images)
        self.btn_clear_images = QPushButton("Clear")
        self.btn_clear_images.clicked.connect(self.clear_images)
        top_row.addWidget(self.btn_clear_images)
        top_row.addStretch(1)
        img_layout.addLayout(top_row)

        self.recursive_images_check = QCheckBox("Include subfolders when adding an image folder")
        self.recursive_images_check.setChecked(False)
        self.recursive_images_check.toggled.connect(self.update_summary)
        img_layout.addWidget(self.recursive_images_check)

        splitter = QSplitter(Qt.Horizontal)
        list_group = QGroupBox("Image Order")
        list_layout = QVBoxLayout(list_group)
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.image_list.currentRowChanged.connect(self.on_select_preview)
        list_layout.addWidget(self.image_list)

        order_row = QHBoxLayout()
        self.btn_move_up = QPushButton("Move Up")
        self.btn_move_up.clicked.connect(self.move_up)
        order_row.addWidget(self.btn_move_up)
        self.btn_move_down = QPushButton("Move Down")
        self.btn_move_down.clicked.connect(self.move_down)
        order_row.addWidget(self.btn_move_down)
        order_row.addStretch(1)
        list_layout.addLayout(order_row)

        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("No preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(360, 360)
        self.preview_label.setFrameShape(QFrame.StyledPanel)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_layout.addWidget(self.preview_label)

        splitter.addWidget(list_group)
        splitter.addWidget(preview_group)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        img_layout.addWidget(splitter, 1)

        left_layout.addWidget(img_actions, 1)

        settings = QGroupBox("Export Settings")
        settings_layout = QGridLayout(settings)
        row = 0

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(list(VIDEO_RESOLUTIONS.keys()))
        self.resolution_combo.setCurrentText("Full HD 1920x1080")
        self.resolution_combo.currentTextChanged.connect(self.update_summary)
        self._add_grid_row(settings_layout, row, "Resolution", self.resolution_combo)
        row += 1

        self.fps_combo = QComboBox()
        for fps in FPS_OPTIONS:
            self.fps_combo.addItem(str(fps), fps)
        self.fps_combo.setCurrentText("30")
        self.fps_combo.currentTextChanged.connect(self.update_summary)
        self._add_grid_row(settings_layout, row, "FPS", self.fps_combo)
        row += 1

        self.target_duration_spin = self._make_double_spin(5.0, 7200.0, 90.0, 5.0, 1)
        self._add_grid_row(settings_layout, row, "Target duration (sec)", self.target_duration_spin)
        row += 1

        self.min_hold_spin = self._make_double_spin(0.5, 30.0, 2.0, 0.5, 1)
        self._add_grid_row(settings_layout, row, "Min hold per image", self.min_hold_spin)
        row += 1

        self.max_hold_spin = self._make_double_spin(0.5, 60.0, 5.0, 0.5, 1)
        self._add_grid_row(settings_layout, row, "Max hold per image", self.max_hold_spin)
        row += 1

        self.transition_spin = self._make_double_spin(0.2, 5.0, 0.8, 0.1, 1)
        self._add_grid_row(settings_layout, row, "Transition length", self.transition_spin)
        row += 1

        self.transition_mode_combo = QComboBox()
        self.transition_mode_combo.addItems(TRANSITIONS)
        self.transition_mode_combo.setCurrentText("Crossfade")
        self.transition_mode_combo.currentTextChanged.connect(self.update_summary)
        self._add_grid_row(settings_layout, row, "Transition style", self.transition_mode_combo)
        row += 1

        self.background_mode_combo = QComboBox()
        self.background_mode_combo.addItems(BACKGROUND_MODES)
        self.background_mode_combo.setCurrentText("Blurred")
        self.background_mode_combo.currentTextChanged.connect(self.update_summary)
        self._add_grid_row(settings_layout, row, "Background / padding", self.background_mode_combo)
        row += 1

        self.motion_intensity_spin = self._make_double_spin(0.0, 5.0, 1.5, 0.1, 1)
        self._add_grid_row(settings_layout, row, "Motion intensity", self.motion_intensity_spin)
        row += 1

        output_wrap = QWidget()
        output_layout = QHBoxLayout(output_wrap)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(6)
        self.output_edit = QLineEdit(str(Path.cwd() / "slideshow_output.mp4"))
        self.output_edit.textChanged.connect(self.update_summary)
        output_layout.addWidget(self.output_edit, 1)
        self.btn_browse_output = QPushButton("Browse")
        self.btn_browse_output.clicked.connect(self.browse_output)
        output_layout.addWidget(self.btn_browse_output)
        self._add_grid_row(settings_layout, row, "Output file", output_wrap)
        row += 1
        settings_layout.setColumnStretch(1, 1)
        right_layout.addWidget(settings)

        audio_box = QGroupBox("Audio")
        audio_layout = QVBoxLayout(audio_box)
        audio_actions = QHBoxLayout()
        self.btn_add_tracks = QPushButton("Add Track(s)")
        self.btn_add_tracks.clicked.connect(self.add_audio_tracks)
        audio_actions.addWidget(self.btn_add_tracks)
        self.btn_add_audio_folder = QPushButton("Add Audio Folder")
        self.btn_add_audio_folder.clicked.connect(self.add_audio_folder)
        audio_actions.addWidget(self.btn_add_audio_folder)
        self.btn_remove_audio = QPushButton("Remove Selected")
        self.btn_remove_audio.clicked.connect(self.remove_selected_audio)
        audio_actions.addWidget(self.btn_remove_audio)
        self.btn_clear_audio = QPushButton("Clear")
        self.btn_clear_audio.clicked.connect(self.clear_audio)
        audio_actions.addWidget(self.btn_clear_audio)
        audio_actions.addStretch(1)
        audio_layout.addLayout(audio_actions)

        self.recursive_audio_check = QCheckBox("Include subfolders when adding an audio folder")
        self.recursive_audio_check.setChecked(False)
        self.recursive_audio_check.toggled.connect(self.update_summary)
        audio_layout.addWidget(self.recursive_audio_check)

        self.audio_list = QListWidget()
        self.audio_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.audio_list.setMaximumHeight(120)
        audio_layout.addWidget(self.audio_list)

        audio_mode_row = QHBoxLayout()
        audio_mode_row.addWidget(QLabel("Audio mode"))
        self.audio_mode_combo = QComboBox()
        self.audio_mode_combo.addItems(["none", "single_loop", "playlist_once", "playlist_loop"])
        self.audio_mode_combo.setCurrentText("single_loop")
        self.audio_mode_combo.currentTextChanged.connect(self.update_summary)
        audio_mode_row.addWidget(self.audio_mode_combo, 1)
        audio_layout.addLayout(audio_mode_row)
        right_layout.addWidget(audio_box)

        summary = QGroupBox("Summary")
        summary_layout = QVBoxLayout(summary)
        self.summary_label = QLabel()
        self.summary_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)
        right_layout.addWidget(summary)

        progress_frame = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_frame)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        progress_layout.addWidget(self.progress)

        self.status_label = QLabel("Idle")
        progress_layout.addWidget(self.status_label)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        progress_layout.addWidget(self.log_text, 1)
        right_layout.addWidget(progress_frame, 1)

        self.export_button = QPushButton("Export Slideshow")
        self.export_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.export_button.clicked.connect(self.start_export)
        right_layout.addWidget(self.export_button)

    def _add_grid_row(self, grid: QGridLayout, row: int, label: str, widget: QWidget) -> None:
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(widget, row, 1)

    def _make_double_spin(self, minimum: float, maximum: float, value: float, step: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.valueChanged.connect(self.update_summary)
        return spin

    def _selected_rows(self, widget: QListWidget) -> List[int]:
        return sorted(index.row() for index in widget.selectedIndexes())

    def _set_preview_pixmap(self) -> None:
        if self.preview_pixmap is None:
            return
        scaled = self.preview_pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._set_preview_pixmap()

    def log(self, msg: str) -> None:
        self.log_text.appendPlainText(msg)

    def set_progress(self, value: float, status: str) -> None:
        self.progress.setValue(int(clamp(value, 0.0, 1.0) * 1000))
        self.status_label.setText(status)

    def update_summary(self) -> None:
        n = len(self.image_paths)
        total = float(self.target_duration_spin.value())
        tr = float(self.transition_spin.value())
        min_hold = float(self.min_hold_spin.value())

        if n > 0:
            max_fit = max(1, int(math.floor((total + tr) / (min_hold + tr))))
            will_drop = max(0, n - max_fit)
        else:
            max_fit = 0
            will_drop = 0

        fps = int(self.fps_combo.currentData() or int(self.fps_combo.currentText()))
        audio_text = f"Tracks: {len(self.audio_paths)} | Mode: {self.audio_mode_combo.currentText()}"
        self.summary_label.setText(
            f"Images loaded: {n}\n"
            f"Target length: {seconds_to_hms(total)}\n"
            f"Max images that fit at current minimum hold: {max_fit}\n"
            f"Estimated dropped by scorer: {will_drop}\n"
            f"Resolution: {self.resolution_combo.currentText()} @ {fps}fps\n"
            f"{audio_text}\n"
            f"Image folder recursion: {'On' if self.recursive_images_check.isChecked() else 'Off'}"
        )

    def add_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select images",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)",
        )
        for f in files:
            if f not in self.image_paths:
                self.image_paths.append(f)
        self.refresh_image_list()

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not folder:
            return
        recursive = self.recursive_images_check.isChecked()
        for f in list_images_in_folder(folder, recursive=recursive):
            if f not in self.image_paths:
                self.image_paths.append(f)
        self.refresh_image_list()

    def remove_selected_images(self) -> None:
        indices = self._selected_rows(self.image_list)
        for idx in reversed(indices):
            del self.image_paths[idx]
        self.refresh_image_list()

    def clear_images(self) -> None:
        self.image_paths.clear()
        self.refresh_image_list()
        self.preview_label.setText("No preview")
        self.preview_label.setPixmap(QPixmap())
        self.preview_pixmap = None

    def move_up(self) -> None:
        indices = self._selected_rows(self.image_list)
        if not indices or indices[0] == 0:
            return
        for idx in indices:
            self.image_paths[idx - 1], self.image_paths[idx] = self.image_paths[idx], self.image_paths[idx - 1]
        self.refresh_image_list([i - 1 for i in indices])

    def move_down(self) -> None:
        indices = self._selected_rows(self.image_list)
        if not indices or indices[-1] >= len(self.image_paths) - 1:
            return
        for idx in reversed(indices):
            self.image_paths[idx + 1], self.image_paths[idx] = self.image_paths[idx], self.image_paths[idx + 1]
        self.refresh_image_list([i + 1 for i in indices])

    def refresh_image_list(self, select_indices: Optional[List[int]] = None) -> None:
        self.image_list.clear()
        for p in self.image_paths:
            self.image_list.addItem(QListWidgetItem(Path(p).name))
        if select_indices:
            with QSignalBlocker(self.image_list.selectionModel()):
                for i in select_indices:
                    item = self.image_list.item(i)
                    if item is not None:
                        item.setSelected(True)
                if select_indices:
                    self.image_list.setCurrentRow(select_indices[0])
        self.update_summary()

    def on_select_preview(self, row: int) -> None:
        if row < 0 or row >= len(self.image_paths):
            return
        path = self.image_paths[row]
        try:
            img = load_and_normalise_image(path)
            preview = img.copy()
            preview.thumbnail((700, 700), Image.Resampling.LANCZOS)
            self.preview_pixmap = pil_to_qpixmap(preview)
            self.preview_label.setText("")
            self._set_preview_pixmap()
        except Exception as exc:
            self.preview_label.setText(f"Preview failed:\n{exc}")
            self.preview_label.setPixmap(QPixmap())
            self.preview_pixmap = None

    def add_audio_tracks(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select audio tracks",
            "",
            "Audio (*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.wma);;All files (*.*)",
        )
        for f in files:
            if f not in self.audio_paths:
                self.audio_paths.append(f)
        self.refresh_audio_list()

    def add_audio_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select audio folder")
        if not folder:
            return
        recursive = self.recursive_audio_check.isChecked()
        for f in list_audio_in_folder(folder, recursive=recursive):
            if f not in self.audio_paths:
                self.audio_paths.append(f)
        self.refresh_audio_list()

    def remove_selected_audio(self) -> None:
        indices = self._selected_rows(self.audio_list)
        for idx in reversed(indices):
            del self.audio_paths[idx]
        self.refresh_audio_list()

    def clear_audio(self) -> None:
        self.audio_paths.clear()
        self.refresh_audio_list()

    def refresh_audio_list(self) -> None:
        self.audio_list.clear()
        for p in self.audio_paths:
            self.audio_list.addItem(QListWidgetItem(Path(p).name))
        self.update_summary()

    def browse_output(self) -> None:
        file, _ = QFileDialog.getSaveFileName(
            self,
            "Save slideshow as",
            str(Path(self.output_edit.text() or "slideshow_output.mp4")),
            "MP4 Video (*.mp4)",
        )
        if file:
            self.output_edit.setText(file)

    def collect_settings(self) -> SlideSettings:
        target = float(self.target_duration_spin.value())
        min_hold = float(self.min_hold_spin.value())
        max_hold = float(self.max_hold_spin.value())
        transition = float(self.transition_spin.value())
        motion = float(self.motion_intensity_spin.value())

        if target <= 0:
            raise ValueError("Target duration must be greater than zero.")
        if min_hold <= 0 or max_hold <= 0:
            raise ValueError("Hold values must be greater than zero.")
        if min_hold > max_hold:
            raise ValueError("Minimum hold cannot be greater than maximum hold.")
        if transition < 0:
            raise ValueError("Transition cannot be negative.")
        if motion < 0:
            raise ValueError("Motion intensity cannot be negative.")

        mode = self.audio_mode_combo.currentText()
        if not self.audio_paths:
            mode = "none"

        output_path = self.output_edit.text().strip()
        if not output_path:
            raise ValueError("Please choose an output file.")

        fps = int(self.fps_combo.currentData() or int(self.fps_combo.currentText()))
        return SlideSettings(
            fps=fps,
            resolution=VIDEO_RESOLUTIONS[self.resolution_combo.currentText()],
            transition_mode=self.transition_mode_combo.currentText(),
            background_mode=self.background_mode_combo.currentText(),
            output_path=output_path,
            target_duration=target,
            min_hold_seconds=min_hold,
            max_hold_seconds=max_hold,
            transition_seconds=transition,
            motion_intensity=motion,
            audio_paths=self.audio_paths.copy(),
            audio_mode=mode,
        )

    def _set_busy(self, busy: bool) -> None:
        widgets = [
            self.btn_add_images, self.btn_add_folder, self.btn_remove_images, self.btn_clear_images,
            self.btn_move_up, self.btn_move_down, self.btn_add_tracks, self.btn_add_audio_folder,
            self.btn_remove_audio, self.btn_clear_audio, self.btn_browse_output, self.export_button,
            self.recursive_images_check, self.recursive_audio_check, self.image_list, self.audio_list,
            self.resolution_combo, self.fps_combo, self.target_duration_spin, self.min_hold_spin,
            self.max_hold_spin, self.transition_spin, self.transition_mode_combo,
            self.background_mode_combo, self.motion_intensity_spin, self.output_edit, self.audio_mode_combo,
        ]
        for widget in widgets:
            widget.setEnabled(not busy)
        self.export_button.setText("Exporting..." if busy else "Export Slideshow")

    def start_export(self) -> None:
        if self.worker_thread is not None and self.worker_thread.isRunning():
            QMessageBox.information(self, "Busy", "An export is already running.")
            return
        if not self.image_paths:
            QMessageBox.warning(self, "No images", "Please add at least one image.")
            return

        try:
            settings = self.collect_settings()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return

        self.progress.setValue(0)
        self.status_label.setText("Starting...")
        self.log_text.clear()
        self._set_busy(True)

        self.worker_thread = QThread(self)
        self.worker = ExportWorker(self.image_paths.copy(), settings)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress_changed.connect(self.set_progress)
        self.worker.log_message.connect(self.log)
        self.worker.finished.connect(self._on_export_finished)
        self.worker.failed.connect(self._on_export_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._cleanup_worker)
        self.worker_thread.start()

    def _on_export_finished(self, output_path: str) -> None:
        self._set_busy(False)
        QMessageBox.information(self, "Done", f"Saved to:\n{output_path}")

    def _on_export_failed(self, message: str, detailed: str) -> None:
        self.log(detailed)
        self.set_progress(0.0, "Failed")
        self._set_busy(False)
        QMessageBox.critical(self, "Export failed", message)

    def _cleanup_worker(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
            self.worker_thread = None


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = App()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
