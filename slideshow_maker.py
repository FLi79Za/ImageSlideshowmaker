from __future__ import annotations

import json
import math
import os
import random
import subprocess
import tempfile
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Slideshow Maker")
        self.geometry("1220x800")
        self.minsize(1080, 720)

        self.image_paths: List[str] = []
        self.audio_paths: List[str] = []
        self.preview_image_tk = None
        self.export_thread: Optional[threading.Thread] = None

        self.recursive_images_var = tk.BooleanVar(value=False)
        self.recursive_audio_var = tk.BooleanVar(value=False)

        self._build_ui()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        left = ttk.Frame(root)
        left.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(root)
        right.pack(side="right", fill="y")

        img_actions = ttk.LabelFrame(left, text="Images", padding=10)
        img_actions.pack(fill="x", pady=(0, 10))

        top_row = ttk.Frame(img_actions)
        top_row.pack(fill="x")

        ttk.Button(top_row, text="Add Images", command=self.add_images).pack(side="left", padx=(0, 8))
        ttk.Button(top_row, text="Add Folder", command=self.add_folder).pack(side="left", padx=(0, 8))
        ttk.Button(top_row, text="Remove Selected", command=self.remove_selected_images).pack(side="left", padx=(0, 8))
        ttk.Button(top_row, text="Clear", command=self.clear_images).pack(side="left")

        ttk.Checkbutton(
            img_actions,
            text="Include subfolders when adding an image folder",
            variable=self.recursive_images_var,
        ).pack(anchor="w", pady=(8, 0))

        panes = ttk.Panedwindow(left, orient="horizontal")
        panes.pack(fill="both", expand=True)

        list_frame = ttk.LabelFrame(panes, text="Image Order", padding=8)
        preview_frame = ttk.LabelFrame(panes, text="Preview", padding=8)
        panes.add(list_frame, weight=3)
        panes.add(preview_frame, weight=2)

        self.image_listbox = tk.Listbox(list_frame, selectmode="extended")
        self.image_listbox.pack(side="left", fill="both", expand=True)
        self.image_listbox.bind("<<ListboxSelect>>", self.on_select_preview)

        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.image_listbox.yview)
        scroll.pack(side="right", fill="y")
        self.image_listbox.configure(yscrollcommand=scroll.set)

        order_btns = ttk.Frame(list_frame)
        order_btns.pack(fill="x", pady=(8, 0))
        ttk.Button(order_btns, text="Move Up", command=self.move_up).pack(side="left", padx=(0, 6))
        ttk.Button(order_btns, text="Move Down", command=self.move_down).pack(side="left")

        self.preview_label = ttk.Label(preview_frame, text="No preview", anchor="center")
        self.preview_label.pack(fill="both", expand=True)

        settings = ttk.LabelFrame(right, text="Export Settings", padding=10)
        settings.pack(fill="x")

        self.resolution_var = tk.StringVar(value="Full HD 1920x1080")
        self.fps_var = tk.IntVar(value=30)
        self.transition_var = tk.DoubleVar(value=0.8)
        self.transition_mode_var = tk.StringVar(value="Crossfade")
        self.background_mode_var = tk.StringVar(value="Blurred")
        self.motion_intensity_var = tk.DoubleVar(value=1.5)
        self.output_var = tk.StringVar(value=str(Path.cwd() / "slideshow_output.mp4"))

        self.target_duration_var = tk.DoubleVar(value=90.0)
        self.min_hold_var = tk.DoubleVar(value=2.0)
        self.max_hold_var = tk.DoubleVar(value=5.0)

        self.audio_mode_var = tk.StringVar(value="single_loop")

        row = 0
        ttk.Label(settings, text="Resolution").grid(row=row, column=0, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.resolution_var,
            values=list(VIDEO_RESOLUTIONS.keys()),
            state="readonly",
            width=24,
        ).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(settings, text="FPS").grid(row=row, column=0, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.fps_var,
            values=FPS_OPTIONS,
            state="readonly",
            width=24,
        ).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(settings, text="Target duration (sec)").grid(row=row, column=0, sticky="w")
        ttk.Spinbox(
            settings,
            from_=5.0,
            to=7200.0,
            increment=5.0,
            textvariable=self.target_duration_var,
            width=12,
        ).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(settings, text="Min hold per image").grid(row=row, column=0, sticky="w")
        ttk.Spinbox(
            settings,
            from_=0.5,
            to=30.0,
            increment=0.5,
            textvariable=self.min_hold_var,
            width=12,
        ).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(settings, text="Max hold per image").grid(row=row, column=0, sticky="w")
        ttk.Spinbox(
            settings,
            from_=0.5,
            to=60.0,
            increment=0.5,
            textvariable=self.max_hold_var,
            width=12,
        ).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(settings, text="Transition length").grid(row=row, column=0, sticky="w")
        ttk.Spinbox(
            settings,
            from_=0.2,
            to=5.0,
            increment=0.1,
            textvariable=self.transition_var,
            width=12,
        ).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(settings, text="Transition style").grid(row=row, column=0, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.transition_mode_var,
            values=TRANSITIONS,
            state="readonly",
            width=24,
        ).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(settings, text="Background / padding").grid(row=row, column=0, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.background_mode_var,
            values=BACKGROUND_MODES,
            state="readonly",
            width=24,
        ).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(settings, text="Motion Intensity").grid(row=row, column=0, sticky="w")
        ttk.Spinbox(
            settings,
            from_=0.0,
            to=5.0,
            increment=0.1,
            textvariable=self.motion_intensity_var,
            width=12,
        ).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(settings, text="Output file").grid(row=row, column=0, sticky="w")
        out_frame = ttk.Frame(settings)
        out_frame.grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Entry(out_frame, textvariable=self.output_var, width=28).pack(side="left", fill="x", expand=True)
        ttk.Button(out_frame, text="Browse", command=self.browse_output).pack(side="left", padx=(6, 0))
        row += 1

        settings.columnconfigure(1, weight=1)

        audio_box = ttk.LabelFrame(right, text="Audio", padding=10)
        audio_box.pack(fill="both", pady=(10, 0))

        audio_actions = ttk.Frame(audio_box)
        audio_actions.pack(fill="x")
        ttk.Button(audio_actions, text="Add Track(s)", command=self.add_audio_tracks).pack(side="left", padx=(0, 8))
        ttk.Button(audio_actions, text="Add Audio Folder", command=self.add_audio_folder).pack(side="left", padx=(0, 8))
        ttk.Button(audio_actions, text="Remove Selected", command=self.remove_selected_audio).pack(side="left", padx=(0, 8))
        ttk.Button(audio_actions, text="Clear", command=self.clear_audio).pack(side="left")

        ttk.Checkbutton(
            audio_box,
            text="Include subfolders when adding an audio folder",
            variable=self.recursive_audio_var,
        ).pack(anchor="w", pady=(8, 0))

        self.audio_listbox = tk.Listbox(audio_box, height=6, selectmode="extended")
        self.audio_listbox.pack(fill="x", pady=(8, 8))

        ttk.Label(audio_box, text="Audio mode").pack(anchor="w")
        ttk.Combobox(
            audio_box,
            textvariable=self.audio_mode_var,
            state="readonly",
            values=["none", "single_loop", "playlist_once", "playlist_loop"],
        ).pack(fill="x", pady=(4, 0))

        summary = ttk.LabelFrame(right, text="Summary", padding=10)
        summary.pack(fill="x", pady=(10, 0))
        self.summary_label = ttk.Label(summary, justify="left")
        self.summary_label.pack(fill="x")

        progress_frame = ttk.LabelFrame(right, text="Progress", padding=10)
        progress_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(progress_frame, maximum=1.0, variable=self.progress_var)
        self.progress.pack(fill="x")

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(progress_frame, textvariable=self.status_var).pack(anchor="w", pady=(6, 6))

        self.log_text = tk.Text(progress_frame, height=14, wrap="word")
        self.log_text.pack(fill="both", expand=True)

        ttk.Button(right, text="Export Slideshow", command=self.start_export).pack(fill="x", pady=(10, 0))

        for var in [
            self.resolution_var,
            self.fps_var,
            self.transition_var,
            self.transition_mode_var,
            self.background_mode_var,
            self.target_duration_var,
            self.min_hold_var,
            self.max_hold_var,
            self.motion_intensity_var,
            self.audio_mode_var,
        ]:
            var.trace_add("write", lambda *_: self.update_summary())

        self.update_summary()

    def log(self, msg: str) -> None:
        self.after(0, self._append_log, msg)

    def _append_log(self, msg: str) -> None:
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")

    def set_progress(self, value: float, status: str) -> None:
        self.after(0, lambda: (self.progress_var.set(clamp(value, 0.0, 1.0)), self.status_var.set(status)))

    def update_summary(self) -> None:
        n = len(self.image_paths)
        total = self.target_duration_var.get()
        tr = self.transition_var.get()
        min_hold = self.min_hold_var.get()

        if n > 0:
            max_fit = max(1, int(math.floor((total + tr) / (min_hold + tr))))
            will_drop = max(0, n - max_fit)
        else:
            max_fit = 0
            will_drop = 0

        audio_text = f"Tracks: {len(self.audio_paths)} | Mode: {self.audio_mode_var.get()}"
        self.summary_label.config(
            text=(
                f"Images loaded: {n}\n"
                f"Target length: {seconds_to_hms(total)}\n"
                f"Max images that fit at current minimum hold: {max_fit}\n"
                f"Estimated dropped by scorer: {will_drop}\n"
                f"Resolution: {self.resolution_var.get()} @ {self.fps_var.get()}fps\n"
                f"{audio_text}\n"
                f"Image folder recursion: {'On' if self.recursive_images_var.get() else 'Off'}"
            )
        )

    def add_images(self) -> None:
        files = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff")],
        )
        for f in files:
            if f not in self.image_paths:
                self.image_paths.append(f)
        self.refresh_image_list()

    def add_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select image folder")
        if not folder:
            return

        recursive = self.recursive_images_var.get()
        for f in list_images_in_folder(folder, recursive=recursive):
            if f not in self.image_paths:
                self.image_paths.append(f)

        self.refresh_image_list()

    def remove_selected_images(self) -> None:
        indices = list(self.image_listbox.curselection())
        for idx in reversed(indices):
            del self.image_paths[idx]
        self.refresh_image_list()

    def clear_images(self) -> None:
        self.image_paths.clear()
        self.refresh_image_list()
        self.preview_label.configure(image="", text="No preview")
        self.preview_image_tk = None

    def move_up(self) -> None:
        indices = list(self.image_listbox.curselection())
        if not indices or indices[0] == 0:
            return
        for idx in indices:
            self.image_paths[idx - 1], self.image_paths[idx] = self.image_paths[idx], self.image_paths[idx - 1]
        self.refresh_image_list([i - 1 for i in indices])

    def move_down(self) -> None:
        indices = list(self.image_listbox.curselection())
        if not indices or indices[-1] >= len(self.image_paths) - 1:
            return
        for idx in reversed(indices):
            self.image_paths[idx + 1], self.image_paths[idx] = self.image_paths[idx], self.image_paths[idx + 1]
        self.refresh_image_list([i + 1 for i in indices])

    def refresh_image_list(self, select_indices: Optional[List[int]] = None) -> None:
        self.image_listbox.delete(0, "end")
        for p in self.image_paths:
            self.image_listbox.insert("end", Path(p).name)
        if select_indices:
            for i in select_indices:
                self.image_listbox.selection_set(i)
        self.update_summary()

    def on_select_preview(self, _event=None) -> None:
        selection = self.image_listbox.curselection()
        if not selection:
            return
        path = self.image_paths[selection[0]]
        try:
            img = load_and_normalise_image(path)
            preview = img.copy()
            preview.thumbnail((360, 360), Image.Resampling.LANCZOS)
            self.preview_image_tk = ImageTk.PhotoImage(preview)
            self.preview_label.configure(image=self.preview_image_tk, text="")
        except Exception as exc:
            self.preview_label.configure(text=f"Preview failed:\n{exc}", image="")
            self.preview_image_tk = None

    def add_audio_tracks(self) -> None:
        files = filedialog.askopenfilenames(
            title="Select audio tracks",
            filetypes=[("Audio", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.wma"), ("All files", "*.*")],
        )
        for f in files:
            if f not in self.audio_paths:
                self.audio_paths.append(f)
        self.refresh_audio_list()

    def add_audio_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select audio folder")
        if not folder:
            return

        recursive = self.recursive_audio_var.get()
        for f in list_audio_in_folder(folder, recursive=recursive):
            if f not in self.audio_paths:
                self.audio_paths.append(f)

        self.refresh_audio_list()

    def remove_selected_audio(self) -> None:
        indices = list(self.audio_listbox.curselection())
        for idx in reversed(indices):
            del self.audio_paths[idx]
        self.refresh_audio_list()

    def clear_audio(self) -> None:
        self.audio_paths.clear()
        self.refresh_audio_list()

    def refresh_audio_list(self) -> None:
        self.audio_listbox.delete(0, "end")
        for p in self.audio_paths:
            self.audio_listbox.insert("end", Path(p).name)
        self.update_summary()

    def browse_output(self) -> None:
        file = filedialog.asksaveasfilename(
            title="Save slideshow as",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")],
            initialfile="slideshow_output.mp4",
        )
        if file:
            self.output_var.set(file)

    def collect_settings(self) -> SlideSettings:
        target = float(self.target_duration_var.get())
        min_hold = float(self.min_hold_var.get())
        max_hold = float(self.max_hold_var.get())
        transition = float(self.transition_var.get())
        motion = float(self.motion_intensity_var.get())

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

        mode = self.audio_mode_var.get()
        if not self.audio_paths:
            mode = "none"  # Gracefully fallback to 'none' if user didn't add tracks
        elif mode == "single_loop" and len(self.audio_paths) < 1:
            raise ValueError("Single loop mode needs at least one track.")

        output_path = self.output_var.get().strip()
        if not output_path:
            raise ValueError("Please choose an output file.")

        return SlideSettings(
            fps=int(self.fps_var.get()),
            resolution=VIDEO_RESOLUTIONS[self.resolution_var.get()],
            transition_mode=self.transition_mode_var.get(),
            background_mode=self.background_mode_var.get(),
            output_path=output_path,
            target_duration=target,
            min_hold_seconds=min_hold,
            max_hold_seconds=max_hold,
            transition_seconds=transition,
            motion_intensity=motion,
            audio_paths=self.audio_paths.copy(),
            audio_mode=mode,
        )

    def start_export(self) -> None:
        if self.export_thread and self.export_thread.is_alive():
            messagebox.showinfo("Busy", "An export is already running.")
            return

        if not self.image_paths:
            messagebox.showwarning("No images", "Please add at least one image.")
            return

        try:
            settings = self.collect_settings()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self.progress_var.set(0.0)
        self.status_var.set("Starting...")
        self.log_text.delete("1.0", "end")

        def worker():
            try:
                exporter = SlideshowExporter(
                    image_paths=self.image_paths.copy(),
                    settings=settings,
                    progress_cb=self.set_progress,
                    log_cb=self.log,
                )
                exporter.export()
                self.after(0, lambda: messagebox.showinfo("Done", f"Saved to:\n{settings.output_path}"))
            except Exception as exc:
                err = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                self.log(err)
                self.set_progress(0.0, "Failed")
                self.after(0, lambda: messagebox.showerror("Export failed", str(exc)))

        self.export_thread = threading.Thread(target=worker, daemon=True)
        self.export_thread.start()


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()