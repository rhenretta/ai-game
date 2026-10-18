"""Central configuration constants for the Autonomous Game Engine."""
import os
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ["STUDIO_DATA_DIR"]) if os.environ.get("STUDIO_DATA_DIR") else (PROJECT_ROOT / "data")
DB_PATH = Path(os.environ["STUDIO_DB_PATH"]) if os.environ.get("STUDIO_DB_PATH") else (DATA_DIR / "telemetry.db")
SPECS_DIR = DATA_DIR / "specs"
ASSETS_DIR = PROJECT_ROOT / "engine" / "assets"

# Ensure data directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
SPECS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Display & Screen Resolution (1280x720 Logical Canvas, High-DPI Scaled)
# ---------------------------------------------------------------------------
CANVAS_WIDTH: int = 1280
CANVAS_HEIGHT: int = 720
CANVAS_SIZE: Tuple[int, int] = (CANVAS_WIDTH, CANVAS_HEIGHT)
ASPECT_RATIO: float = CANVAS_WIDTH / CANVAS_HEIGHT

WINDOW_WIDTH: int = 1280
WINDOW_HEIGHT: int = 720
WINDOW_SIZE: Tuple[int, int] = (WINDOW_WIDTH, WINDOW_HEIGHT)
WINDOW_TITLE: str = "Autonomous Game Engine & Showrunner"

TARGET_FPS: int = 60
FIXED_DT: float = 1.0 / TARGET_FPS  # ~0.0166667s
MAX_FRAME_TIME: float = 0.25         # Clamp spike

# ---------------------------------------------------------------------------
# Audio Settings
# ---------------------------------------------------------------------------
AUDIO_ENABLED: bool = True
SAMPLE_RATE: int = 44100
AUDIO_CHANNELS: int = 2
AUDIO_BUFFER_SIZE: int = 512

# ---------------------------------------------------------------------------
# Telemetry Streaming Constants
# ---------------------------------------------------------------------------
TELEMETRY_ENABLED: bool = True
TELEMETRY_BATCH_SIZE: int = 50
TELEMETRY_FLUSH_INTERVAL: float = 0.1  # 100 ms
TELEMETRY_QUEUE_MAX_SIZE: int = 10000

# ---------------------------------------------------------------------------
# Style Bible 16-Color Palette Quantization (Canonical RGB)
# ---------------------------------------------------------------------------
STYLE_BIBLE_16_PALETTE: List[Tuple[int, int, int]] = [
    (0, 0, 0),        # 0: Pure Black
    (29, 43, 83),     # 1: Dark Navy
    (126, 37, 83),    # 2: Deep Crimson / Violet
    (0, 135, 81),     # 3: Dark Green
    (171, 82, 54),    # 4: Rust Brown
    (95, 87, 79),     # 5: Muted Slate Gray
    (194, 195, 199),  # 6: Light Gray
    (255, 241, 232),  # 7: Off-White / Pearl
    (255, 0, 77),     # 8: Neon Red
    (255, 163, 0),    # 9: Amber Orange
    (255, 236, 39),   # 10: Bright Yellow
    (0, 228, 54),     # 11: Electric Green
    (41, 173, 255),   # 12: Cyan Sky Blue
    (131, 118, 156),  # 13: Lavender Purple
    (255, 119, 168),  # 14: Pastel Rose Pink
    (255, 204, 170),  # 15: Peach Flesh Tint
]

PALETTE: Dict[str, Tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "dark_navy": (29, 43, 83),
    "dark_blue": (29, 43, 83),
    "deep_crimson": (126, 37, 83),
    "dark_green": (0, 135, 81),
    "rust_brown": (171, 82, 54),
    "brown": (171, 82, 54),
    "muted_gray": (95, 87, 79),
    "dark_gray": (95, 87, 79),
    "light_gray": (194, 195, 199),
    "white": (255, 241, 232),
    "red": (255, 0, 77),
    "orange": (255, 163, 0),
    "yellow": (255, 236, 39),
    "green": (0, 228, 54),
    "cyan": (41, 173, 255),
    "blue": (41, 173, 255),
    "purple": (131, 118, 156),
    "pink": (255, 119, 168),
    "peach": (255, 204, 170),
    "beige": (255, 204, 170),
}

COLOR_BG = PALETTE["black"]
COLOR_TEXT = PALETTE["white"]
COLOR_ACCENT = PALETTE["cyan"]
COLOR_HEALTH = PALETTE["green"]
COLOR_DAMAGE = PALETTE["red"]


def quantize_color(rgb: Tuple[int, int, int], palette: List[Tuple[int, int, int]] = STYLE_BIBLE_16_PALETTE) -> Tuple[int, int, int]:
    """Quantizes an RGB color to the closest entry in the Style Bible palette via Euclidean distance."""
    r1, g1, b1 = rgb
    best_color = palette[0]
    min_dist = float("inf")
    for pr, pg, pb in palette:
        dist_sq = (r1 - pr) ** 2 + (g1 - pg) ** 2 + (b1 - pb) ** 2
        if dist_sq < min_dist:
            min_dist = dist_sq
            best_color = (pr, pg, pb)
            if dist_sq == 0:
                break
    return best_color
