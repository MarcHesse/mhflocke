"""
MH-FLOCKE — Overlay Base v0.4.1
========================================
Font loading and base utilities for dashboard rendering.
"""

import math
import numpy as np
from typing import Tuple, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter


# =====================================================================
# COLOR PALETTE — Flocke Dark Theme
# =====================================================================

# Backgrounds
DEEP_SPACE = (10, 12, 18)
PANEL_BG = (16, 20, 30)

# Primary
COOL_BLUE = (60, 130, 200)
DEEP_BLUE = (30, 60, 120)
ICE_BLUE = (140, 180, 220)
CYAN_ACCENT = (0, 200, 220)

# Warm
FIRE_ORANGE = (255, 107, 53)
WARN_AMBER = (245, 158, 11)

# Signal
SPIKE_RED = (255, 60, 40)
SUCCESS_GREEN = (50, 200, 100)
VIOLET_PULSE = (160, 80, 220)

# Neutral
BORDER = (40, 50, 70)
BAR_BG = (25, 30, 42)
SECONDARY = (100, 110, 130)


# =====================================================================
# FONT SYSTEM
# =====================================================================

_font_cache = {}


def get_font(size: int = 11, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Get a monospace font, cached."""
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]

    suffix = '-Bold' if bold else ''
    for path in [
        f'/usr/share/fonts/truetype/dejavu/DejaVuSansMono{suffix}.ttf',
        f'C:/Windows/Fonts/consola{"b" if bold else ""}.ttf',
        f'/System/Library/Fonts/Menlo.ttc',
    ]:
        try:
            font = ImageFont.truetype(path, size)
            _font_cache[key] = font
            return font
        except (OSError, IOError):
            continue

    font = ImageFont.load_default()
    _font_cache[key] = font
    return font


# =====================================================================
# COLOR UTILITIES
# =====================================================================

def _with_alpha(color: Tuple[int, int, int], alpha: int = 255) -> Tuple[int, int, int, int]:
    """Add alpha channel to RGB color."""
    return (color[0], color[1], color[2], max(0, min(255, int(alpha))))


def _lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int],
                t: float) -> Tuple[int, int, int]:
    """Linear interpolation between two RGB colors. t=0→c1, t=1→c2."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def _dim_color(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    """Dim a color by factor (0=black, 1=unchanged)."""
    return (
        int(color[0] * factor),
        int(color[1] * factor),
        int(color[2] * factor),
    )


# =====================================================================
# GLASS PANEL SYSTEM
# =====================================================================

def create_glass_panel(width: int, height: int,
                       opacity: int = 140,
                       color: Tuple[int, int, int] = DEEP_SPACE) -> Image.Image:
    """
    Create a semi-transparent glass panel.

    Returns RGBA Image with uniform opacity.
    """
    panel = Image.new('RGBA', (width, height), (*color, opacity))

    # Subtle border glow at edges
    draw = ImageDraw.Draw(panel)
    draw.rectangle([0, 0, width - 1, height - 1],
                   outline=(*BORDER, 80))

    return panel


def apply_glass_overlay(base_image: Image.Image,
                        overlay: Image.Image,
                        panel_x: int = 0, panel_y: int = 0,
                        blur_behind: int = 2) -> Image.Image:
    """
    Composite a glass overlay onto a base image.

    Args:
        base_image: Background (RGB or RGBA)
        overlay: Glass panel (RGBA with transparency)
        panel_x, panel_y: Position on base
        blur_behind: Gaussian blur radius for frosted glass effect

    Returns:
        Composited image (RGB)
    """
    result = base_image.convert('RGBA').copy()
    ow, oh = overlay.size

    # Optional: blur the region behind the panel for frosted glass effect
    if blur_behind > 0:
        # Extract the region, blur it, paste back
        region = result.crop((panel_x, panel_y,
                              min(panel_x + ow, result.width),
                              min(panel_y + oh, result.height)))
        blurred = region.filter(ImageFilter.GaussianBlur(radius=blur_behind))
        result.paste(blurred, (panel_x, panel_y))

    # Composite overlay with alpha
    result.paste(overlay, (panel_x, panel_y), overlay)

    return result.convert('RGB')


# =====================================================================
# DRAWING PRIMITIVES
# =====================================================================

def draw_section_header(draw: ImageDraw.Draw, x: int, y: int,
                        text: str, width: int,
                        color: Tuple[int, int, int] = CYAN_ACCENT,
                        font: Optional[ImageFont.FreeTypeFont] = None):
    """
    Draw a section header with subtle underline.

    Returns: y position after header
    """
    font = font or get_font(10, bold=True)
    draw.text((x, y), text, fill=(*color, 200), font=font)
    y += 14
    # Subtle line
    draw.line([(x, y), (x + width - 8, y)], fill=(*BORDER, 100), width=1)
    return y + 4


def draw_labeled_bar(draw: ImageDraw.Draw, x: int, y: int,
                     label: str, value: float,
                     color: Tuple[int, int, int] = COOL_BLUE,
                     width: int = 200, height: int = 8,
                     max_val: float = 1.0,
                     font: Optional[ImageFont.FreeTypeFont] = None) -> int:
    """
    Draw a labeled progress bar.

    Returns: y position after bar
    """
    font = font or get_font(9)

    # Label
    draw.text((x, y), label, fill=(*SECONDARY, 180), font=font)
    label_w = 65

    # Bar track
    bx = x + label_w
    bw = width - label_w - 40
    draw.rectangle([bx, y + 2, bx + bw, y + 2 + height],
                   fill=(*BAR_BG, 180))

    # Bar fill
    fill_w = int(bw * min(max(value / max_val, 0), 1.0))
    if fill_w > 0:
        draw.rectangle([bx, y + 2, bx + fill_w, y + 2 + height],
                       fill=(*color, 200))

    # Value text
    draw.text((bx + bw + 4, y), f"{value:.2f}",
              fill=(*SECONDARY, 140), font=font)

    return y + height + 6


def draw_key_value(draw: ImageDraw.Draw, x: int, y: int,
                   key: str, value: str,
                   key_color: Tuple[int, int, int] = SECONDARY,
                   val_color: Tuple[int, int, int] = ICE_BLUE,
                   font: Optional[ImageFont.FreeTypeFont] = None) -> int:
    """
    Draw a key: value pair.

    Returns: y position after line
    """
    font = font or get_font(9)
    draw.text((x, y), f"{key}:", fill=(*key_color, 160), font=font)
    draw.text((x + 75, y), str(value), fill=(*val_color, 200), font=font)
    return y + 14


def draw_sparkline(draw: ImageDraw.Draw, x: int, y: int,
                   values: list, width: int = 100, height: int = 20,
                   color: Tuple[int, int, int] = COOL_BLUE):
    """
    Draw a mini sparkline graph.
    """
    if not values or len(values) < 2:
        return

    # Normalize values
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        vmax = vmin + 1

    points = []
    for i, v in enumerate(values[-width:]):
        px = x + int(i * width / len(values[-width:]))
        py = y + height - int((v - vmin) / (vmax - vmin) * height)
        points.append((px, py))

    if len(points) >= 2:
        draw.line(points, fill=(*color, 160), width=1)


def draw_emotion_indicator(draw: ImageDraw.Draw, x: int, y: int,
                           valence: float, arousal: float,
                           width: int = 60, height: int = 60,
                           color: Tuple[int, int, int] = FIRE_ORANGE):
    """
    Draw a Russell circumplex emotion indicator (2D: valence × arousal).
    """
    cx = x + width // 2
    cy = y + height // 2

    # Background circle
    r = min(width, height) // 2 - 2
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 outline=(*BORDER, 80), width=1)

    # Crosshair
    draw.line([(cx - r, cy), (cx + r, cy)], fill=(*BORDER, 40), width=1)
    draw.line([(cx, cy - r), (cx, cy + r)], fill=(*BORDER, 40), width=1)

    # Dot position
    dx = int(valence * r * 0.8)
    dy = -int(arousal * r * 0.8)
    draw.ellipse([cx + dx - 3, cy + dy - 3, cx + dx + 3, cy + dy + 3],
                 fill=(*color, 220))


# Additional colors needed by evo_overlay, brain_overlay etc.
OUTPUT_GOLD = (255, 200, 60)
GLASS_TINT = (16, 20, 30)
FLOCKE_GOLD = (255, 200, 60)
AROUSAL_GOLD = (245, 180, 40)
TEXT_PRIMARY = (220, 225, 235)
TEXT_SECONDARY = (140, 150, 170)
TEXT_DIM = (80, 90, 110)
MIDNIGHT = (10, 12, 18)
SLATE = (30, 35, 50)
