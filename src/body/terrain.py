"""
MH-FLOCKE — Terrain v0.4.1
========================================
Procedural heightfield generation and ball injection.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class TerrainConfig:
    """Configuration for terrain generation."""
    terrain_type: str = 'flat'       # flat, hilly_grassland, rocky, slopes, steps
    difficulty: float = 0.3          # 0.0 = trivial, 1.0 = extreme
    size_x: float = 100.0            # meters — EXACT config of v034_1773145381 (29.9m, 0 falls)
    size_y: float = 100.0
    resolution: int = 200            # heightfield grid points per axis — ORIGINAL
    max_height: float = 0.80         # 0.80m: visible with seitliche Beleuchtung. 0.40m=invisible!
    seed: int = 42
    # Visual
    texture_rgb1: str = '0.22 0.38 0.12'   # grass green dark (more contrast)
    texture_rgb2: str = '0.42 0.62 0.25'   # grass green light (more contrast)


def generate_heightfield(config: TerrainConfig) -> np.ndarray:
    """
    Generate a 2D heightmap array for MuJoCo <hfield>.

    Returns:
        heightmap: float array [resolution, resolution], values 0.0 to 1.0
                   (MuJoCo scales by hfield size z-component)
    """
    rng = np.random.RandomState(config.seed)
    res = config.resolution
    hmap = np.zeros((res, res), dtype=np.float32)

    if config.terrain_type == 'flat':
        return hmap

    elif config.terrain_type == 'hilly_grassland':
        # Two-layer approach: long waves (background) + medium detail (visible)
        # Layer 1: gentle background (freq=2, visible from far)
        base = _perlin_like(res, octaves=3, persistence=0.4, rng=rng, freq_start=2)
        # Layer 2: medium hills (freq=6, visible near camera)
        detail = _perlin_like(res, octaves=3, persistence=0.5, rng=rng, freq_start=6)
        # Blend: mostly base (walkable) + some detail (visible)
        hmap = base * 0.6 + detail * 0.4
        hmap *= config.difficulty
        # Gradient: terrain gets hillier further from center
        # This gives the Go2 a flat start area that gradually gets hilly
        cx, cy = res // 2, res // 2
        for i in range(res):
            for j in range(res):
                dist_from_center = np.sqrt((i - cx)**2 + (j - cy)**2) / (res * 0.5)
                # 0 at center, 1 at edges — hills grow with distance
                gradient = np.clip(dist_from_center * 1.5 - 0.15, 0.0, 1.0)
                hmap[i, j] *= gradient

    elif config.terrain_type == 'rocky_mountain':
        hmap = _perlin_like(res, octaves=6, persistence=0.65, rng=rng)
        # Add sharp ridges
        ridges = _perlin_like(res, octaves=2, persistence=0.3, rng=rng)
        hmap += np.abs(ridges) * 0.5
        hmap *= config.difficulty

    elif config.terrain_type in ('slopes', 'gentle_slopes'):
        # Long sine waves
        x = np.linspace(0, 4 * np.pi, res)
        y = np.linspace(0, 3 * np.pi, res)
        X, Y = np.meshgrid(x, y)
        hmap = (np.sin(X) * 0.5 + np.sin(Y * 0.7) * 0.3 +
                np.sin(X * 0.3 + Y * 0.5) * 0.2)
        hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
        hmap *= config.difficulty

    elif config.terrain_type == 'steps':
        # Discrete height levels
        base = _perlin_like(res, octaves=2, persistence=0.3, rng=rng)
        n_levels = max(3, int(5 * config.difficulty + 3))
        hmap = np.round(base * n_levels) / n_levels
        hmap = np.clip(hmap, 0.0, 1.0)

    else:
        # Unknown type → gentle hills as fallback
        hmap = _perlin_like(res, octaves=3, persistence=0.4, rng=rng)
        hmap *= config.difficulty * 0.5

    # Flatten spawn area (center 3x3m) so creature starts on flat ground
    cx, cy = res // 2, res // 2
    spawn_r = int(res * 1.5 / config.size_x)  # ~1.5m radius
    for i in range(max(0, cx - spawn_r), min(res, cx + spawn_r)):
        for j in range(max(0, cy - spawn_r), min(res, cy + spawn_r)):
            dist = np.sqrt((i - cx)**2 + (j - cy)**2) / spawn_r
            if dist < 1.0:
                # Smooth blend to flat
                blend = dist ** 2  # quadratic: flat center, gradual transition
                hmap[i, j] *= blend

    # Normalize to [0, 1]
    if hmap.max() > 0:
        hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)

    return hmap


def _perlin_like(res: int, octaves: int = 4, persistence: float = 0.5,
                 rng: np.random.RandomState = None,
                 freq_start: int = 2) -> np.ndarray:
    """
    Simple multi-octave value noise (Perlin-like, no external deps).

    Each octave doubles frequency, halves amplitude. Produces smooth
    terrain suitable for locomotion training.
    
    freq_start: base frequency (2=very long waves, 8=visible bumps at 100m)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    total = np.zeros((res, res), dtype=np.float32)
    amplitude = 1.0
    freq = freq_start  # Higher = more visible hills

    for _ in range(octaves):
        # Random grid at this frequency
        grid = rng.randn(freq + 1, freq + 1).astype(np.float32)
        # Bilinear interpolation to full resolution
        from_x = np.linspace(0, freq, res)
        from_y = np.linspace(0, freq, res)
        # Simple bilinear via numpy
        xi = np.clip(from_x, 0, freq - 1e-6)
        yi = np.clip(from_y, 0, freq - 1e-6)
        x0 = np.floor(xi).astype(int)
        y0 = np.floor(yi).astype(int)
        x1 = np.minimum(x0 + 1, freq)
        y1 = np.minimum(y0 + 1, freq)
        xf = (xi - x0).reshape(1, -1)
        yf = (yi - y0).reshape(-1, 1)

        c00 = grid[y0][:, x0]
        c10 = grid[y1][:, x0]
        c01 = grid[y0][:, x1]
        c11 = grid[y1][:, x1]

        interp = (c00 * (1 - xf) * (1 - yf) +
                  c01 * xf * (1 - yf) +
                  c10 * (1 - xf) * yf +
                  c11 * xf * yf)

        total += interp * amplitude
        amplitude *= persistence
        freq *= 2

    # Normalize to [0, 1]
    total = (total - total.min()) / (total.max() - total.min() + 1e-8)
    return total


def heightfield_to_png(hmap: np.ndarray) -> bytes:
    """
    Convert heightmap to 8-bit grayscale PNG bytes.
    MuJoCo can load heightfields from PNG files.
    Uses raw PNG encoding (no PIL dependency).
    """
    import struct
    import zlib

    h, w = hmap.shape
    # Quantize to 8-bit
    data_8bit = (np.clip(hmap, 0, 1) * 255).astype(np.uint8)

    # PNG minimal encoder
    def make_chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack('>I', len(data)) + c + crc

    # IHDR
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0)  # 8bit grayscale
    # IDAT
    raw_rows = b''
    for row in range(h):
        raw_rows += b'\x00' + data_8bit[row].tobytes()  # filter=None
    compressed = zlib.compress(raw_rows)
    # Build PNG
    png = b'\x89PNG\r\n\x1a\n'
    png += make_chunk(b'IHDR', ihdr)
    png += make_chunk(b'IDAT', compressed)
    png += make_chunk(b'IEND', b'')
    return png


def inject_terrain(xml_string: str, config: TerrainConfig,
                   hfield_png_path: str = '/tmp/mhflocke_terrain.png') -> str:
    """
    Inject heightfield terrain into MuJoCo XML string.

    1. Generates heightmap
    2. Saves as PNG
    3. Injects <hfield> asset + <geom type="hfield"> into XML

    Args:
        xml_string: base MJCF XML (creature model)
        config: terrain configuration
        hfield_png_path: where to save the PNG (must be accessible by MuJoCo)

    Returns:
        Modified XML string with terrain
    """
    import re

    # Generate and save heightmap
    hmap = generate_heightfield(config)
    png_bytes = heightfield_to_png(hmap)
    with open(hfield_png_path, 'wb') as f:
        f.write(png_bytes)

    max_h = config.max_height * config.difficulty / 0.3  # scale height by difficulty
    max_h = max(0.001, min(1.0, max_h))  # clamp
    half_x = config.size_x / 2
    half_y = config.size_y / 2

    # Asset: heightfield definition
    hfield_asset = (
        f'\n    <hfield name="terrain" file="{hfield_png_path}" '
        f'size="{half_x} {half_y} {max_h} 0.01"/>'
    )

    # Texture for terrain (gradient instead of checker — shows hills via shading)
    terrain_texture = (
        f'\n    <texture name="terrain_tex" type="2d" builtin="gradient" '
        f'rgb1="{config.texture_rgb1}" rgb2="{config.texture_rgb2}" '
        f'width="512" height="512"/>'
        f'\n    <material name="terrain_mat" texture="terrain_tex" '
        f'texrepeat="12 12" texuniform="true" specular="0.1" shininess="0.3"/>'
    )

    # Geom: the actual terrain surface in worldbody
    terrain_geom = (
        f'\n    <geom name="terrain_floor" type="hfield" hfield="terrain" '
        f'material="terrain_mat" pos="0 0 0" '
        f'friction="1.0 0.5 0.01" conaffinity="1" condim="3"/>'
    )

    # Inject into XML
    # 1. Add hfield + texture to <asset>
    if '<asset>' in xml_string:
        xml_string = xml_string.replace(
            '<asset>',
            '<asset>' + hfield_asset + terrain_texture
        )
    else:
        # No asset section — add before worldbody
        xml_string = xml_string.replace(
            '<worldbody>',
            '<asset>' + hfield_asset + terrain_texture + '\n  </asset>\n  <worldbody>'
        )

    # 2. Add terrain geom to worldbody (BEFORE creature body)
    # Also remove existing flat floor if present
    xml_string = re.sub(
        r'<geom[^>]*name="floor"[^/]*/>', '', xml_string)
    xml_string = re.sub(
        r'<geom[^>]*type="plane"[^/]*/>', '', xml_string)

    xml_string = xml_string.replace(
        '<worldbody>',
        '<worldbody>' + terrain_geom
    )

    return xml_string


def inject_terrain_geoms(xml_string: str, config: TerrainConfig) -> str:
    """
    Inject visible 3D hill objects into MuJoCo XML.
    
    Instead of a heightfield (which is invisible at camera distance),
    this places ellipsoid geoms as hills on the ground plane.
    These have real 3D silhouettes, cast shadows, and are clearly visible.
    
    The Go2 walks over/around these hills. Foot contact sensors + terrain
    reflexes handle the interaction.
    """
    import re
    rng = np.random.RandomState(config.seed)
    
    n_hills = int(15 + config.difficulty * 30)  # 15-45 hills
    max_h = config.max_height * config.difficulty / 0.3
    half_x = config.size_x / 2
    half_y = config.size_y / 2
    spawn_clear = 3.0  # meters clear around spawn
    
    # Green hill material
    hill_material = (
        '\n    <texture name="hill_tex" type="2d" builtin="gradient" '
        f'rgb1="{config.texture_rgb1}" rgb2="{config.texture_rgb2}" '
        'width="256" height="256"/>'
        '\n    <material name="hill_mat" texture="hill_tex" '
        'specular="0.05" shininess="0.1" reflectance="0.05"/>'
    )
    
    # Generate hill geoms
    hill_geoms = ''
    for i in range(n_hills):
        # Random position (avoid spawn area)
        while True:
            hx = rng.uniform(-half_x * 0.7, half_x * 0.7)
            hy = rng.uniform(-half_y * 0.7, half_y * 0.7)
            if np.sqrt(hx**2 + hy**2) > spawn_clear:
                break
        
        # Random size — height 0.15 to max_h, radius 1.5 to 4m
        h = rng.uniform(0.15, max(0.2, max_h))
        rx = rng.uniform(1.5, 4.0)
        ry = rng.uniform(1.5, 4.0)
        angle = rng.uniform(0, 360)
        
        # Ellipsoid half-buried: pos z = -h*0.3 (30% underground)
        hz = -h * 0.3
        
        hill_geoms += (
            f'\n    <geom name="hill_{i}" type="ellipsoid" '
            f'size="{rx:.2f} {ry:.2f} {h:.2f}" '
            f'pos="{hx:.2f} {hy:.2f} {hz:.2f}" '
            f'euler="0 0 {angle:.0f}" '
            f'material="hill_mat" '
            f'friction="1.0 0.5 0.01" conaffinity="1" condim="3"/>'
        )
    
    # Inject material into <asset>
    if '<asset>' in xml_string:
        xml_string = xml_string.replace('<asset>', '<asset>' + hill_material)
    
    # Inject hill geoms into worldbody (before creature)
    xml_string = xml_string.replace(
        '<worldbody>',
        '<worldbody>' + hill_geoms
    )
    
    print(f'  Injected {n_hills} hill geoms (h={max_h:.2f}m, r=1.5-4.0m)')
    return xml_string


def terrain_type_from_scene(scene_text: str) -> str:
    """
    Infer terrain type from free-text scene description.
    Used when LLM returns scene description but no explicit terrain type.

    Examples:
        "hilly grassland" -> "hilly_grassland"
        "rocky mountain" -> "rocky_mountain"
        "flat meadow" -> "flat"
    """
    text = scene_text.lower()
    if any(w in text for w in ['hill', 'huegel', 'hilly', 'rolling']):
        return 'hilly_grassland'
    if any(w in text for w in ['rock', 'fels', 'mountain', 'berg', 'rough']):
        return 'rocky_mountain'
    if any(w in text for w in ['slope', 'ramp', 'steigung', 'incline']):
        return 'gentle_slopes'
    if any(w in text for w in ['step', 'stufe', 'stair', 'treppe']):
        return 'steps'
    if any(w in text for w in ['flat', 'flach', 'eben', 'meadow', 'wiese']):
        return 'flat'
    return 'hilly_grassland'  # default: something interesting


def difficulty_from_scene(scene_text: str) -> float:
    """
    Infer terrain difficulty from scene description.

    Examples:
        "gentle hilly grassland" -> 0.25
        "rough rocky terrain" -> 0.7
        "extreme mountain" -> 0.9
    """
    text = scene_text.lower()
    d = 0.3  # default moderate
    if any(w in text for w in ['gentle', 'sanft', 'leicht', 'easy', 'mild']):
        d -= 0.15
    if any(w in text for w in ['moderate', 'mittel']):
        d = 0.4
    if any(w in text for w in ['rough', 'rau', 'difficult', 'schwer', 'hard']):
        d += 0.2
    if any(w in text for w in ['extreme', 'extrem', 'steep', 'steil']):
        d += 0.4
    return max(0.05, min(0.95, d))


def inject_ball(xml_string: str, pos=(8.0, 1.0, 0.12), radius=0.1, mass=0.3) -> str:
    """
    Inject a free-rolling ball into a MuJoCo scene XML.

    The ball is a scene object, NOT part of the robot model.
    It gets injected dynamically like terrain — only when the scene requires it.

    Biology: a toy/ball on a meadow is an environmental stimulus,
    not part of the dog's body. It triggers visual salience -> approach -> play.

    Args:
        xml_string: MuJoCo XML as string
        pos: (x, y, z) spawn position
        radius: ball radius in meters
        mass: ball mass in kg

    Returns:
        Modified XML string with ball body added to worldbody
    """
    ball_xml = f"""
    <!-- Ball: free-rolling sphere for interaction (injected by scene) -->
    <body name="ball" pos="{pos[0]} {pos[1]} {pos[2]}">
      <freejoint name="ball_joint"/>
      <geom name="ball_geom" type="sphere" size="{radius}" mass="{mass}"
            rgba="0.9 0.2 0.15 1" friction="0.8 0.3 0.01"
            condim="3" conaffinity="1"/>
    </body>"""
    # Insert before </worldbody>
    return xml_string.replace('</worldbody>', ball_xml + '\n  </worldbody>')


def inject_light(xml_string: str, pos=(2.0, 0.5, 0.02), radius=0.15) -> str:
    """
    Inject a bright light source (glowing sphere) into a MuJoCo scene.

    The light source is a visual target for phototaxis navigation.
    It has a freejoint so its position can be updated via qpos,
    but very low mass so it doesn't affect physics meaningfully.

    The sphere uses high emission (1.0) and bright white/yellow color
    so the onboard camera can detect it via bilateral brightness.
    A real MuJoCo <light> is also attached to create actual illumination
    on the ground around the sphere, making the camera gradient realistic.

    Hardware equivalent: flashlight pointed at the floor.

    Args:
        xml_string: MuJoCo XML as string
        pos: (x, y, z) initial position
        radius: visual sphere radius

    Returns:
        Modified XML with light source body added
    """
    # Emissive material must go in <asset> section
    light_material = """
    <material name="light_emissive" rgba="1.0 0.95 0.7 0.9" emission="1"/>"""

    light_body = f"""
    <!-- Light source: glowing sphere for phototaxis (injected by scene) -->
    <body name="light_target" pos="{pos[0]} {pos[1]} {pos[2]}">
      <freejoint name="light_joint"/>
      <geom name="light_geom" type="sphere" size="{radius}" mass="0.001"
            material="light_emissive"
            condim="1" conaffinity="0" contype="0"/>
      <light name="light_spot" pos="0 0 0.05" dir="0 0 -1"
             diffuse="1.0 0.95 0.7" specular="0.5 0.5 0.3"
             attenuation="1.0 0.5 0.2" cutoff="60" exponent="10"/>
    </body>"""

    # Insert material into <asset> (before closing </asset>)
    if '</asset>' in xml_string:
        xml_string = xml_string.replace('</asset>',
                                        light_material + '\n  </asset>')

    # Insert body before </worldbody>
    return xml_string.replace('</worldbody>', light_body + '\n  </worldbody>')


def inject_wall(xml_string: str, distance: float = 1.5,
                width: float = 3.0, height: float = 0.3,
                thickness: float = 0.20) -> str:
    """
    Inject a static wall obstacle into a MuJoCo scene XML.

    The wall is a BARE GEOM directly in worldbody — not wrapped in a <body>.
    This is critical because load_creature_from_string() parses all <body>
    children as robot parts. A bare geom in worldbody is truly static and
    belongs to the world body (body 0), which is the immovable ground.

    The wall is perpendicular to the X-axis (walking direction),
    centered at Y=0. The creature must learn to stop or turn.

    Args:
        xml_string: MuJoCo XML as string
        distance: X position of wall center (meters from origin)
        width: Wall width in Y direction (meters)
        height: Wall height in Z direction (meters)
        thickness: Wall thickness in X direction (meters)

    Returns:
        Modified XML string with wall geom added to worldbody
    """
    # Wall material (dark red, clearly visible)
    wall_material = (
        '\n    <material name="wall_mat" rgba="0.6 0.15 0.1 1" '
        'specular="0.2" shininess="0.3"/>'
    )

    # Size is half-extents for box geom
    hx = thickness / 2
    hy = width / 2
    hz = height / 2

    # Bare geom in worldbody — no <body> wrapper!
    # A geom directly in <worldbody> belongs to world body 0 (immovable).
    # load_creature_from_string only parses <body> children, so this
    # geom survives as a static world obstacle.
    wall_xml = f"""
    <!-- Wall obstacle: static barrier for obstacle avoidance (Issue #103) -->
    <geom name="wall_geom" type="box" pos="{distance:.3f} 0 {hz:.3f}"
          size="{hx:.3f} {hy:.3f} {hz:.3f}"
          material="wall_mat" friction="1.0 0.5 0.01"
          conaffinity="1" contype="1" condim="3"/>"""

    # Inject material into <asset>
    if '<asset>' in xml_string and 'wall_mat' not in xml_string:
        xml_string = xml_string.replace('<asset>', '<asset>' + wall_material)

    # Insert wall geom right after the floor geom (before any <body>)
    # This ensures it's a worldbody-level geom, not inside a creature body
    if '<geom name="floor"' in xml_string:
        # Find end of floor geom line and insert after it
        floor_end = xml_string.find('/>', xml_string.find('<geom name="floor"'))
        if floor_end >= 0:
            insert_pos = floor_end + 2
            xml_string = xml_string[:insert_pos] + wall_xml + xml_string[insert_pos:]
        else:
            xml_string = xml_string.replace('</worldbody>', wall_xml + '\n  </worldbody>')
    else:
        xml_string = xml_string.replace('</worldbody>', wall_xml + '\n  </worldbody>')

    print(f'  Wall: injected at x={distance:.1f}m (w={width:.1f}m, h={height:.1f}m)')
    return xml_string
