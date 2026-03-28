"""
MH-FLOCKE — Scene Builder v0.4.1
========================================
Scene composition from terrain, objects, and creatures.
"""

import os
import copy
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class SceneObject:
    """Ein Objekt in der Szene (Stein, Rampe, Ball, Wand, ...)."""
    name: str
    geom_type: str = 'box'        # box, sphere, cylinder, capsule, mesh
    pos: str = '0 0 0'
    size: str = '0.1 0.1 0.1'
    euler: str = '0 0 0'
    rgba: str = '0.5 0.5 0.5 1'
    friction: str = '1.0 0.5 0.01'
    mass: float = 0.0             # 0 = static
    material: str = ''            # Optional material name


@dataclass
class Scene:
    """Definition einer MuJoCo-Welt/Szene."""
    name: str
    description: str = ''

    # Boden
    floor_friction: str = '1.0 0.5 0.01'
    floor_rgba: str = ''           # Leer = Textur behalten
    floor_material: str = ''       # Leer = default grass_mat
    floor_size: str = '20 20 0.1'

    # Texture overrides (name -> RGBA or texture params)
    floor_texture_rgb1: str = ''   # Checker RGB1
    floor_texture_rgb2: str = ''   # Checker RGB2
    floor_texture_type: str = ''   # '2d' oder 'cube'

    # Skybox
    skybox_rgb1: str = ''          # Gradient oben
    skybox_rgb2: str = ''          # Gradient unten

    # Licht
    light_diffuse: str = '0.6 0.55 0.5'
    light_specular: str = '0.3 0.3 0.3'
    light_pos: str = '0 0 3'
    light_dir: str = '0 0 -1'
    fill_light: bool = True

    # Atmosphaere
    haze_rgba: str = '0.04 0.055 0.10 1'

    # Physik-Overrides
    gravity: str = ''              # Leer = default
    wind: str = ''                 # x y z Kraft

    # Objekte
    objects: List[SceneObject] = field(default_factory=list)

    # Extra MJCF-Snippets (roh)
    extra_assets: str = ''
    extra_worldbody: str = ''


class SceneBuilder:
    """Injiziert eine Szene in die Hunde-MJCF."""

    @staticmethod
    def build(base_xml_path: str, scene: Scene) -> str:
        """
        Liest Hunde-MJCF, ersetzt Welt-Elemente mit Szene.
        Gibt kombinierten XML-String zurueck.
        """
        import re
        base_xml_path = os.path.abspath(base_xml_path)
        xml_dir = os.path.dirname(base_xml_path).replace('\\', '/')

        with open(base_xml_path, 'r', encoding='utf-8') as f:
            xml = f.read()

        # meshdir/texturedir auf absolute Pfade patchen
        xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{xml_dir}/"', xml)
        xml = re.sub(r'texturedir="[^"]*"', f'texturedir="{xml_dir}/"', xml)

        # 1. Boden-Friction ersetzen
        if scene.floor_friction:
            xml = SceneBuilder._replace_attr(xml, 'geom', 'name', 'floor',
                                             'friction', scene.floor_friction)

        # 2. Ground material or color
        if scene.floor_rgba:
            xml = SceneBuilder._replace_attr(xml, 'geom', 'name', 'floor',
                                             'rgba', scene.floor_rgba)
            # Remove material if direct color
            xml = xml.replace('material="grass_mat"', '', 1)

        # 3. Change floor texture (checker pattern for ice etc.)
        if scene.floor_texture_rgb1:
            # Ersetze grass_tex mit Checker
            checker = (f'<texture name="grass_tex" type="2d" builtin="checker" '
                       f'rgb1="{scene.floor_texture_rgb1}" '
                       f'rgb2="{scene.floor_texture_rgb2}" '
                       f'width="512" height="512"/>')
            # Alten Texture-Eintrag ersetzen
            import re
            xml = re.sub(r'<texture name="grass_tex"[^/]*/>', checker, xml)

        # 4. Licht
        if scene.light_diffuse:
            xml = SceneBuilder._replace_light(xml, scene)

        # 5. Haze
        if scene.haze_rgba:
            xml = SceneBuilder._replace_attr(xml, 'rgba', None, None,
                                             'haze', scene.haze_rgba)
            import re
            xml = re.sub(r'rgba haze="[^"]*"',
                         f'rgba haze="{scene.haze_rgba}"', xml)

        # 6. Gravity Override
        if scene.gravity:
            import re
            xml = re.sub(r'gravity="[^"]*"', f'gravity="{scene.gravity}"', xml)

        # 6b. Visual Quality: Shadows + Antialiasing
        import re
        if '<visual>' not in xml:
            # Inject visual section before </mujoco>
            visual_xml = (
                '\n  <visual>'
                '\n    <quality shadowsize="2048" offsamples="8"/>'
                '\n    <map shadowclip="2.0"/>'
                '\n    <headlight ambient="0.15 0.15 0.18" diffuse="0.0 0.0 0.0" specular="0.0 0.0 0.0"/>'
                '\n  </visual>\n'
            )
            xml = xml.replace('</mujoco>', f'{visual_xml}</mujoco>')
        else:
            # Upgrade existing visual: ensure shadow quality
            if 'shadowsize' not in xml:
                xml = re.sub(
                    r'(<quality[^/]*)(/>)',
                    r'\1 shadowsize="2048"\2', xml, count=1)

        # 7. Wind (als Option)
        if scene.wind:
            import re
            xml = re.sub(r'(<option[^>]*)(/>|>)',
                         rf'\1 wind="{scene.wind}"\2', xml, count=1)

        # 8. Skybox Override
        if scene.skybox_rgb1:
            skybox = (f'<texture name="skybox" type="skybox" builtin="gradient" '
                      f'rgb1="{scene.skybox_rgb1}" rgb2="{scene.skybox_rgb2}" '
                      f'width="512" height="3072"/>')
            xml = re.sub(r'<texture name="skybox"[^/]*/>', skybox, xml)

        # 9. Extra Assets injizieren
        if scene.extra_assets:
            xml = xml.replace('</asset>', f'    {scene.extra_assets}\n  </asset>')

        # 9. Objekte in Worldbody injizieren (vor dem Hund)
        if scene.objects or scene.extra_worldbody:
            obj_xml = ''
            for obj in scene.objects:
                attrs = f'name="{obj.name}" type="{obj.geom_type}" '
                attrs += f'pos="{obj.pos}" size="{obj.size}" '
                if obj.euler and obj.euler != '0 0 0':
                    attrs += f'euler="{obj.euler}" '
                if obj.rgba:
                    attrs += f'rgba="{obj.rgba}" '
                if obj.friction:
                    attrs += f'friction="{obj.friction}" '
                if obj.mass > 0:
                    attrs += f'mass="{obj.mass}" '
                if obj.material:
                    attrs += f'material="{obj.material}" '
                obj_xml += f'    <geom {attrs}/>\n'
            if scene.extra_worldbody:
                obj_xml += f'    {scene.extra_worldbody}\n'
            # Nach Floor-Geom einfuegen
            xml = xml.replace(
                'friction="1.0 0.5 0.01"/>',
                f'friction="{scene.floor_friction}"/>\n{obj_xml}',
                1)

        return xml

    @staticmethod
    def _replace_attr(xml, tag, match_attr, match_val, attr, new_val):
        """Ersetzt ein Attribut in einem spezifischen XML-Element."""
        import re
        if match_attr and match_val:
            pattern = f'({tag}[^>]*{match_attr}="{match_val}"[^>]*){attr}="[^"]*"'
            xml = re.sub(pattern, rf'\1{attr}="{new_val}"', xml)
        return xml

    @staticmethod
    def _replace_light(xml, scene):
        """Ersetzt Hauptlicht-Parameter und fuegt 3-Punkt-Beleuchtung hinzu."""
        import re
        # KEY LIGHT: Hauptlicht (erstes directional) — staerkstes Licht
        xml = re.sub(
            r'(<light[^>]*directional="true"[^>]*)(diffuse=)"[^"]*"',
            rf'\1\2"{scene.light_diffuse}"', xml, count=1)
        xml = re.sub(
            r'(<light[^>]*directional="true"[^>]*)(specular=)"[^"]*"',
            rf'\1\2"{scene.light_specular}"', xml, count=1)

        # FILL LIGHT + RIM LIGHT (3-Punkt-Beleuchtung)
        if scene.fill_light:
            # Parse key light diffuse to dim for fill
            kd = [float(v) for v in scene.light_diffuse.split()]
            fill_d = ' '.join(f'{v*0.35:.2f}' for v in kd)  # 35% intensity
            rim_d = ' '.join(f'{v*0.25:.2f}' for v in kd)   # 25% intensity, behind

            fill_and_rim = (
                f'\n    <!-- 3-Point Lighting: Fill + Rim -->'
                f'\n    <light name="fill_light" directional="true" '
                f'pos="-3 -2 2" dir="0.5 0.3 -0.5" '
                f'diffuse="{fill_d}" specular="0.05 0.05 0.05" '
                f'castshadow="false"/>'
                f'\n    <light name="rim_light" directional="true" '
                f'pos="1 -4 2" dir="-0.2 0.8 -0.3" '
                f'diffuse="{rim_d}" specular="0.15 0.15 0.2" '
                f'castshadow="false"/>'
            )
            # Inject after the main light
            xml = re.sub(
                r'(</light>)',
                rf'\1{fill_and_rim}',
                xml, count=1)

        return xml

    @staticmethod
    def get_scene_names() -> List[str]:
        return list(SCENES.keys())


# =====================================================================
# VORDEFINIERTE SZENEN
# =====================================================================

SCENES: Dict[str, Scene] = {}

# --- Flat Grass (Default) ---
SCENES['flat_grass'] = Scene(
    name='flat_grass',
    description='Flache Graswiese — Standard-Lernumgebung',
    floor_friction='1.0 0.5 0.01',
    floor_texture_rgb1='0.22 0.42 0.12',   # Dark grass green
    floor_texture_rgb2='0.28 0.50 0.16',   # Light grass green
    light_diffuse='0.7 0.65 0.55',
    light_specular='0.25 0.25 0.2',
    haze_rgba='0.12 0.22 0.08 1',
    skybox_rgb1='0.45 0.65 0.85',
    skybox_rgb2='0.10 0.15 0.08',
)

# --- Ice ---
SCENES['ice'] = Scene(
    name='ice',
    description='Eisfläche — extrem rutschig, lernt Balance',
    floor_friction='0.05 0.01 0.001',
    floor_texture_rgb1='0.75 0.85 0.95',
    floor_texture_rgb2='0.85 0.92 0.98',
    light_diffuse='0.7 0.75 0.85',
    light_specular='0.5 0.5 0.6',
    haze_rgba='0.15 0.20 0.30 1',
    skybox_rgb1='0.6 0.72 0.88',
    skybox_rgb2='0.12 0.15 0.25',
)

# --- Sand ---
SCENES['sand'] = Scene(
    name='sand',
    description='Sandiger Boden — mittlere Reibung, weich',
    floor_friction='0.6 0.3 0.02',
    floor_texture_rgb1='0.82 0.72 0.55',
    floor_texture_rgb2='0.75 0.65 0.48',
    light_diffuse='0.85 0.75 0.55',
    light_specular='0.2 0.2 0.15',
    haze_rgba='0.22 0.20 0.14 1',
    skybox_rgb1='0.70 0.60 0.40',
    skybox_rgb2='0.15 0.12 0.08',
)

# --- Rocky ---
SCENES['rocky'] = Scene(
    name='rocky',
    description='Steiniges Gelaende mit Hindernissen',
    floor_friction='1.2 0.6 0.01',
    floor_texture_rgb1='0.4 0.38 0.35',
    floor_texture_rgb2='0.5 0.48 0.45',
    light_diffuse='0.5 0.5 0.55',
    haze_rgba='0.08 0.08 0.10 1',
    skybox_rgb1='0.35 0.40 0.50',
    skybox_rgb2='0.05 0.06 0.08',
    objects=[
        SceneObject(name='rock1', geom_type='box', pos='1.0 0.3 0.08',
                    size='0.15 0.12 0.08', rgba='0.45 0.42 0.38 1'),
        SceneObject(name='rock2', geom_type='sphere', pos='2.0 -0.2 0.1',
                    size='0.1 0.1 0.1', rgba='0.5 0.48 0.42 1'),
        SceneObject(name='rock3', geom_type='box', pos='3.0 0.1 0.12',
                    size='0.2 0.15 0.12', euler='5 0 15',
                    rgba='0.42 0.40 0.36 1'),
        SceneObject(name='rock4', geom_type='box', pos='1.5 -0.4 0.06',
                    size='0.1 0.08 0.06', rgba='0.48 0.45 0.40 1'),
        SceneObject(name='rock5', geom_type='sphere', pos='4.0 0.0 0.15',
                    size='0.15 0.15 0.15', rgba='0.38 0.36 0.33 1'),
    ],
)

# --- Neon Grassland (Warehouse.AI Arena) ---
SCENES['neon_grassland'] = Scene(
    name='neon_grassland',
    description='Futuristic AI training arena — green hills with neon grid and ambient glow',
    floor_friction='0.9 0.45 0.01',
    floor_texture_rgb1='0.06 0.18 0.04',    # Deep dark green
    floor_texture_rgb2='0.10 0.28 0.08',    # Slightly brighter green
    floor_size='30 30 0.1',
    light_diffuse='0.45 0.65 0.50',         # Green-tinted key light
    light_specular='0.20 0.35 0.25',
    fill_light=True,
    haze_rgba='0.02 0.08 0.06 1',           # Green atmospheric haze
    skybox_rgb1='0.15 0.45 0.75',           # Bright blue sky top
    skybox_rgb2='0.04 0.12 0.06',           # Dark green horizon
    extra_assets=(
        '<texture name="glow_tex" type="2d" builtin="checker" '
        'rgb1="0.0 0.15 0.0" rgb2="0.0 0.4 0.1" width="64" height="64"/>\n'
        '    <material name="glow_green" texture="glow_tex" emission="0.6" '
        'texrepeat="8 8" texuniform="true" reflectance="0.3"/>\n'
        '    <material name="neon_cyan" rgba="0.0 0.85 0.9 0.7" emission="1.2"/>\n'
        '    <material name="neon_green" rgba="0.0 0.9 0.3 0.7" emission="1.0"/>\n'
        '    <material name="arena_pillar" rgba="0.08 0.08 0.12 1" emission="0.05"/>'
    ),
    objects=[
        # === TERRAIN: Gentle hills as wide ramps ===
        SceneObject(name='hill1', geom_type='box', pos='2.0 0 0.04',
                    size='0.8 1.2 0.02', euler='0 -5 0',
                    rgba='0.08 0.22 0.06 1', friction='0.9 0.45 0.01'),
        SceneObject(name='hill2', geom_type='box', pos='4.5 0.3 0.10',
                    size='1.0 1.0 0.02', euler='0 -8 3',
                    rgba='0.07 0.20 0.05 1', friction='0.9 0.45 0.01'),
        SceneObject(name='hill3', geom_type='box', pos='7.0 -0.2 0.18',
                    size='1.2 1.2 0.02', euler='0 -10 -2',
                    rgba='0.09 0.24 0.07 1', friction='0.9 0.45 0.01'),
        SceneObject(name='hill4', geom_type='box', pos='10.0 0.1 0.12',
                    size='0.9 1.0 0.02', euler='0 6 0',
                    rgba='0.08 0.21 0.06 1', friction='0.9 0.45 0.01'),
        SceneObject(name='plateau1', geom_type='box', pos='5.5 0 0.15',
                    size='0.5 0.8 0.15',
                    rgba='0.06 0.18 0.05 1', friction='0.9 0.45 0.01'),
        # === NEON GRID LINES on floor (thin glowing strips) ===
        SceneObject(name='grid_x1', geom_type='box', pos='0 0 0.002',
                    size='15 0.005 0.002', rgba='0.0 0.6 0.3 0.5'),
        SceneObject(name='grid_x2', geom_type='box', pos='0 2.0 0.002',
                    size='15 0.005 0.002', rgba='0.0 0.6 0.3 0.4'),
        SceneObject(name='grid_x3', geom_type='box', pos='0 -2.0 0.002',
                    size='15 0.005 0.002', rgba='0.0 0.6 0.3 0.4'),
        SceneObject(name='grid_z1', geom_type='box', pos='3.0 0 0.002',
                    size='0.005 4.0 0.002', rgba='0.0 0.5 0.8 0.4'),
        SceneObject(name='grid_z2', geom_type='box', pos='6.0 0 0.002',
                    size='0.005 4.0 0.002', rgba='0.0 0.5 0.8 0.4'),
        SceneObject(name='grid_z3', geom_type='box', pos='9.0 0 0.002',
                    size='0.005 4.0 0.002', rgba='0.0 0.5 0.8 0.4'),
        # === CHECKPOINT PYLONS (glowing pillars at sides) ===
        SceneObject(name='pylon_l1', geom_type='cylinder', pos='3.0 2.5 0.25',
                    size='0.04 0.25', rgba='0.08 0.08 0.12 1'),
        SceneObject(name='pylon_r1', geom_type='cylinder', pos='3.0 -2.5 0.25',
                    size='0.04 0.25', rgba='0.08 0.08 0.12 1'),
        SceneObject(name='pylon_tip_l1', geom_type='sphere', pos='3.0 2.5 0.52',
                    size='0.06', rgba='0.0 0.9 0.3 0.8'),
        SceneObject(name='pylon_tip_r1', geom_type='sphere', pos='3.0 -2.5 0.52',
                    size='0.06', rgba='0.0 0.9 0.3 0.8'),
        SceneObject(name='pylon_l2', geom_type='cylinder', pos='6.0 2.5 0.25',
                    size='0.04 0.25', rgba='0.08 0.08 0.12 1'),
        SceneObject(name='pylon_r2', geom_type='cylinder', pos='6.0 -2.5 0.25',
                    size='0.04 0.25', rgba='0.08 0.08 0.12 1'),
        SceneObject(name='pylon_tip_l2', geom_type='sphere', pos='6.0 2.5 0.52',
                    size='0.06', rgba='0.0 0.85 0.9 0.8'),
        SceneObject(name='pylon_tip_r2', geom_type='sphere', pos='6.0 -2.5 0.52',
                    size='0.06', rgba='0.0 0.85 0.9 0.8'),
        SceneObject(name='pylon_l3', geom_type='cylinder', pos='9.0 2.5 0.25',
                    size='0.04 0.25', rgba='0.08 0.08 0.12 1'),
        SceneObject(name='pylon_r3', geom_type='cylinder', pos='9.0 -2.5 0.25',
                    size='0.04 0.25', rgba='0.08 0.08 0.12 1'),
        SceneObject(name='pylon_tip_l3', geom_type='sphere', pos='9.0 2.5 0.52',
                    size='0.06', rgba='0.0 0.9 0.3 0.8'),
        SceneObject(name='pylon_tip_r3', geom_type='sphere', pos='9.0 -2.5 0.52',
                    size='0.06', rgba='0.0 0.9 0.3 0.8'),
        # === AMBIENT GROUND GLOW (flat emissive panels) ===
        SceneObject(name='glow_pad1', geom_type='box', pos='0 0 0.001',
                    size='1.5 1.5 0.001', rgba='0.0 0.15 0.05 0.3'),
        SceneObject(name='glow_pad2', geom_type='box', pos='5.0 0 0.001',
                    size='1.5 1.5 0.001', rgba='0.0 0.12 0.08 0.3'),
        SceneObject(name='glow_pad3', geom_type='box', pos='10.0 0 0.001',
                    size='1.5 1.5 0.001', rgba='0.0 0.15 0.05 0.3'),
    ],
    extra_worldbody=(
        '<!-- Ambilight: upward green glow -->\n'
        '    <light name="ambi_green1" pos="0 0 0.05" dir="0 0 1" '
        'diffuse="0.05 0.18 0.06" specular="0 0 0" castshadow="false" '
        'attenuation="1 0.3 0.1"/>\n'
        '    <light name="ambi_green2" pos="5 0 0.05" dir="0 0 1" '
        'diffuse="0.04 0.15 0.05" specular="0 0 0" castshadow="false" '
        'attenuation="1 0.3 0.1"/>\n'
        '    <light name="ambi_cyan" pos="10 0 0.05" dir="0 0 1" '
        'diffuse="0.02 0.10 0.12" specular="0 0 0" castshadow="false" '
        'attenuation="1 0.3 0.1"/>\n'
        '    <!-- Blue sky accent light from above -->\n'
        '    <light name="sky_accent" directional="true" pos="0 0 5" dir="0 0 -1" '
        'diffuse="0.08 0.12 0.22" specular="0.05 0.08 0.15" castshadow="false"/>'
    ),
)

# --- Hills (Rampen) ---
SCENES['hills'] = Scene(
    name='hills',
    description='Huegel und Rampen — lernt Steigung',
    floor_friction='0.9 0.4 0.01',
    floor_texture_rgb1='0.35 0.55 0.25',
    floor_texture_rgb2='0.30 0.48 0.20',
    light_diffuse='0.55 0.55 0.45',
    haze_rgba='0.10 0.15 0.08 1',
    skybox_rgb1='0.50 0.65 0.45',
    skybox_rgb2='0.08 0.12 0.06',
    objects=[
        # Rampen als gekippte Boxen
        SceneObject(name='ramp1', geom_type='box', pos='1.5 0 0.05',
                    size='0.5 0.4 0.02', euler='0 -8 0',
                    rgba='0.4 0.55 0.3 1'),
        SceneObject(name='ramp2', geom_type='box', pos='3.0 0 0.12',
                    size='0.6 0.4 0.02', euler='0 -12 0',
                    rgba='0.38 0.52 0.28 1'),
        SceneObject(name='plateau', geom_type='box', pos='4.5 0 0.2',
                    size='0.4 0.5 0.2', rgba='0.42 0.5 0.32 1'),
    ],
)

# --- Night ---
SCENES['night'] = Scene(
    name='night',
    description='Nachtszene — reduzierte Sicht, Mondlicht',
    floor_friction='1.0 0.5 0.01',
    light_diffuse='0.12 0.15 0.25',
    light_specular='0.05 0.05 0.1',
    fill_light=False,
    haze_rgba='0.02 0.03 0.06 1',
    skybox_rgb1='0.05 0.07 0.15',
    skybox_rgb2='0.01 0.01 0.03',
)

# --- Windy ---
SCENES['windy'] = Scene(
    name='windy',
    description='Starker Seitenwind — Balance-Challenge',
    floor_friction='0.8 0.4 0.01',
    wind='3.0 1.0 0.0',
    light_diffuse='0.5 0.55 0.6',
    haze_rgba='0.15 0.18 0.22 1',
    skybox_rgb1='0.45 0.50 0.60',
    skybox_rgb2='0.10 0.12 0.18',
)

# --- Obstacle Course ---
SCENES['obstacle_course'] = Scene(
    name='obstacle_course',
    description='Hindernisparcours — Steine + Rampen + Enge',
    floor_friction='1.0 0.5 0.01',
    objects=[
        # Start-Bereich frei
        # Erste Hindernis-Reihe
        SceneObject(name='wall1', geom_type='box', pos='1.0 0.3 0.1',
                    size='0.02 0.15 0.1', rgba='0.6 0.3 0.2 1'),
        SceneObject(name='wall2', geom_type='box', pos='1.0 -0.3 0.1',
                    size='0.02 0.15 0.1', rgba='0.6 0.3 0.2 1'),
        # Luecke in der Mitte: 0.3m breit
        # Rampe
        SceneObject(name='ramp_obs', geom_type='box', pos='2.0 0 0.06',
                    size='0.3 0.3 0.02', euler='0 -10 0',
                    rgba='0.5 0.45 0.4 1'),
        # Enger Durchgang
        SceneObject(name='narrow_l', geom_type='box', pos='3.0 0.15 0.15',
                    size='0.3 0.02 0.15', rgba='0.55 0.35 0.25 1'),
        SceneObject(name='narrow_r', geom_type='box', pos='3.0 -0.15 0.15',
                    size='0.3 0.02 0.15', rgba='0.55 0.35 0.25 1'),
        # Stufe
        SceneObject(name='step1', geom_type='box', pos='4.0 0 0.05',
                    size='0.2 0.4 0.05', rgba='0.5 0.5 0.5 1'),
    ],
)
