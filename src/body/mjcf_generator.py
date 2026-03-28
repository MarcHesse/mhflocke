"""
MH-FLOCKE — MJCF Generator v0.4.1
========================================
Procedural MuJoCo XML generation from genome.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


# ================================================================
# FLOCKE PALETTE — Visual Identity MH-Flocke
# ================================================================

# Creature colors per template
FLOCKE_MATERIALS = {
    'biped': {
        'torso':  {'rgba': (1.0, 0.42, 0.21, 1.0), 'emission': 0.3},     # Warm Orange
        'head':   {'rgba': (1.0, 0.55, 0.35, 1.0), 'emission': 0.4},     # Heller
        'joint':  {'rgba': (0.02, 0.71, 0.83, 1.0), 'emission': 0.6},    # Cyan
        'limb':   {'rgba': (0.85, 0.35, 0.15, 1.0), 'emission': 0.2},    # Dunkler Orange
    },
    'quadruped': {
        'torso':  {'rgba': (0.96, 0.62, 0.04, 1.0), 'emission': 0.3},    # Gold
        'head':   {'rgba': (1.0, 0.72, 0.20, 1.0), 'emission': 0.4},
        'joint':  {'rgba': (0.02, 0.71, 0.83, 1.0), 'emission': 0.6},
        'limb':   {'rgba': (0.80, 0.52, 0.02, 1.0), 'emission': 0.2},
    },
    # ---- MOGLI / SYNPAW — Flocke's Zeichnung ----
    # Predominantly black. White accents: tail tip,
    # Front legs lower half (white with black spots),
    # small white spot on the head.
    'synpaw': {
        'torso':       {'rgba': (0.08, 0.08, 0.08, 1.0), 'emission': 0.1},  # Schwarz
        'head':        {'rgba': (0.08, 0.08, 0.08, 1.0), 'emission': 0.1},  # Schwarz (Fleck via texture)
        'head_spot':   {'rgba': (0.92, 0.92, 0.90, 1.0), 'emission': 0.3},  # Weißer Fleck oben
        'joint':       {'rgba': (0.12, 0.12, 0.12, 1.0), 'emission': 0.1},  # Dunkle Gelenke
        'limb':        {'rgba': (0.08, 0.08, 0.08, 1.0), 'emission': 0.1},  # Hinterbeine schwarz
        'front_upper': {'rgba': (0.08, 0.08, 0.08, 1.0), 'emission': 0.1},  # Vorderbein oben: schwarz
        'front_lower': {'rgba': (0.88, 0.88, 0.85, 1.0), 'emission': 0.2},  # Vorderbein unten: weiß
        'tail':        {'rgba': (0.08, 0.08, 0.08, 1.0), 'emission': 0.1},  # Schwanz schwarz
        'tail_tip':    {'rgba': (0.92, 0.92, 0.90, 1.0), 'emission': 0.3},  # Schwanzspitze weiß
        'eye':         {'rgba': (0.12, 0.12, 0.15, 1.0), 'emission': 0.0},  # Dunkle Augen
    },
    'worm': {
        'torso':  {'rgba': (0.08, 0.72, 0.65, 1.0), 'emission': 0.3},    # Teal
        'head':   {'rgba': (0.12, 0.82, 0.75, 1.0), 'emission': 0.4},
        'joint':  {'rgba': (0.02, 0.71, 0.83, 1.0), 'emission': 0.6},
        'limb':   {'rgba': (0.06, 0.60, 0.54, 1.0), 'emission': 0.2},
    },
    'random': {
        'torso':  {'rgba': (0.65, 0.55, 0.98, 1.0), 'emission': 0.3},    # Violett
        'head':   {'rgba': (0.75, 0.65, 1.0, 1.0), 'emission': 0.4},
        'joint':  {'rgba': (0.02, 0.71, 0.83, 1.0), 'emission': 0.6},
        'limb':   {'rgba': (0.55, 0.45, 0.85, 1.0), 'emission': 0.2},
    },
}

# Welt-Farben
FLOCKE_WORLD = {
    'deep_space':    (0.04, 0.055, 0.10, 1.0),    # #0A0E1A
    'ground_stone':  (0.22, 0.25, 0.32, 1.0),     # #374151
    'ground_grass':  (0.14, 0.28, 0.10, 1.0),     # Wiese grün, lebendig
    'metal_silver':  (0.58, 0.64, 0.72, 1.0),     # #94A3B8
    'obstacle_wood': (0.47, 0.21, 0.06, 1.0),     # #78350F
}

# Beleuchtungs-Presets
FLOCKE_LIGHTS = {
    'meadow': {  # Goldene Stunde
        'sun':  {'pos': '3 2 5', 'dir': '-0.3 -0.2 -1',
                 'diffuse': '0.95 0.85 0.65', 'specular': '0.4 0.35 0.25'},
        'fill': {'pos': '-2 0 4', 'dir': '0.2 0 -1',
                 'diffuse': '0.12 0.15 0.22', 'specular': '0 0 0'},
    },
    'lab': {  # Dunkles Labor
        'spot':    {'pos': '0 0 4', 'dir': '0 0.1 -1',
                    'diffuse': '0.9 0.85 0.7', 'specular': '0.5 0.5 0.5'},
        'ambient': {'pos': '0 0 6', 'dir': '0 0 -1',
                    'diffuse': '0.15 0.17 0.25', 'specular': '0 0 0'},
    },
    'arena': {  # Gleichmäßig für Multi-Agent
        'overhead': {'pos': '0 0 5', 'dir': '0 0 -1',
                     'diffuse': '0.7 0.7 0.75', 'specular': '0.3 0.3 0.3'},
        'fill':     {'pos': '3 3 3', 'dir': '-0.3 -0.3 -1',
                     'diffuse': '0.2 0.2 0.25', 'specular': '0 0 0'},
    },
}

# Legacy compatibility
CREATURE_COLORS = {
    'default':       FLOCKE_MATERIALS['worm']['torso']['rgba'],
    'generation_0':  (0.35, 0.55, 0.75, 1.0),
    'learning':      (0.25, 0.70, 0.55, 1.0),
    'best_fitness':  FLOCKE_MATERIALS['biped']['torso']['rgba'],
    'mutant':        FLOCKE_MATERIALS['random']['torso']['rgba'],
}


def get_creature_color(generation: int = 0, is_best: bool = False,
                       is_mutant: bool = False,
                       template: str = 'biped') -> tuple:
    """Farbe basierend auf Template und Status (Flocke-Palette)."""
    if is_best:
        # Champion: heller, mehr Emission
        mat = FLOCKE_MATERIALS.get(template, FLOCKE_MATERIALS['biped'])
        r, g, b, a = mat['torso']['rgba']
        return (min(1.0, r * 1.2), min(1.0, g * 1.2), min(1.0, b * 1.2), a)
    if is_mutant:
        return FLOCKE_MATERIALS['random']['torso']['rgba']
    # Template-Farbe
    mat = FLOCKE_MATERIALS.get(template, FLOCKE_MATERIALS['biped'])
    return mat['torso']['rgba']


def get_flocke_material(template: str, part: str) -> dict:
    """
    Gibt Flocke-Material für einen Kreatur-Teil zurück.

    Args:
        template: 'biped', 'quadruped', 'synpaw', 'mogli', 'worm', 'random'
        part: 'torso', 'head', 'head_spot', 'joint', 'limb',
              'front_upper', 'front_lower', 'tail', 'tail_tip', 'eye'

    Returns:
        {'rgba': (r,g,b,a), 'emission': float}
    """
    if template == 'mogli':
        template = 'synpaw'
    mats = FLOCKE_MATERIALS.get(template, FLOCKE_MATERIALS['biped'])
    return mats.get(part, mats.get('torso', {'rgba': (0.5, 0.5, 0.5, 1.0), 'emission': 0.1}))


# ================================================================
# MJCF GENERATOR
# ================================================================

class MJCFGenerator:
    """Konvertiert ein Genome in MuJoCo MJCF XML."""

    # Segment-Typ → MJCF Geometry Mapping
    GEOM_MAP = {
        'limb':    'capsule',
        'body':    'box',
        'torso':   'box',
        'head':    'sphere',
        'wheel':   'cylinder',
        'gripper': 'box',
    }

    @staticmethod
    def generate(genome, color: tuple = None,
                 name: str = "creature",
                 template: str = None) -> Tuple[str, str, str]:
        """
        Genome → MJCF XML Fragments.

        Args:
            genome: Genome Objekt
            color: (r, g, b, a) RGBA Basisfarbe (Legacy, überschrieben durch template)
            name: Name-Prefix für Bodies
            template: 'biped', 'quadruped', 'mogli', 'worm', 'random'
                      Wenn gesetzt, verwendet Flocke-Materialien per Segment-Typ.

        Returns:
            (body_xml, actuator_xml, sensor_xml) — drei XML-Strings
        """
        # Resolve template
        if template == 'mogli':
            template = 'synpaw'
        flocke_template = template  # None = Legacy-Modus

        if color is None:
            color = get_creature_color(genome.generation, template=template or 'biped')

        # Build Kinderliste: parent_id → [child_joints]
        children_map: Dict[int, list] = {}
        for j in genome.joints:
            children_map.setdefault(j.parent_id, []).append(j)

        # Find root segment (id=0 or first without parent)
        child_ids = {j.child_id for j in genome.joints}
        root_seg = None
        for s in genome.segments:
            if s.id not in child_ids:
                root_seg = s
                break
        if root_seg is None:
            root_seg = genome.segments[0]

        # Segment-Lookup
        seg_map = {s.id: s for s in genome.segments}

        # Joint-Lookup: child_id → Joint
        joint_map = {j.child_id: j for j in genome.joints}

        # Rekursiv Body-XML bauen
        body_xml = MJCFGenerator._build_body(
            root_seg, seg_map, children_map, joint_map,
            color, name, is_root=True, depth=0,
            flocke_template=flocke_template
        )

        # Actuator-XML: ein Motor pro Joint
        actuator_lines = []
        for j in genome.joints:
            joint_name = f"{name}_j{j.parent_id}_{j.child_id}"
            motor_name = f"{name}_m{j.parent_id}_{j.child_id}"
            gear = max(10, j.max_torque * 30)  # Torque → MuJoCo gear
            actuator_lines.append(
                f'    <motor name="{motor_name}" joint="{joint_name}" '
                f'gear="{gear:.0f}"/>'
            )
        actuator_xml = "\n".join(actuator_lines)

        # Sensor XML (optional, MuJoCo has its own sensors)
        sensor_lines = []
        # Gyro + Accelerometer am Root
        root_name = f"{name}_s{root_seg.id}"
        sensor_lines.append(f'    <gyro name="{name}_gyro" site="{root_name}_site"/>')
        sensor_lines.append(f'    <accelerometer name="{name}_accel" site="{root_name}_site"/>')
        sensor_xml = "\n".join(sensor_lines)

        return body_xml, actuator_xml, sensor_xml

    @staticmethod
    def generate_full(genome, gravity: float = -9.81,
                      color: tuple = None, name: str = "creature",
                      ground: bool = True,
                      template: str = None,
                      scene: str = None) -> str:
        """
        Komplettes MJCF XML mit worldbody, actuator, sensor.

        Args:
            scene: 'meadow', 'lab', 'arena' — Flocke-Beleuchtung + Boden
            template: 'mogli', 'biped', 'quadruped', etc. — Flocke-Materialien
        """
        body_xml, actuator_xml, sensor_xml = MJCFGenerator.generate(
            genome, color=color, name=name, template=template)

        ground_xml = ""
        if ground:
            if scene and scene in FLOCKE_LIGHTS:
                # Flocke-Szene: dunkler Hintergrund, spezielle Beleuchtung
                lights = FLOCKE_LIGHTS[scene]
                light_xml = ""
                for light_name, props in lights.items():
                    shadow = 'true' if light_name in ('sun', 'spot') else 'false'
                    light_xml += (
                        f'\n    <light name="{light_name}" pos="{props["pos"]}" '
                        f'dir="{props["dir"]}" diffuse="{props["diffuse"]}" '
                        f'specular="{props["specular"]}" castshadow="{shadow}"/>'
                    )

                # Boden per Szene
                if scene == 'meadow':
                    floor_rgba = '0.14 0.28 0.10 1'  # Grüne Wiese
                elif scene == 'arena':
                    floor_rgba = '0.18 0.20 0.25 1'  # Dunkles Grau-Blau
                else:
                    floor_rgba = '0.22 0.25 0.32 1'  # Ground Stone

                ground_xml = f"""{light_xml}
    <geom name="floor" type="plane" size="20 20 0.1"
          rgba="{floor_rgba}" friction="1.0 0.5 0.01"/>"""
            else:
                # Legacy: helle Standard-Szene
                ground_xml = """
    <light pos="0 0 3" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="20 20 0.1"
          rgba="0.9 0.9 0.92 1" friction="1.0 0.5 0.01"/>"""

        # Haze/fog for Flocke scenes
        visual_xml = ""
        if scene:
            visual_xml = """
  <visual>
    <rgba haze="0.04 0.055 0.10 1"/>
    <quality shadowsize="4096"/>
    <map znear="0.01"/>
  </visual>"""

        xml = f"""<?xml version="1.0" ?>
<mujoco model="{name}">
  <option timestep="0.002" gravity="0 0 {gravity}" integrator="RK4"/>{visual_xml}

  <default>
    <geom condim="3" friction="1.0 0.5 0.01"/>
    <joint damping="0.5" armature="0.01"/>
    <motor ctrlrange="-1 1" ctrllimited="true"/>
  </default>

  <worldbody>{ground_xml}
    {body_xml}
  </worldbody>

  <actuator>
{actuator_xml}
  </actuator>

  <sensor>
{sensor_xml}
  </sensor>
</mujoco>"""
        return xml

    # ================================================================
    # REKURSIVER BODY-BUILDER
    # ================================================================

    @staticmethod
    def _build_body(segment, seg_map, children_map, joint_map,
                    color, name, is_root, depth,
                    flocke_template: str = None) -> str:
        """Rekursiv: Segment → <body> XML mit Kindern."""
        seg_name = f"{name}_s{segment.id}"
        indent = "    " * (depth + 2)

        # --- Flocke-Farbe per Segment-Typ ---
        resolved_template = flocke_template
        if resolved_template == 'mogli':
            resolved_template = 'synpaw'

        if flocke_template:
            mat = get_flocke_material(flocke_template, segment.segment_type)
            seg_color = mat['rgba']
        else:
            # Legacy: einheitliche Farbe mit Tiefe-Dimming
            dim = max(0.5, 1.0 - depth * 0.1)
            seg_color = (color[0] * dim, color[1] * dim, color[2] * dim, color[3])

        # --- Is this segment an eye? ---
        is_eye = (segment.segment_type == 'head' and segment.radius < 0.04
                  and segment.length < 0.05)

        # --- Is this segment a tail part? ---
        # Heuristik: Limb direkt am Torso mit kleinem Radius + keine eigenen Limb-Kinder
        parent_joint = joint_map.get(segment.id)
        parent_seg = seg_map.get(parent_joint.parent_id) if parent_joint else None
        is_tail = False
        if (parent_seg and parent_seg.segment_type == 'torso'
                and segment.segment_type == 'limb' and segment.radius < 0.05):
            # Check ob Segment nach dem Schwanz-Position kommt (hohe ID = hinten)
            limb_children = [cj for cj in children_map.get(parent_seg.id, [])
                             if seg_map.get(cj.child_id) and
                             seg_map[cj.child_id].segment_type == 'limb']
            if limb_children:
                max_limb_id = max(cj.child_id for cj in limb_children)
                # Schwanz = letztes Limb-Kind mit kleinem Radius
                if segment.id >= max_limb_id - 1 and segment.radius < 0.05:
                    is_tail = True
        # Auch Kinder eines Schwanz-Segments sind Schwanz
        if parent_seg and parent_seg.segment_type == 'limb' and parent_seg.radius < 0.05:
            parent_parent_joint = joint_map.get(parent_seg.id)
            if parent_parent_joint:
                pp_seg = seg_map.get(parent_parent_joint.parent_id)
                if pp_seg and pp_seg.segment_type == 'torso':
                    is_tail = True

        # --- Synpaw/Mogli: Segment-spezifische Farben ---
        is_sub_limb = (not is_root and parent_seg
                       and parent_seg.segment_type == 'limb'
                       and parent_seg.radius >= 0.05)

        if resolved_template == 'synpaw' and flocke_template:
            if is_eye:
                seg_color = get_flocke_material('synpaw', 'eye')['rgba']
            elif is_tail:
                if (parent_seg and parent_seg.segment_type == 'limb'
                        and parent_seg.radius < 0.05):
                    seg_color = get_flocke_material('synpaw', 'tail_tip')['rgba']
                else:
                    seg_color = get_flocke_material('synpaw', 'tail')['rgba']
            elif segment.segment_type == 'limb' and not is_tail:
                if parent_seg and parent_seg.segment_type == 'torso':
                    limb_cs = [cj for cj in children_map.get(parent_seg.id, [])
                               if seg_map.get(cj.child_id) and
                               seg_map[cj.child_id].segment_type == 'limb'
                               and seg_map[cj.child_id].radius >= 0.05]
                    li = next((i for i, cj in enumerate(limb_cs)
                               if cj.child_id == segment.id), 0)
                    if li < 2:  # vorne
                        seg_color = get_flocke_material('synpaw', 'front_upper')['rgba']
                elif is_sub_limb:
                    pp_j = joint_map.get(parent_seg.id)
                    if pp_j:
                        pp_s = seg_map.get(pp_j.parent_id)
                        if pp_s and pp_s.segment_type == 'torso':
                            pp_ls = [cj for cj in children_map.get(pp_s.id, [])
                                     if seg_map.get(cj.child_id) and
                                     seg_map[cj.child_id].segment_type == 'limb'
                                     and seg_map[cj.child_id].radius >= 0.05]
                            pp_i = next((i for i, cj in enumerate(pp_ls)
                                         if cj.child_id == parent_seg.id), 0)
                            if pp_i < 2:
                                seg_color = get_flocke_material('synpaw', 'front_lower')['rgba']

        # Position relativ zum Parent
        if is_root:
            # Root schwebt über dem Boden — hoch genug für Beine
            n_children = len(children_map.get(segment.id, []))
            leg_length = 0.0
            for cj in children_map.get(segment.id, []):
                child = seg_map.get(cj.child_id)
                if child and child.segment_type == 'limb' and child.radius >= 0.05:
                    leg_length = max(leg_length, child.length)
            height = max(0.8, leg_length * 1.5 + 0.3)
            pos_str = f'0 0 {height:.3f}'
        elif is_eye:
            # Augen: seitlich am Kopf positioniert
            # Bestimme ob linkes oder rechtes Auge
            siblings = children_map.get(parent_joint.parent_id, [])
            eye_siblings = [cj for cj in siblings
                           if seg_map.get(cj.child_id) and
                           seg_map[cj.child_id].segment_type == 'head' and
                           seg_map[cj.child_id].radius < 0.04]
            eye_idx = next((i for i, cj in enumerate(eye_siblings)
                           if cj.child_id == segment.id), 0)
            # Links = +Y, Rechts = -Y, leicht vorne (+X), leicht oben (+Z)
            side = 0.06 if eye_idx == 0 else -0.06
            pos_str = f'0.06 {side:.3f} 0.03'
            # Augen sind dunkel mit hellem Punkt
            seg_color = (0.12, 0.12, 0.15, 1.0)  # Dunkle Pupille
        elif is_tail:
            if parent_seg and parent_seg.segment_type == 'torso':
                # Schwanz-Basis: hinten am Torso (-X), leicht oben (+Z)
                pos_str = f'-{parent_seg.length * 0.6:.3f} 0 {parent_seg.radius * 0.3:.3f}'
            else:
                # Schwanz-Spitze: verlängert nach hinten
                pos_str = f'-{segment.length:.3f} 0 0.02'
        else:
            # Standard-Positionierung
            if parent_joint:
                parent_is_root = parent_seg and parent_seg.id not in {
                    j.child_id for j in joint_map.values() if j is not None}

                children_of_parent = children_map.get(parent_joint.parent_id, [])

                if parent_is_root and segment.segment_type == 'head':
                    # Kopf: vorne am Torso (Quadruped) oder oben (Biped)
                    if parent_seg:
                        # Vorne = +X für Quadruped, oben = +Z für Biped
                        has_4_legs = sum(1 for cj in children_of_parent
                                        if seg_map.get(cj.child_id) and
                                        seg_map[cj.child_id].segment_type == 'limb'
                                        and seg_map[cj.child_id].radius >= 0.05) >= 4
                        if has_4_legs:
                            # Quadruped: Kopf vorne
                            pos_str = f'{parent_seg.length * 0.5 + segment.radius:.3f} 0 {parent_seg.radius * 0.3:.3f}'
                        else:
                            # Biped: Kopf oben
                            offset_z = (parent_seg.radius * 0.6 + segment.radius)
                            pos_str = f'0 0 {offset_z:.3f}'
                    else:
                        pos_str = f'0 0 0.2'
                elif parent_is_root and segment.segment_type == 'limb' and not is_tail:
                    # Beine: bei Quadruped vorne/hinten + seitlich
                    limb_children = [cj for cj in children_of_parent
                                     if seg_map.get(cj.child_id) and
                                     seg_map[cj.child_id].segment_type == 'limb'
                                     and seg_map[cj.child_id].radius >= 0.05]
                    limb_idx = next((i for i, cj in enumerate(limb_children)
                                    if cj.child_id == segment.id), 0)
                    n_limbs = len(limb_children)

                    if n_limbs == 4 and parent_seg:
                        # Quadruped: 2 vorne, 2 hinten
                        spread_y = max(0.12, parent_seg.radius * 0.8)
                        spread_x = max(0.15, parent_seg.length * 0.35)
                        positions = [
                            (spread_x, spread_y),    # Vorne links
                            (spread_x, -spread_y),   # Vorne rechts
                            (-spread_x, spread_y),   # Hinten links
                            (-spread_x, -spread_y),  # Hinten rechts
                        ]
                        px, py = positions[limb_idx] if limb_idx < 4 else (0, 0)
                        offset_z = -(parent_seg.radius * 0.5)
                        pos_str = f'{px:.3f} {py:.3f} {offset_z:.3f}'
                    else:
                        # Biped/andere: seitlich
                        spread = max(0.15, (parent_seg.length / 2) if parent_seg else 0.2)
                        if n_limbs > 1:
                            offset_y = -spread + limb_idx * (2 * spread / (n_limbs - 1))
                        else:
                            offset_y = 0
                        offset_z = -(parent_seg.radius * 0.5 if parent_seg else 0.1)
                        pos_str = f'0 {offset_y:.3f} {offset_z:.3f}'
                else:
                    # Sub-Limb: nach unten verlängern
                    if parent_seg:
                        offset_z = -(parent_seg.length / 2 + segment.length / 4)
                    else:
                        offset_z = -0.2
                    pos_str = f'0 0 {offset_z:.3f}'
            else:
                pos_str = '0 0 -0.2'

        lines = [f'{indent}<body name="{seg_name}" pos="{pos_str}">']

        # Freejoint nur am Root
        if is_root:
            lines.append(f'{indent}  <freejoint name="{name}_root"/>')

        # Gelenk (wenn nicht Root)
        if not is_root and segment.id in joint_map:
            joint = joint_map[segment.id]
            joint_name = f"{name}_j{joint.parent_id}_{joint.child_id}"
            axis = joint.axis
            axis_str = f'{axis[0]:.1f} {axis[1]:.1f} {axis[2]:.1f}'
            range_str = f'{joint.angle_min:.2f} {joint.angle_max:.2f}'

            if joint.joint_type == 'hinge':
                lines.append(
                    f'{indent}  <joint name="{joint_name}" type="hinge" '
                    f'axis="{axis_str}" range="{range_str}" limited="true"/>'
                )
            elif joint.joint_type == 'ball':
                # Ball joint: 3 DOF — modelliere als 2 hinge joints
                lines.append(
                    f'{indent}  <joint name="{joint_name}_x" type="hinge" '
                    f'axis="1 0 0" range="{range_str}" limited="true"/>'
                )
                lines.append(
                    f'{indent}  <joint name="{joint_name}_y" type="hinge" '
                    f'axis="0 1 0" range="{range_str}" limited="true"/>'
                )

        # Inertia — MuJoCo braucht mass > mjMINVAL (1e-15) und Inertia > 0
        mass = max(0.05, segment.mass)  # Mindestens 50g
        r = max(0.02, segment.radius)
        h = max(0.05, segment.length)
        # Diag-Inertia für Zylinder: Ixx=Iyy=m(3r²+h²)/12, Izz=mr²/2
        ixx = max(1e-4, mass * (3 * r**2 + h**2) / 12)
        izz = max(1e-4, mass * r**2 / 2)
        lines.append(
            f'{indent}  <inertial pos="0 0 0" mass="{mass:.3f}" '
            f'diaginertia="{ixx:.4f} {ixx:.4f} {izz:.4f}"/>'
        )

        # Geometrie
        geom_type = MJCFGenerator.GEOM_MAP.get(segment.segment_type, 'capsule')
        rgba_str = f'{seg_color[0]:.2f} {seg_color[1]:.2f} {seg_color[2]:.2f} {seg_color[3]:.2f}'

        # Schwanz rendert horizontal (fromto entlang -X statt -Z)
        tail_geom = is_tail

        if geom_type == 'box':
            # Box: size = half-extents [x, y, z]
            # Torso: breit(Y), flach(Z), mittel(X)
            hx = max(0.04, r * 0.8)
            hy = max(0.04, h / 2)
            hz = max(0.04, r * 0.6)
            lines.append(
                f'{indent}  <geom name="{seg_name}_geom" type="box" '
                f'size="{hx:.3f} {hy:.3f} {hz:.3f}" rgba="{rgba_str}"/>'
            )
        elif geom_type == 'sphere':
            lines.append(
                f'{indent}  <geom name="{seg_name}_geom" type="sphere" '
                f'size="{r:.3f}" rgba="{rgba_str}"/>'
            )
            # Synpaw head spot: small white sphere on top
            if (resolved_template == 'synpaw' and segment.segment_type == 'head'
                    and not is_eye and r >= 0.04):
                spot = get_flocke_material('synpaw', 'head_spot')['rgba']
                spot_str = f'{spot[0]:.2f} {spot[1]:.2f} {spot[2]:.2f} {spot[3]:.2f}'
                spot_r = r * 0.35
                lines.append(
                    f'{indent}  <geom name="{seg_name}_spot" type="sphere" '
                    f'pos="0 0 {r * 0.7:.3f}" size="{spot_r:.3f}" rgba="{spot_str}"/>'
                )
        elif geom_type == 'capsule':
            if tail_geom:
                # Schwanz: horizontal nach hinten (-X)
                lines.append(
                    f'{indent}  <geom name="{seg_name}_geom" type="capsule" '
                    f'fromto="0 0 0 -{segment.length:.3f} 0 0" '
                    f'size="{r:.3f}" rgba="{rgba_str}"/>'
                )
            else:
                # Gliedmaßen vertikal (Z-Achse, nach unten)
                lines.append(
                    f'{indent}  <geom name="{seg_name}_geom" type="capsule" '
                    f'fromto="0 0 0 0 0 -{segment.length:.3f}" '
                    f'size="{r:.3f}" rgba="{rgba_str}"/>'
                )
        elif geom_type == 'cylinder':
            lines.append(
                f'{indent}  <geom name="{seg_name}_geom" type="cylinder" '
                f'size="{r:.3f} {h/2:.3f}" rgba="{rgba_str}"/>'
            )
        else:
            lines.append(
                f'{indent}  <geom name="{seg_name}_geom" type="capsule" '
                f'fromto="0 0 0 0 0 -{segment.length:.3f}" '
                f'size="{r:.3f}" rgba="{rgba_str}"/>'
            )

        # Site für Sensoren (am Root)
        if is_root:
            lines.append(
                f'{indent}  <site name="{seg_name}_site" pos="0 0 0" size="0.01"/>'
            )

        # Rekursiv: Kinder
        child_joints = children_map.get(segment.id, [])
        for child_joint in child_joints:
            child_seg = seg_map.get(child_joint.child_id)
            if child_seg:
                child_xml = MJCFGenerator._build_body(
                    child_seg, seg_map, children_map, joint_map,
                    color, name, is_root=False, depth=depth + 1,
                    flocke_template=flocke_template
                )
                lines.append(child_xml)

        lines.append(f'{indent}</body>')
        return "\n".join(lines)

    # ================================================================
    # UTILITIES
    # ================================================================

    @staticmethod
    def count_actuators(genome) -> int:
        """Wie viele Actuatoren braucht das Genome?"""
        count = 0
        for j in genome.joints:
            if j.joint_type == 'hinge':
                count += 1
            elif j.joint_type == 'ball':
                count += 2  # 2 hinge joints pro ball
        return count

    @staticmethod
    def get_joint_names(genome, name: str = "creature") -> list:
        """Liste aller Joint-Namen in Reihenfolge."""
        names = []
        for j in genome.joints:
            joint_name = f"{name}_j{j.parent_id}_{j.child_id}"
            if j.joint_type == 'hinge':
                names.append(joint_name)
            elif j.joint_type == 'ball':
                names.append(f"{joint_name}_x")
                names.append(f"{joint_name}_y")
        return names
