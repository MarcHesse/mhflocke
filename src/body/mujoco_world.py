"""
MH-FLOCKE — MuJoCo World v0.4.1
========================================
Physics world management and simulation stepping.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import tempfile
import os

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False


# ================================================================
# DEFAULT SCENE XML
# ================================================================

_BASE_XML = """<?xml version="1.0" ?>
<mujoco model="genesis_world">
  <option timestep="{timestep}" gravity="0 0 {gravity}" integrator="RK4"/>

  <default>
    <geom condim="3" friction="1.0 0.5 0.01" rgba="0.6 0.6 0.6 1"/>
    <joint damping="0.5" armature="0.01"/>
    <motor ctrlrange="-1 1" ctrllimited="true"/>
  </default>

  <visual>
    <headlight diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20" offwidth="1920" offheight="1080"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.9 0.9 0.92" rgb2="0.82 0.82 0.85" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    {ground}
    {bodies}
  </worldbody>

  <actuator>
    {actuators}
  </actuator>

  <sensor>
    {sensors}
  </sensor>
</mujoco>
"""

_GROUND_XML = """
    <light pos="0 0 3" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="{size} {size} 0.1"
          material="groundplane" friction="{friction} 0.5 0.01"/>
"""


@dataclass
class BodyProxy:
    """Lightweight proxy mimicking RigidBody for compatibility."""
    id: int
    position: np.ndarray
    rotation: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    mass: float
    length: float
    radius: float
    is_static: bool = False


class MuJoCoWorld:
    """MuJoCo-basierte Physik-Welt, kompatibel mit PhysicsWorld Interface."""

    def __init__(self, size: float = 20.0, gravity: float = -9.81,
                 dt: float = 0.002, render: bool = False,
                 snn_dt: float = 0.01):
        """
        Args:
            size: Weltgröße (für Boden-Plane)
            gravity: Schwerkraft (default: Erde, negativ = nach unten)
            dt: MuJoCo Zeitschritt (default: 2ms)
            render: True = offscreen Renderer aktivieren
            snn_dt: SNN-Zeitschritt in Sekunden (default: 10ms)
        """
        if not MUJOCO_AVAILABLE:
            raise ImportError("mujoco not installed. Run: pip install mujoco")

        self.size = size
        self.gravity = gravity
        self.dt = dt
        self.snn_dt = snn_dt
        self.render = render
        # NOTE: substeps=1 ensures CPG timing matches evolution.
        # The CPG advances phase by dt each step; with substeps>1 the physics
        # would run ahead of the CPG rhythm (Issue #31).
        self._substeps = 1

        # Deferred build: sammle XML-Teile, baue Modell bei erstem step()
        self._ground_xml = ""
        self._body_xmls: List[str] = []
        self._actuator_xmls: List[str] = []
        self._sensor_xmls: List[str] = []
        self._built = False

        # MuJoCo state (nach build)
        self._model = None
        self._data = None
        self._renderer = None

        # Tracking
        self._body_registry: Dict[str, dict] = {}  # name → metadata
        self._creature_names: List[str] = []
        self._object_count = 0
        self._contacts: List[Tuple[int, int, float]] = []
        self.step_count = 0

    # ================================================================
    # BUILD
    # ================================================================

    def _build(self):
        """Baue MuJoCo Modell aus gesammelten XML-Teilen."""
        xml = _BASE_XML.format(
            timestep=self.dt,
            gravity=self.gravity,
            ground=self._ground_xml,
            bodies="\n".join(self._body_xmls),
            actuators="\n".join(self._actuator_xmls),
            sensors="\n".join(self._sensor_xmls),
        )

        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)

        if self.render:
            self._renderer = mujoco.Renderer(self._model, 720, 1280)

        self._built = True

    def _ensure_built(self):
        if not self._built:
            self._build()

    def load_from_xml_path(self, xml_path: str):
        """
        Lade komplettes MJCF von Datei (inkl. Meshes, Texturen).
        
        Uses MuJoCo's native from_xml_path which correctly resolves
        <include>, meshdir, texturedir relative to the XML file location.
        """
        import os
        xml_path = os.path.abspath(xml_path)
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)
        # Reset to keyframe if available (standing pose)
        if self._model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self._model, self._data, 0)
        if self.render:
            self._renderer = mujoco.Renderer(self._model, 720, 1280)
        self._built = True

    def load_from_xml_string(self, xml_string: str, assets: dict = None):
        """
        Lade komplettes MJCF aus String.
        
        Args:
            xml_string: Vollständiges MJCF XML
            assets: Dict {filename: bytes} für eingebettete Meshes/Texturen
        """
        if assets:
            self._model = mujoco.MjModel.from_xml_string(xml_string, assets)
        else:
            self._model = mujoco.MjModel.from_xml_string(xml_string)
        self._data = mujoco.MjData(self._model)
        if self.render:
            self._renderer = mujoco.Renderer(self._model, 720, 1280)
        self._built = True

    # ================================================================
    # WELT AUFBAUEN
    # ================================================================

    def add_ground(self, friction: float = 1.0, color: list = None):
        """Boden-Plane hinzufügen."""
        self._ground_xml = _GROUND_XML.format(
            size=self.size, friction=friction)
        self._built = False  # Force rebuild
        return 0

    def load_creature_xml(self, xml_body: str, xml_actuators: str = "",
                           xml_sensors: str = "",
                           name: str = None) -> str:
        """
        Lade Kreatur aus MJCF XML Fragments.

        Args:
            xml_body: <body>...</body> XML Fragment
            xml_actuators: <motor .../> Zeilen
            xml_sensors: <sensor .../> Zeilen
            name: Kreatur-Name (auto-generiert wenn None)

        Returns:
            Kreatur-Name (für spätere Referenz)
        """
        if name is None:
            name = f"creature_{len(self._creature_names)}"

        self._body_xmls.append(xml_body)
        if xml_actuators:
            self._actuator_xmls.append(xml_actuators)
        if xml_sensors:
            self._sensor_xmls.append(xml_sensors)

        self._creature_names.append(name)
        self._body_registry[name] = {
            'type': 'creature',
            'index': len(self._creature_names) - 1,
        }
        self._built = False
        return name

    def load_creature_from_string(self, full_xml: str) -> str:
        """
        Lade Kreatur aus komplettem MJCF XML (parst body/actuator/sensor).
        Für Kompatibilität mit URDFGenerator der jetzt MJCF erzeugt.

        Returns:
            Kreatur-Name
        """
        # Parse: extrahiere worldbody-Kinder, actuator-Kinder, sensor-Kinder
        import xml.etree.ElementTree as ET
        root = ET.fromstring(full_xml)

        name = root.get('model', f'creature_{len(self._creature_names)}')

        # Bodies aus worldbody (ohne floor/light)
        wb = root.find('worldbody')
        if wb is not None:
            for child in wb:
                if child.tag == 'body':
                    self._body_xmls.append(ET.tostring(child, encoding='unicode'))

        # Actuators
        act = root.find('actuator')
        if act is not None:
            for child in act:
                self._actuator_xmls.append(ET.tostring(child, encoding='unicode'))

        # Sensors
        sens = root.find('sensor')
        if sens is not None:
            for child in sens:
                self._sensor_xmls.append(ET.tostring(child, encoding='unicode'))

        self._creature_names.append(name)
        self._body_registry[name] = {'type': 'creature'}
        self._built = False
        return name

    def add_object(self, shape: str, position: list, mass: float,
                   color: list = None, size: list = None) -> str:
        """
        Objekt hinzufügen (box/sphere/cylinder).

        Returns:
            Objekt-Name
        """
        if size is None:
            size = [0.2, 0.2, 0.2]
        if color is None:
            color = [0.6, 0.6, 0.6, 1.0]

        name = f"obj_{self._object_count}"
        self._object_count += 1

        rgba = " ".join(f"{c:.2f}" for c in color)

        if shape == 'box':
            half = " ".join(f"{s/2:.3f}" for s in size[:3])
            geom = f'<geom type="box" size="{half}" rgba="{rgba}"/>'
        elif shape == 'sphere':
            r = size[0] if isinstance(size, list) else size
            geom = f'<geom type="sphere" size="{r:.3f}" rgba="{rgba}"/>'
        elif shape == 'cylinder':
            r = size[0] if len(size) > 0 else 0.1
            h = (size[1] / 2) if len(size) > 1 else 0.15
            geom = f'<geom type="cylinder" size="{r:.3f} {h:.3f}" rgba="{rgba}"/>'
        else:
            raise ValueError(f"Unknown shape: {shape}")

        pos = " ".join(f"{p:.3f}" for p in position)
        body_xml = f'<body name="{name}" pos="{pos}">'
        if mass > 0:
            body_xml += f'\n      <freejoint name="{name}_free"/>'
        body_xml += f'\n      <inertial pos="0 0 0" mass="{mass:.3f}" diaginertia="0.001 0.001 0.001"/>'
        body_xml += f'\n      {geom}'
        body_xml += f'\n    </body>'

        self._body_xmls.append(body_xml)
        self._body_registry[name] = {
            'type': 'object',
            'shape': shape,
            'mass': mass,
        }
        self._built = False
        return name

    # ================================================================
    # SIMULATION
    # ================================================================

    def step(self):
        """Ein SNN-Zeitschritt (= mehrere MuJoCo Substeps)."""
        self._ensure_built()
        self._contacts = []

        for _ in range(self._substeps):
            mujoco.mj_step(self._model, self._data)

        self._collect_contacts()
        self.step_count += 1

    def step_single(self):
        """Ein einzelner MuJoCo-Zeitschritt (dt)."""
        self._ensure_built()
        mujoco.mj_step(self._model, self._data)
        self.step_count += 1

    def _collect_contacts(self):
        """Sammle alle aktiven Kontaktpunkte."""
        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            # Kontaktkraft aus efc_force (vereinfacht: Normalkomponente)
            # MuJoCo stores forces in efc_force, here we approximate
            force = abs(contact.dist) * 1000  # Rough approximation
            # Versuche echte Kontaktkraft zu bekommen
            c_force = np.zeros(6)
            mujoco.mj_contactForce(self._model, self._data, i, c_force)
            normal_force = abs(c_force[0])

            body1 = self._model.geom_bodyid[geom1]
            body2 = self._model.geom_bodyid[geom2]
            self._contacts.append((int(body1), int(body2), float(normal_force)))

    # ================================================================
    # ABFRAGEN — BODY STATE
    # ================================================================

    def get_body_state(self, body_name: str) -> dict:
        """Position, Rotation, Velocity eines benannten Bodies."""
        self._ensure_built()
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found")

        pos = self._data.xpos[body_id].copy()
        # Rotation: xmat ist 3x3 flatten → Quaternion
        mat = self._data.xmat[body_id].reshape(3, 3)
        quat = self._mat_to_quat(mat)

        # Velocity: cvel ist [angular, linear] = 6D
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self._model, self._data,
                                  mujoco.mjtObj.mjOBJ_BODY, body_id, vel, 0)
        # vel = [wx, wy, wz, vx, vy, vz]

        return {
            'position': pos,
            'rotation': quat,
            'velocity': vel[3:6].copy(),
            'angular_velocity': vel[0:3].copy(),
        }

    def get_body_state_by_id(self, body_id: int) -> dict:
        """Body state per numerischer ID."""
        self._ensure_built()
        pos = self._data.xpos[body_id].copy()
        mat = self._data.xmat[body_id].reshape(3, 3)
        quat = self._mat_to_quat(mat)
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self._model, self._data,
                                  mujoco.mjtObj.mjOBJ_BODY, body_id, vel, 0)
        return {
            'position': pos,
            'rotation': quat,
            'velocity': vel[3:6].copy(),
            'angular_velocity': vel[0:3].copy(),
        }

    @staticmethod
    def _mat_to_quat(mat):
        """3x3 Rotationsmatrix → Quaternion [w, x, y, z]."""
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        return quat

    # ================================================================
    # GELENKE
    # ================================================================

    def get_joint_angles(self, creature_name: str = None) -> np.ndarray:
        """Alle Gelenkwinkel (hinge joints) als Array."""
        self._ensure_built()
        # Collect all hinge joints
        angles = []
        for i in range(self._model.njnt):
            if self._model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
                addr = self._model.jnt_qposadr[i]
                angles.append(self._data.qpos[addr])
        return np.array(angles)

    def get_joint_velocities(self, creature_name: str = None) -> np.ndarray:
        """Alle Gelenkgeschwindigkeiten als Array."""
        self._ensure_built()
        vels = []
        for i in range(self._model.njnt):
            if self._model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
                addr = self._model.jnt_dofadr[i]
                vels.append(self._data.qvel[addr])
        return np.array(vels)

    def get_joint_states(self, creature_name: str = None) -> list:
        """Detaillierte Gelenkzustände."""
        self._ensure_built()
        states = []
        for i in range(self._model.njnt):
            if self._model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
                name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
                addr_q = self._model.jnt_qposadr[i]
                addr_v = self._model.jnt_dofadr[i]
                states.append({
                    'joint_index': i,
                    'name': name or f'joint_{i}',
                    'angle': float(self._data.qpos[addr_q]),
                    'velocity': float(self._data.qvel[addr_v]),
                    'lower_limit': float(self._model.jnt_range[i][0]),
                    'upper_limit': float(self._model.jnt_range[i][1]),
                })
        return states

    # ================================================================
    # MOTORSTEUERUNG
    # ================================================================

    def set_controls(self, controls: np.ndarray):
        """Setze Actuator-Controls direkt (normalisiert -1..1)."""
        self._ensure_built()
        n = min(len(controls), self._model.nu)
        self._data.ctrl[:n] = controls[:n]

    def set_joint_torques(self, torques: list):
        """Setze Drehmomente auf Actuatoren."""
        self._ensure_built()
        n = min(len(torques), self._model.nu)
        for i in range(n):
            self._data.ctrl[i] = np.clip(torques[i], -1.0, 1.0)

    # ================================================================
    # SENSOR-DATEN
    # ================================================================

    def get_sensor_data(self, creature_name: str = None) -> dict:
        """Alle Sensor-Daten."""
        self._ensure_built()

        # Find base body of creature
        root_body = 1  # Default: erstes freies Body (0 = world)
        if creature_name:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, creature_name)
            if bid >= 0:
                root_body = bid

        pos = self._data.xpos[root_body].copy()
        mat = self._data.xmat[root_body].reshape(3, 3)
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self._model, self._data,
                                  mujoco.mjtObj.mjOBJ_BODY, root_body, vel, 0)

        # Euler aus Rotationsmatrix
        quat = self._mat_to_quat(mat)
        euler = self._quat_to_euler(quat)

        # Bodenkontakte
        ground_contacts = []
        for a, b, force in self._contacts:
            if a == 0 or b == 0:  # World body = 0 = Boden
                ground_contacts.append(force)

        return {
            'position': pos,
            'velocity': vel[3:6].copy(),
            'angular_velocity': vel[0:3].copy(),
            'orientation_euler': euler,
            'orientation_quat': quat,
            'joint_angles': self.get_joint_angles(creature_name),
            'joint_velocities': self.get_joint_velocities(creature_name),
            'ground_contacts': ground_contacts,
            'is_grounded': len(ground_contacts) > 0,
            'height': float(pos[2]),
            'forward_velocity': float(vel[3]),  # X-Richtung
            'upright': float(mat[2, 2]),  # Z-Komponente der Up-Achse (1.0 = aufrecht)
        }

    @staticmethod
    def _quat_to_euler(quat):
        """Quaternion [w,x,y,z] → Euler [roll, pitch, yaw]."""
        w, x, y, z = quat
        sinr = 2 * (w * x + y * z)
        cosr = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr, cosr)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        siny = 2 * (w * z + x * y)
        cosy = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny, cosy)
        return np.array([roll, pitch, yaw])

    # ================================================================
    # KONTAKTE
    # ================================================================

    def get_contacts(self) -> List[Tuple[int, int, float]]:
        """Alle aktiven Kontakte: [(body_a, body_b, force), ...]"""
        return self._contacts

    # ================================================================
    # KAMERA / RENDERING
    # ================================================================

    def get_camera_image(self, width: int = 1280, height: int = 720,
                         camera_name: str = None,
                         target: list = None, distance: float = 3.0,
                         azimuth: float = 120, elevation: float = -20) -> np.ndarray:
        """
        Rendere ein Frame als RGB numpy array.

        Returns:
            RGB Frame [height, width, 3] uint8
        """
        self._ensure_built()

        renderer = mujoco.Renderer(self._model, height, width)

        if camera_name:
            cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if cam_id >= 0:
                renderer.update_scene(self._data, camera=cam_id)
            else:
                renderer.update_scene(self._data)
        else:
            # Free camera mit benutzerdefinierten Parametern
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            if target is not None:
                cam.lookat[:] = target
            else:
                cam.lookat[:] = [0, 0, 0.5]
            cam.distance = distance
            cam.azimuth = azimuth
            cam.elevation = elevation
            renderer.update_scene(self._data, camera=cam)

        frame = renderer.render()
        renderer.close()
        return frame

    # ================================================================
    # COMPATIBILITY WITH PhysicsWorld
    # ================================================================

    @property
    def bodies(self) -> Dict[int, BodyProxy]:
        """Dict body_id → BodyProxy (kompatibel mit PhysicsWorld.bodies)."""
        self._ensure_built()
        result = {}
        for i in range(self._model.nbody):
            if i == 0:
                continue  # Skip world body
            pos = self._data.xpos[i].copy()
            mat = self._data.xmat[i].reshape(3, 3)
            quat = self._mat_to_quat(mat)
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(self._model, self._data,
                                      mujoco.mjtObj.mjOBJ_BODY, i, vel, 0)
            mass = self._model.body_mass[i]
            result[i] = BodyProxy(
                id=i,
                position=pos,
                rotation=quat,
                velocity=vel[3:6].copy(),
                angular_velocity=vel[0:3].copy(),
                mass=float(mass),
                length=0.3,
                radius=0.15,
                is_static=(mass == 0),
            )
        return result

    def get_center_of_mass(self, body_ids: List[int] = None) -> np.ndarray:
        """Schwerpunkt."""
        self._ensure_built()
        if body_ids is None:
            # Gesamt-CoM
            return self._data.subtree_com[0].copy()
        total_mass = 0.0
        com = np.zeros(3)
        for bid in body_ids:
            mass = self._model.body_mass[bid]
            com += self._data.xpos[bid] * mass
            total_mass += mass
        if total_mass > 1e-8:
            return com / total_mass
        return com

    # ================================================================
    # INFO
    # ================================================================

    @property
    def n_actuators(self) -> int:
        self._ensure_built()
        return self._model.nu

    @property
    def n_bodies(self) -> int:
        self._ensure_built()
        return self._model.nbody

    @property
    def n_joints(self) -> int:
        self._ensure_built()
        return self._model.njnt

    @property
    def n_creatures(self) -> int:
        return len(self._creature_names)

    @property
    def simulation_time(self) -> float:
        if self._data is not None:
            return self._data.time
        return 0.0

    # ================================================================
    # RESET / CLOSE
    # ================================================================

    def reset(self):
        """Welt zurücksetzen (State auf Anfang)."""
        if self._model is not None and self._data is not None:
            mujoco.mj_resetData(self._model, self._data)
        self._contacts = []
        self.step_count = 0

    def reset_full(self):
        """Komplett neu aufbauen (alle Bodies entfernen)."""
        self._model = None
        self._data = None
        if self._renderer:
            self._renderer.close()
            self._renderer = None
        self._body_xmls.clear()
        self._actuator_xmls.clear()
        self._sensor_xmls.clear()
        self._body_registry.clear()
        self._creature_names.clear()
        self._contacts = []
        self._built = False
        self._object_count = 0
        self.step_count = 0

    def close(self):
        """Aufräumen."""
        if self._renderer:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
        self._model = None
        self._data = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        nb = self._model.nbody if self._model else 0
        return (f"MuJoCoWorld(bodies={nb}, "
                f"creatures={self.n_creatures}, "
                f"steps={self.step_count}, "
                f"built={self._built})")
