"""Shared PyBullet sim for the SmolVLA pick-and-place study scripts.

    sim_env.py          <- this file: SO-100-style arm, scene, cameras, scripted expert
    record_demos.py     <- expert solves the task, saved as a LeRobot dataset
    lerobot-train       <- fine-tune smolvla_base on that dataset (command in record_demos.py)
    smolvla_pybullet.py <- run base or fine-tuned SmolVLA in the sim, report success rate

Action/state convention (same as SO-100 data): 6 ABSOLUTE joint targets
[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper],
5 joints in degrees, gripper in "dataset units" (0..100 = % open for our own data).
"""
import os
import tempfile

import numpy as np
import pybullet as p
import pybullet_data
import torch

TASK = 'Pick up the cube and place it in the box.'
JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
SIM_HZ, CTRL_HZ = 240, 30
IMG_H, IMG_W = 240, 320
MAX_STEP_DEG = 10.0
FINGER_MAX = 0.03                       # m, per finger

# Map SO-100 degrees -> sim radians:  rad = deg2rad(sign * deg + offset)
JOINT_SIGN = np.array([1.0, -1.0, -1.0, 1.0, 1.0])
JOINT_OFFSET_DEG = np.zeros(5)

# Arm geometry (must match the URDF below)
H_SHOULDER = 0.04 + 0.06                # base -> shoulder_lift axis
L1, L2 = 0.116, 0.135                   # upper arm, forearm
L_WRIST_TO_TCP = 0.05 + 0.065           # wrist_flex axis -> tool centre point (between finger tips)


# ---------------------------------------------------------------- URDF
def _link(name, size, z_center, rgba, mass=0.1):
    sx, sy, sz = size
    return f'''
  <link name="{name}">
    <visual><origin xyz="0 0 {z_center}"/><geometry><box size="{sx} {sy} {sz}"/></geometry>
      <material name="{name}_m"><color rgba="{rgba}"/></material></visual>
    <collision><origin xyz="0 0 {z_center}"/><geometry><box size="{sx} {sy} {sz}"/></geometry></collision>
    <inertial><origin xyz="0 0 {z_center}"/><mass value="{mass}"/>
      <inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>'''


def _joint(name, jtype, parent, child, xyz, axis, lo, hi):
    return f'''
  <joint name="{name}" type="{jtype}">
    <parent link="{parent}"/><child link="{child}"/>
    <origin xyz="{xyz}"/><axis xyz="{axis}"/>
    <limit lower="{lo}" upper="{hi}" effort="20" velocity="4"/>
  </joint>'''


def make_so100_like_urdf() -> str:
    """5-DoF arm + 2-finger gripper, roughly SO-100 link lengths. Joint angle 0 = link points up."""
    orange, black, pi = '1 0.5 0.1 1', '0.15 0.15 0.15 1', 3.1416
    body = (
        _link('base', (0.08, 0.08, 0.04), 0.02, black, mass=1.0)
        + _link('shoulder', (0.05, 0.05, 0.06), 0.03, orange)
        + _link('upper_arm', (0.03, 0.03, L1), L1 / 2, orange)
        + _link('forearm', (0.03, 0.03, L2), L2 / 2, orange)
        + _link('wrist', (0.03, 0.03, 0.05), 0.025, orange)
        + _link('hand', (0.07, 0.025, 0.02), 0.01, black)
        + _link('finger_l', (0.008, 0.02, 0.06), 0.03, orange, mass=0.02)
        + _link('finger_r', (0.008, 0.02, 0.06), 0.03, orange, mass=0.02)
        + _joint('shoulder_pan', 'revolute', 'base', 'shoulder', '0 0 0.04', '0 0 1', -pi, pi)
        + _joint('shoulder_lift', 'revolute', 'shoulder', 'upper_arm', '0 0 0.06', '0 1 0', -pi, pi)
        + _joint('elbow_flex', 'revolute', 'upper_arm', 'forearm', f'0 0 {L1}', '0 1 0', -pi, pi)
        + _joint('wrist_flex', 'revolute', 'forearm', 'wrist', f'0 0 {L2}', '0 1 0', -pi, pi)
        + _joint('wrist_roll', 'revolute', 'wrist', 'hand', '0 0 0.05', '0 0 1', -pi, pi)
        + _joint('finger_l_joint', 'prismatic', 'hand', 'finger_l', '0.004 0 0.02', '1 0 0', 0, FINGER_MAX)
        + _joint('finger_r_joint', 'prismatic', 'hand', 'finger_r', '-0.004 0 0.02', '-1 0 0', 0, FINGER_MAX)
    )
    path = os.path.join(tempfile.gettempdir(), 'so100_like.urdf')
    with open(path, 'w') as f:
        f.write(f'<?xml version="1.0"?>\n<robot name="so100_like">{body}\n</robot>')
    return path


def deg_to_rad(deg5):
    return np.deg2rad(JOINT_SIGN * np.asarray(deg5) + JOINT_OFFSET_DEG)


def rad_to_deg(rad5):
    return (np.rad2deg(np.asarray(rad5)) - JOINT_OFFSET_DEG) * JOINT_SIGN


# ---------------------------------------------------------------- analytic IK
def ik_top_down(x, y, z) -> np.ndarray:
    """Joint angles (rad, 5) that put the TCP at (x, y, z) with the gripper pointing straight down
    and the fingers opening along world x.

    The 3 pitch joints form a planar chain in the vertical plane at angle `pan`.
    Link direction angles are measured from 'up' towards 'forward':
        phi1 = lift, phi2 = lift + elbow, phi3 = phi2 + wrist_flex = pi (pointing down).
    So the wrist_flex axis sits L_WRIST_TO_TCP straight above the TCP -> 2-link IK to that point.
    """
    pan = np.arctan2(y, x)
    r = np.hypot(x, y)
    dx, dz = r, z + L_WRIST_TO_TCP - H_SHOULDER          # wrist point, relative to shoulder axis
    D2 = dx * dx + dz * dz
    c = (D2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c) > 1:
        raise ValueError(f'target ({x:.3f},{y:.3f},{z:.3f}) out of reach')
    elbow = np.arccos(c)                                   # > 0 -> elbow-up
    lift = np.arctan2(dx, dz) - np.arctan2(L2 * np.sin(elbow), L1 + L2 * np.cos(elbow))
    wrist = np.pi - lift - elbow
    roll = pan                                             # derived: hand x = (-cos(roll-pan), sin(roll-pan))
    return np.array([pan, lift, elbow, wrist, roll])


# ---------------------------------------------------------------- environment
class ArmSim:
    ARM, FINGERS, HAND = [0, 1, 2, 3, 4], [5, 6], 4

    def __init__(self, gui: bool, grip_range=(0.0, 100.0), seed: int = 0):
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(0.6, 50, -35, [0.12, 0, 0.05])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / SIM_HZ)
        self.renderer = p.ER_BULLET_HARDWARE_OPENGL if gui else p.ER_TINY_RENDERER
        self.grip_lo, self.grip_hi = grip_range
        self.rng = np.random.default_rng(seed)
        p.loadURDF('plane.urdf')
        self.robot = p.loadURDF(make_so100_like_urdf(), [0, 0, 0], useFixedBase=True)
        for j in self.FINGERS:
            p.changeDynamics(self.robot, j, lateralFriction=2.0)
        self.cube, self.box_ids = None, []

    # ---- scene ----
    def reset(self, init_state_deg=None, randomize=True):
        """New episode: random cube + box placement, arm at home (or given) pose."""
        for b in ([self.cube] if self.cube is not None else []) + self.box_ids:
            p.removeBody(b)
        if randomize:
            cube_xy = [self.rng.uniform(0.15, 0.20), self.rng.uniform(0.03, 0.09)]
            self.box_center = np.array([self.rng.uniform(0.15, 0.19), self.rng.uniform(-0.12, -0.07)])
        else:
            cube_xy, self.box_center = [0.18, 0.06], np.array([0.17, -0.09])
        self.box_half = 0.045
        he = [0.0125] * 3
        self.cube = p.createMultiBody(
            0.03, p.createCollisionShape(p.GEOM_BOX, halfExtents=he),
            p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=[0.9, 0.1, 0.1, 1]),
            basePosition=[*cube_xy, 0.0125])
        p.changeDynamics(self.cube, -1, lateralFriction=2.0)
        self.box_ids = self._make_box(self.box_center, self.box_half, height=0.035)
        if init_state_deg is None:
            init_state_deg = self.home_deg(jitter=randomize)
        self.set_state_deg(init_state_deg)
        for _ in range(30):
            p.stepSimulation()

    def _make_box(self, c, h, height, t=0.004):
        col, ids = [0.3, 0.5, 0.9, 1], []
        for pos, he in [([c[0], c[1], t / 2], [h, h, t / 2]),
                        ([c[0] + h, c[1], height / 2], [t, h, height / 2]),
                        ([c[0] - h, c[1], height / 2], [t, h, height / 2]),
                        ([c[0], c[1] + h, height / 2], [h, t, height / 2]),
                        ([c[0], c[1] - h, height / 2], [h, t, height / 2])]:
            ids.append(p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=he),
                                         p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=col), pos))
        return ids

    def home_deg(self, jitter=False):
        xyz = np.array([0.12, 0.0, 0.12])
        if jitter:
            xyz += self.rng.uniform(-0.02, 0.02, 3)
        return np.concatenate([rad_to_deg(ik_top_down(*xyz)), [self.grip_hi]])

    # ---- gripper units ----
    def _grip_to_finger(self, g):
        frac = (g - self.grip_lo) / (self.grip_hi - self.grip_lo + 1e-8)
        return float(np.clip(frac, 0, 1) * FINGER_MAX)

    def _finger_to_grip(self, x):
        return self.grip_lo + x / FINGER_MAX * (self.grip_hi - self.grip_lo)

    # ---- state / action ----
    def set_state_deg(self, s):
        for j, q in zip(self.ARM, deg_to_rad(s[:5])):
            p.resetJointState(self.robot, j, q)
        for j in self.FINGERS:
            p.resetJointState(self.robot, j, self._grip_to_finger(s[5]))
        self.target_deg = np.array(s, dtype=np.float64)
        self.apply_action(self.target_deg, substeps=0)

    def get_state_deg(self) -> np.ndarray:
        q = np.array([p.getJointState(self.robot, j)[0] for j in self.ARM + [self.FINGERS[0]]])
        return np.concatenate([rad_to_deg(q[:5]), [self._finger_to_grip(q[5])]])

    def apply_action(self, a_deg, substeps=SIM_HZ // CTRL_HZ):
        """a_deg: [6] absolute joint targets. Clipped to MAX_STEP_DEG change per control step."""
        step = np.clip(np.asarray(a_deg, dtype=np.float64) - self.target_deg, -MAX_STEP_DEG, MAX_STEP_DEG)
        self.target_deg = self.target_deg + step
        p.setJointMotorControlArray(self.robot, self.ARM, p.POSITION_CONTROL,
                                    targetPositions=deg_to_rad(self.target_deg[:5]).tolist(),
                                    forces=[20.0] * 5)
        f = self._grip_to_finger(self.target_deg[5])
        p.setJointMotorControlArray(self.robot, self.FINGERS, p.POSITION_CONTROL,
                                    targetPositions=[f, f], forces=[15.0, 15.0])
        for _ in range(substeps):
            p.stepSimulation()

    # ---- cameras ----
    def _render(self, view, fov=60) -> np.ndarray:
        proj = p.computeProjectionMatrixFOV(fov, IMG_W / IMG_H, 0.01, 2.0)
        _, _, rgba, _, _ = p.getCameraImage(IMG_W, IMG_H, view, proj, renderer=self.renderer)
        return np.asarray(rgba, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)[..., :3].copy()  # [H,W,3] uint8

    def get_images(self) -> dict[str, np.ndarray]:
        top = p.computeViewMatrix([0.50, 0.0, 0.40], [0.14, 0.0, 0.0], [0, 0, 1])
        pos, orn = p.getLinkState(self.robot, self.HAND, computeForwardKinematics=True)[4:6]
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        eye = np.array(pos) + R @ [0.0, 0.04, -0.02]
        wrist = p.computeViewMatrix(eye, eye + R @ [0, 0, 0.1], R @ [1, 0, 0])
        return {'top': self._render(top), 'wrist': self._render(wrist, fov=75)}

    @staticmethod
    def to_tensor(img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0     # [3,H,W] in [0,1]

    # ---- task ----
    def cube_pos(self):
        return np.array(p.getBasePositionAndOrientation(self.cube)[0])

    def success(self) -> bool:
        x, y, z = self.cube_pos()
        inside = np.all(np.abs(np.array([x, y]) - self.box_center) < self.box_half - 0.005)
        return bool(inside and z < 0.03)


# ---------------------------------------------------------------- scripted expert
def smoothstep(n):
    s = np.linspace(0, 1, n + 1)[1:]
    return s * s * (3 - 2 * s)


def expert_plan(env: ArmSim, rng: np.random.Generator) -> list[np.ndarray]:
    """Full action sequence (list of [6] deg targets) for pick-and-place, from the env's true state.
    Waypoints in Cartesian space -> analytic IK -> smooth interpolation in joint space."""
    cx, cy, _ = env.cube_pos()
    bx, by = env.box_center
    o, c = env.grip_hi, env.grip_lo
    hover, grasp_z, carry, drop = 0.075, 0.022, 0.075, 0.07
    dur = lambda base: int(base * rng.uniform(0.85, 1.2))      # speed variation across demos
    waypoints = [  # (tcp xyz, gripper, steps)
        ((cx, cy, hover), o, dur(40)),
        ((cx, cy, grasp_z), o, dur(25)),
        ((cx, cy, grasp_z), c, dur(15)),
        ((cx, cy, carry), c, dur(25)),
        ((bx, by, carry), c, dur(40)),
        ((bx, by, drop), c, dur(10)),
        ((bx, by, drop), o, dur(12)),
        ((bx, by, hover + 0.02), o, dur(20)),
    ]
    plan, prev = [], env.get_state_deg()
    for xyz, g, n in waypoints:
        goal = np.concatenate([rad_to_deg(ik_top_down(*xyz)), [g]])
        for s in smoothstep(n):
            plan.append(prev + s * (goal - prev))
        prev = goal
    return plan