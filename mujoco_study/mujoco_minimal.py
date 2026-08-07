import mujoco
import mujoco.viewer
import time
import numpy as np
from PIL import Image

model = mujoco.MjModel.from_xml_path('franka_panda.xml')
data = mujoco.MjData(model)

# --- Solved via numerical IK against this model's own forward kinematics (scipy) ---
# Only joint2/4/6 (the pitch joints) affect end-effector reach; joint1/3/5/7 (yaw) are
# left at 0 so the fingers close along world Y. The optimizer was run with a penalty
# term that keeps every intermediate link above the table surface (z > 0.10) across
# the ENTIRE interpolated path, not just at the waypoints -- an earlier version of
# this solve matched the end-effector position fine but swung the forearm/link6
# straight through the tabletop to get there, since the fingers were also forced to
# point exactly straight down. Dropping that "must point down" constraint (finger
# open/close is along world-Y regardless of arm orientation, so it isn't actually
# needed for the pinch) gave the optimizer enough freedom to find a table-clear path.
HOME  = [0.0, 0.0, 0.0]
HOVER = [0.8109, -1.5103, 2.1098]   # above block, clear of table
GRASP = [0.5667, -2.3374, -0.2627]  # fingers straddle block center exactly (0.4, 0, 0.132)
LIFT  = [-0.2994, -2.3837, -0.0455] # raised, still gripping
OPEN, CLOSED = 0.0, 0.020           # finger_joint travel: 0=open, ~0.02=touching a 4cm-wide block

def set_arm_ctrl(data, th2, th4, th6, finger):
    data.ctrl[0] = 0.0     # joint1 (yaw, unused)
    data.ctrl[1] = th2
    data.ctrl[2] = 0.0     # joint3 (yaw, unused)
    data.ctrl[3] = th4
    data.ctrl[4] = 0.0     # joint5 (yaw, unused)
    data.ctrl[5] = th6
    data.ctrl[6] = 0.0     # joint7 (yaw, unused)
    data.ctrl[7] = finger
    data.ctrl[8] = finger

def smoothstep(u):
    """Ease-in/ease-out (zero velocity at both ends). Prevents the torque spike
    you get from snapping ctrl straight to a new target, which was flinging the
    block out of the gripper."""
    u = min(max(u, 0.0), 1.0)
    return 3 * u * u - 2 * u * u * u

# init at HOME pose (use named joint access, NOT raw qpos indices -- the block's
# free joint occupies qpos[0:7], so joint2 is qpos[8], not qpos[1])
set_arm_ctrl(data, *HOME, OPEN)
mujoco.mj_forward(model, data)

# (start_pose, end_pose, start_grip, end_grip, duration_s)
segments = [
    (HOME,  HOVER, OPEN,   OPEN,   1.5),   # swing up and over to the block
    (HOVER, GRASP, OPEN,   OPEN,   1.5),   # descend
    (GRASP, GRASP, OPEN,   CLOSED, 1.2),   # close gripper
    (GRASP, LIFT,  CLOSED, CLOSED, 3.0),   # lift
]

cam_width, cam_height = 320, 240
renderer = mujoco.Renderer(model, height=cam_height, width=cam_width)

print("Running grasp sequence... Press ESC in the GUI window to stop.")
step_count = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    for p0, p1, g0, g1, dur in segments:
        steps = int(dur / model.opt.timestep)
        for i in range(steps):
            if not viewer.is_running():
                break
            step_start = time.time()
            u = smoothstep(i / steps)
            pose = [p0[j] + (p1[j] - p0[j]) * u for j in range(3)]
            grip = g0 + (g1 - g0) * u
            set_arm_ctrl(data, *pose, grip)

            mujoco.mj_step(model, data)
            viewer.sync()

            if step_count % 250 == 0:
                renderer.update_scene(data, camera="vla_front_camera")
                rgb_array = renderer.render()
                Image.fromarray(rgb_array).save("vla_observation.png")

            step_count += 1
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    block_z = data.body('green_block').xpos[2]
    print(f"\nFinal block height: {block_z:.3f} (resting height was ~0.13) "
          f"-> {'GRASP SUCCESS' if block_z > 0.2 else 'grasp failed'}")
    print("Sequence complete, holding final pose. Press ESC to exit.")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()