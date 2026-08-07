import mujoco
import time

# 1. Load the MJCF robot model
try:
    model = mujoco.MjModel.from_xml_path('panda_test.xml')
    data = mujoco.MjData(model)
    print("MuJoCo model loaded successfully!")
except ValueError as e:
    print(f"Error loading model: {e}")
    exit()

# 2. Inspect Model Structure (Equivalent to p.getNumJoints)
num_joints = model.nq  # nq represents the number of generalized coordinates
print(f"Number of joint positions (nq): {num_joints}")

# Print names of the joints
for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"Joint {i}: {joint_name}")

# 3. Step Simulation Headless
print("\nStarting headless simulation loop...")
for step in range(500):
    # Advance physics engine by one timestep
    mujoco.mj_step(model, data)

    # Read joint angles every 100 steps
    if step % 100 == 0:
        print(f"\n--- Simulation Step: {step} ---")
        for i in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            # qpos holds position data; joint Qpos address maps the index
            joint_pos = data.qpos[model.jnt_qposadr[i]]
            print(f"{joint_name} angle: {joint_pos:.4f} rad")

    time.sleep(0.002)

print("\nSimulation complete.")
