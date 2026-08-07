import mujoco
import mujoco.viewer
import numpy as np
import time
import torch


class Car():
    def __init__(self, max_steps: int=3*500, seed: int=0, render: bool=False):
        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path('./car.xml')
        self.data: mujoco.MjData = mujoco.MjData(self.model)
        self.duration: int = int(max_steps // 500)
        self.single_action_space: tuple = (2,)
        self.single_observation_space: tuple = (13,)
        self.viewer: mujoco.viewer = None
        self.reset()

        if render:
            self._run_with_render()

    def reset(self) -> None:
        self.model = mujoco.MjModel.from_xml_path('./car.xml')
        self.data = mujoco.MjData(self.model)

    def reward(self, state: list, action):
        goal: list = [-1,4]
        car_dist: float = (np.linalg.norm(np.array(goal-state[:2])))
        return np.exp(-((car_dist)))

    def _run_with_render(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print("Press ESC to exit")

            # --- Camera tracking setup ---
            car_body_name: str = "car"
            car_body_id: int = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, car_body_name
            )

            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = car_body_id
            viewer.cam.distance = 5.0     # how far the camera sits from the car
            viewer.cam.azimuth = 90.0     # horizontal angle (deg)
            viewer.cam.elevation = -20.0  # vertical angle (deg, negative looks down)

            step_count: int = 0
            print_every: int = 500

            while viewer.is_running():
                self.data.ctrl[0] = 2.0
                self.data.ctrl[1] = 0.1

                mujoco.mj_step(self.model, self.data)
                # Sync changes to viewer
                viewer.sync()

                step_count += 1
                if step_count % print_every == 0:
                    print(
                        f"t={self.data.time:.2f}s | "
                        f"pos={np.round(self.data.qpos[:3], 3)} | "
                        f"quat={np.round(self.data.xquat[1], 3)} | "
                        f"vel={np.round(self.data.qvel[:3], 3)}"
                    )

                time.sleep(0.001)


if __name__ == "__main__":
    car: Car = Car(render=True)