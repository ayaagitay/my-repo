import os
import numpy as np
import mujoco
import imageio

RL_GYM = r"C:\Users\ENVY\PycharmProjects\unitree_rl_gym"

XML = os.path.join(RL_GYM, "resources", "robots", "g1_description", "g1_29dof.xml")
CSV = "motion.csv"
OUT = "motion.mp4"
FPS = 30

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
motion = np.loadtxt(CSV, delimiter=",")

print("CSV shape:", motion.shape, "| model nq:", model.nq)
assert motion.shape[1] == model.nq, "CSV columns do not match the model's nq"

print("quat norm (row 0):", np.linalg.norm(motion[0, 3:7]))
lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]   # skip the free joint
q = motion[:, 7:]
print("joint values outside limits:", int(((q < lo) | (q > hi)).sum()))

for j in range(model.njnt):
    print(j, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))

model.vis.global_.offwidth = 1280
model.vis.global_.offheight = 720
renderer = mujoco.Renderer(model, height=720, width=1280)

cam = mujoco.MjvCamera()
cam.distance, cam.elevation, cam.azimuth = 3.0, -10, 150

with imageio.get_writer(OUT, fps=FPS) as writer:
    for row in motion:
        data.qpos[0:3] = row[0:3]
        x, y, z, w = row[3:7]
        data.qpos[3:7] = [w, x, y, z]      # CSV is xyzw, MuJoCo wants wxyz
        data.qpos[7:] = row[7:]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = data.qpos[0:3]
        renderer.update_scene(data, cam)
        writer.append_data(renderer.render())

print("Saved", OUT)