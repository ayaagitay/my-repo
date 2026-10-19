import os
import numpy as np
import mujoco
import torch
import yaml
import imageio

RL_GYM = r"C:\Users\ENVY\PycharmProjects\unitree_rl_gym"
CONFIG = os.path.join(RL_GYM, "deploy", "deploy_mujoco", "configs", "g1.yaml")
OUT = "walk_circle.mp4"
SIM_SECONDS = 20
FPS = 30
FOLLOW = False


def get_gravity_orientation(q):
    qw, qx, qy, qz = q
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


with open(CONFIG, "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
policy_path = cfg["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", RL_GYM)
xml_path = cfg["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", RL_GYM)
dt = cfg["simulation_dt"]
decimation = cfg["control_decimation"]
kps = np.array(cfg["kps"], dtype=np.float32)
kds = np.array(cfg["kds"], dtype=np.float32)
default_angles = np.array(cfg["default_angles"], dtype=np.float32)
ang_vel_scale = cfg["ang_vel_scale"]
dof_pos_scale = cfg["dof_pos_scale"]
dof_vel_scale = cfg["dof_vel_scale"]
action_scale = cfg["action_scale"]
cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
num_actions = cfg["num_actions"]
num_obs = cfg["num_obs"]
cmd = np.array(cfg["cmd_init"], dtype=np.float32)
print("command [forward, lateral, yaw]:", cmd)

action = np.zeros(num_actions, dtype=np.float32)
target_dof_pos = default_angles.copy()
obs = np.zeros(num_obs, dtype=np.float32)

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
m.opt.timestep = dt
policy = torch.jit.load(policy_path)

m.vis.global_.offwidth = 1280
m.vis.global_.offheight = 720
renderer = mujoco.Renderer(m, height=720, width=1280)
cam = mujoco.MjvCamera()
cam.distance, cam.elevation, cam.azimuth = 6.0, -50, 90
cam.lookat[:] = [0.0, 1.0, 0.6]

counter = 0
next_frame = 0.0
xy = []
with imageio.get_writer(OUT, fps=FPS) as writer:
    while d.time < SIM_SECONDS:
        d.ctrl[:] = pd_control(target_dof_pos, d.qpos[7:], kps,
                               np.zeros_like(kds), d.qvel[6:], kds)
        mujoco.mj_step(m, d)
        counter += 1

        if counter % decimation == 0:
            qj = (d.qpos[7:] - default_angles) * dof_pos_scale
            dqj = d.qvel[6:] * dof_vel_scale
            omega = d.qvel[3:6] * ang_vel_scale
            grav = get_gravity_orientation(d.qpos[3:7])
            phase = (counter * dt) % 0.8 / 0.8
            obs[:3] = omega
            obs[3:6] = grav
            obs[6:9] = cmd * cmd_scale
            obs[9:9 + num_actions] = qj
            obs[9 + num_actions:9 + 2 * num_actions] = dqj
            obs[9 + 2 * num_actions:9 + 3 * num_actions] = action
            obs[9 + 3 * num_actions:9 + 3 * num_actions + 2] = [
                np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
            action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            target_dof_pos = action * action_scale + default_angles

        if d.time >= next_frame:
            if FOLLOW:
                cam.lookat[:] = d.qpos[0:3]
            renderer.update_scene(d, cam)
            writer.append_data(renderer.render())
            xy.append(d.qpos[:2].copy())
            next_frame += 1.0 / FPS

xy = np.array(xy)
A = np.c_[2 * xy, np.ones(len(xy))]
b = (xy ** 2).sum(axis=1)
c = np.linalg.lstsq(A, b, rcond=None)[0]
radius = np.sqrt(c[2] + c[0] ** 2 + c[1] ** 2)
print(f"fitted circle centre=({c[0]:.2f}, {c[1]:.2f}) radius={radius:.2f} m")
print(f"final pelvis height: {d.qpos[2]:.2f} m (standing is about 0.7-0.8)")
print("Saved", OUT)