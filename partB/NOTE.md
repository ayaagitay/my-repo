# Part B: G1 walking in a circle

I cloned unitree_rl_gym and ran the pretrained G1 policy (`deploy/pre_train/g1/motion.pt`) with `deploy_mujoco.py g1.yaml`. 
By default the robot walks straight ahead. The only thing I changed was `cmd_init` in `deploy/deploy_mujoco/configs/g1.yaml`:

    cmd_init: [0.5, 0, 0]  to  cmd_init: [0.5, 0, 0.5]

The three values are forward speed in m/s, sideways speed in m/s and turning rate in rad/s.
My copy of the config is `g1_circle.yaml`. The live MuJoCo viewer ran in slow motion on my laptop, so `record_circle.py` runs the same control loop but renders offscreen and writes `walk_circle.mp4` at real-time speed.


The script reads the command once and gives the policy the same command at every step. 
The command is in the robot's own frame, so "forward" always means wherever the robot is currently facing. 
With a turning rate of zero the heading never changes and the path is a straight line. 
With 0.5 rad/s the heading rotates all the time while the robot keeps stepping forward at 0.5 m/s, and that traces a circle of radius speed divided by turning rate, 0.5 / 0.5 = 1 m. 
I checked this by fitting a circle to the pelvis position over 20 s of simulation. 
The radius came out at 1.08 m, and the pelvis stayed at 0.77 m, so the robot did not fall. 
The policy follows the turning command only approximately, which probably explains the small difference from 1 m.

## What inputs does the policy receive, and where do they come from?
The policy receives a vector of 47 numbers every 0.02 s (once every 10 physics steps).
Three numbers are the body's angular velocity from `d.qvel`, scaled by 0.25. 
Three are the gravity direction in the body frame, calculated from the base quaternion in `d.qpos`, which tells the policy how the body is tilted. 
Three are the command from `cmd_init`, multiplied by `cmd_scale` and this is the only part that I changed. 
Twelve are the leg joint positions from `d.qpos` minus the default standing pose, and twelve are the joint velocities from `d.qvel`, scaled by 0.05.
Twelve are the policy's own previous action, which gives it a memory of what it just did. 
The last two are the sine and cosine of a clock that repeats every 0.8 s, which lets the policy keep a regular stepping rhythm. 
In simulation the body and joint values come straight from MuJoCo. On the real robot they would come from the IMU and the joint encoders.

## What does the policy output, and how is that converted into joint torques?
The output is 12 numbers, one per leg joint. The code turns them into target angles with `target_dof_pos = action * action_scale + default_angles`, where `action_scale` is 0.25, so the policy only asks for small offsets around the default standing pose. 
The torque for each joint then comes from a PD controller: `torque = kp * (target_angle - current_angle) + kd * (0 - current_velocity)`. The kp term acts like a spring pulling the joint toward its target, and the kd term acts like a damper that stops it from oscillating. 
The gains are the per-joint `kps` and `kds` lists in the yaml. These torques are written to `d.ctrl` and MuJoCo steps the physics every 0.002 s (500 Hz), while the policy only updates the target angles at 50 Hz, so the same target is held for 10 physics steps.