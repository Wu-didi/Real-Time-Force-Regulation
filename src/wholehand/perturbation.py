"""World-frame passive spring/damper wrench applied about object COM."""
import numpy as np,mujoco
from scipy.spatial.transform import Rotation

def spring_wrench(model,data,body,target_com,target_rotation,translation_kp=100.,translation_kd=2.,rotation_kp=.5,rotation_kd=.005):
    mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data);mujoco.mj_comVel(model,data)
    velocity=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,int(body),velocity,0)
    rotation=data.xmat[body].reshape(3,3)
    error=Rotation.from_matrix(np.asarray(target_rotation)@rotation.T).as_rotvec()
    force=translation_kp*(np.asarray(target_com)-data.xipos[body])-translation_kd*velocity[3:]
    torque=rotation_kp*error-rotation_kd*velocity[:3]
    return np.r_[force,torque],velocity
