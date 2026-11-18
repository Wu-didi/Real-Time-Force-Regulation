"""Public Robotis rev2 model adapter; fixed-palm engineering simulation only."""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import trimesh
from .geometry import farthest_point_subsample

@dataclass
class HandSurface:
    geom_ids: np.ndarray
    points_local: list
    body_ids: list
    active: list

    @classmethod
    def from_model(cls, model, seed=260930082, density_per_cm2=2., draw_per_link=2000, mode="visual"):
        if mode not in ("visual", "collision_hull"): raise ValueError(mode)
        """Sample raw surface triangles via compiled, geom-frame mesh coordinates."""
        rng=np.random.default_rng(seed)
        gids=[]; clouds=[]; bodies=[]; active=[]
        for gid in range(model.ngeom):
            if model.geom_group[gid] != (2 if mode == "visual" else 3) or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            bid=int(model.geom_bodyid[gid])
            if not mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,bid).startswith(("finger_r_link","hx5_d20_right_base")):
                continue
            mid=int(model.geom_dataid[gid])
            va=int(model.mesh_vertadr[mid]); vn=int(model.mesh_vertnum[mid])
            fa=int(model.mesh_faceadr[mid]); fn=int(model.mesh_facenum[mid])
            mesh=trimesh.Trimesh(vertices=model.mesh_vert[va:va+vn],
                                 faces=model.mesh_face[fa:fa+fn],process=False)
            if mode == "collision_hull": mesh=mesh.convex_hull
            # trimesh API permits Generator seed; deterministic per stored seed.
            points,_=trimesh.sample.sample_surface(mesh,draw_per_link,seed=rng)
            count=max(1,int(round(mesh.area*density_per_cm2*10000)))
            sampled=farthest_point_subsample(points,count)
            gids.append(gid); clouds.append(sampled); bodies.append(bid)
            active.append(model.body_jntnum[bid] > 0)
        return cls(np.array(gids,dtype=int),clouds,bodies,active)

    def world(self,data):
        clouds=[]; links=[]; active=[]
        for gid,p,bid,is_active in zip(self.geom_ids,self.points_local,self.body_ids,self.active):
            R=data.geom_xmat[gid].reshape(3,3)
            clouds.append(p@R.T+data.geom_xpos[gid])
            links.extend([bid]*len(p)); active.extend([is_active]*len(p))
        return np.concatenate(clouds),np.array(links),np.array(active)

def build_model(source, object_spec=None, timestep=.002):
    source = Path(source).resolve()
    tree=ET.parse(source)
    root=tree.getroot()
    compiler=root.find("compiler")
    compiler.set("meshdir",str((source.parent/compiler.get("meshdir")).resolve()))
    option=root.find("option")
    option.set("gravity","0 0 0")
    option.set("timestep",str(timestep))
    # Generated derivative only: immutable public source model is never overwritten.
    keyframe=root.find("keyframe")
    if keyframe is not None: root.remove(keyframe)
    actuator=root.find("actuator")
    for child in list(actuator):
        actuator.remove(child)
    for i in range(1,21):
        ET.SubElement(actuator,"motor",{"name":f"finger_r_joint{i}",
            "joint":f"finger_r_joint{i}","gear":"1",
            "ctrllimited":"true","ctrlrange":"-0.2 0.2"})
    for geom in root.iter("geom"):
        if geom.get("class","").endswith("_collision"):
            geom.set("friction","0.7 0 0"); geom.set("condim","3")
    if object_spec is not None:
        body=ET.SubElement(root.find("worldbody"),"body",{
            "name":"test_object","pos":" ".join(map(str,object_spec["position"]))})
        ET.SubElement(body,"freejoint",{"name":"object_free"})
        attrs={"name":"object_geom","type":object_spec["type"],
               "size":" ".join(map(str,object_spec["size"])),
               "mass":str(object_spec.get("mass",.1)),
               "friction":"0.7 0 0","condim":"3","rgba":"0.8 0.3 0.1 1"}
        ET.SubElement(body,"geom",attrs)
    xml=ET.tostring(root,encoding="unicode")
    model=mujoco.MjModel.from_xml_string(xml)
    robot_joints=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,
                                            f"finger_r_joint{i}") for i in range(1,21)])
    qadr=model.jnt_qposadr[robot_joints].copy()
    dof=model.jnt_dofadr[robot_joints].copy()
    assert model.nu == 20 and len(set(dof)) == 20
    np.testing.assert_allclose(model.actuator_gear[:,0],1)
    np.testing.assert_array_equal(model.actuator_trnid[:,0],robot_joints)
    return model,xml,qadr,dof

def robot_jacobian(model,data,robot_dofs):
    def jac(body,point):
        J=np.zeros((3,model.nv))
        mujoco.mj_jac(model,data,J,None,np.asarray(point),body)
        return J[:,robot_dofs]
    return jac
