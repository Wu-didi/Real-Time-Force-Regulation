"""Public Robotis20DoF + public YCB rigid-body simulation adapter."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np,mujoco
from .robotis import build_model,HandSurface,robot_jacobian

def build_ycb(robotis_xml, directory, mass, position=(.055,0,.14), quaternion=(1,0,0,0)):
    directory = Path(directory).resolve()
    record = json.loads((directory/'READY.json').read_text())
    _,xml,_,_=build_model(robotis_xml);root=ET.fromstring(xml)
    a=root.find('asset');world=root.find('worldbody')
    body=ET.SubElement(world,'body',{'name':'test_object','pos':' '.join(map(str,position)),'quat':' '.join(map(str,quaternion))})
    ET.SubElement(body,'freejoint',{'name':'object_free'})
    I=np.asarray(record['inertia_density1'])*mass/record['volume_m3']
    ET.SubElement(body,'inertial',{'mass':str(mass),'pos':' '.join(map(str,record['COM_m'])),'fullinertia':' '.join(map(str,[I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]))})
    ET.SubElement(a,'mesh',{'name':'object_visual','file':str(directory/'canonical.ply')})
    # MuJoCo accepts obj/stl, so use original transformed canonical OBJ export.
    visual=directory/'canonical.obj'
    a.find("mesh[@name='object_visual']").set('file',str(visual))
    ET.SubElement(body,'geom',{'name':'object_visual','mesh':'object_visual','type':'mesh','contype':'0','conaffinity':'0','group':'2','rgba':'.8 .5 .15 1','mass':'0'})
    collision_parts = sorted(directory.glob('collision_*.obj'))
    if not collision_parts:
        raise ValueError(f'No collision meshes in {directory}')
    for i,path in enumerate(collision_parts):
        namepart=f'object_collision_{i}';ET.SubElement(a,'mesh',{'name':namepart,'file':str(path)})
        ET.SubElement(body,'geom',{'name':namepart,'type':'mesh','mesh':namepart,'friction':'.7 0 0','condim':'3','rgba':'.8 .5 .15 0','group':'3','mass':'0'})
    xml=ET.tostring(root,encoding='unicode');model=mujoco.MjModel.from_xml_string(xml)
    ids=np.array([model.joint(f'finger_r_joint{i}').id for i in range(1,21)])
    return model,xml,model.jnt_qposadr[ids],model.jnt_dofadr[ids],record

def hand_hull_points(model,data,surface):
    points=[]
    for b in [4,8,12,16,20]:
        bid=model.body(f'finger_r_link{b}').id
        clouds=[p for p,gid,bid2 in zip(surface.points_local,surface.geom_ids,surface.body_ids) if bid2==bid]
        if not clouds:raise ValueError('missing distal link mesh')
        # Geometry centroid convention, explicitly local: author exact tip sites unpublished.
        world=[]
        for p,gid,bid2 in zip(surface.points_local,surface.geom_ids,surface.body_ids):
            if bid2==bid:world.append(p@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid])
        points.append(np.concatenate(world).mean(0))
    palm=[]
    for p,gid,is_active in zip(surface.points_local,surface.geom_ids,surface.active):
        if not is_active:palm.append(p@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid])
    points.append(np.concatenate(palm).mean(0));return np.array(points)

# Group all visual components of a link before drawing 2000 points, as the paper states.
BaseHandSurface=HandSurface
class PaperHandSurface(BaseHandSurface):
    @classmethod
    def from_model(cls,model,seed=260930082,density_per_cm2=2.,draw_per_link=2000,mode='visual'):
        from wholehand.geometry import farthest_point_subsample
        import trimesh
        rng=np.random.default_rng(seed);groups={}
        for gid in range(model.ngeom):
            if model.geom_group[gid]!=(2 if mode=='visual' else 3) or model.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_MESH:continue
            bid=int(model.geom_bodyid[gid]);name=model.body(bid).name
            if not name.startswith(('finger_r_link','hx5_d20_right_base')):continue
            groups.setdefault(bid,[]).append(gid)
        gids=[];clouds=[];bodies=[];active=[]
        for bid,items in sorted(groups.items()):
            vertices=[];faces=[];count=0
            for gid in items:
                mid=model.geom_dataid[gid];va=model.mesh_vertadr[mid];vn=model.mesh_vertnum[mid];fa=model.mesh_faceadr[mid];fn=model.mesh_facenum[mid]
                R=np.zeros(9);mujoco.mju_quat2Mat(R,model.geom_quat[gid])
                v=model.mesh_vert[va:va+vn]@R.reshape(3,3).T+model.geom_pos[gid]
                vertices.append(v);faces.append(model.mesh_face[fa:fa+fn]+count);count+=vn
            mesh=trimesh.Trimesh(np.concatenate(vertices),np.concatenate(faces),process=False)
            if mode=='collision_hull':mesh=mesh.convex_hull
            sampled,_=trimesh.sample.sample_surface(mesh,draw_per_link,seed=rng)
            sampled=farthest_point_subsample(sampled,max(1,round(mesh.area*density_per_cm2*10000)))
            gid=items[0];R=np.zeros(9);mujoco.mju_quat2Mat(R,model.geom_quat[gid])
            # Store in the representative geom frame to retain the verified world() API.
            local=(sampled-model.geom_pos[gid])@R.reshape(3,3)
            gids.append(gid);clouds.append(local);bodies.append(bid);active.append(model.body_jntnum[bid]>0)
        return cls(np.array(gids,dtype=int),clouds,bodies,active)
HandSurface=PaperHandSurface
