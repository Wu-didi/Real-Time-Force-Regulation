"""Mesh, hand surface, and SDF preparation for the native DexJoCo scene."""
import numpy as np
import mujoco
import trimesh
from scipy.spatial import cKDTree
from .geometry import BoxSDF,SphereSDF,MeshSDF,farthest_point_subsample
from .kinematics import HandKinematics,quaternion_matrix

class CylinderSDF:
    def __init__(self,radius,half_height):self.radius=radius;self.half_height=half_height
    def query(self,points):
        p=np.asarray(points);rad=np.linalg.norm(p[:,:2],axis=1)
        xy=p[:,:2]/np.maximum(rad[:,None],1e-12);xy[rad<1e-12]=(1,0)
        v=np.c_[rad-self.radius,np.abs(p[:,2])-self.half_height]
        outside=np.maximum(v,0);length=np.linalg.norm(outside,axis=1)
        d=length+np.minimum(v.max(axis=1),0)
        weights=outside/np.maximum(length[:,None],1e-12)
        inside=length<1e-12;weights[inside]=0
        ids=np.flatnonzero(inside);weights[ids,np.argmax(v[inside],axis=1)]=1
        n=np.c_[xy*weights[:,0,None],np.where(p[:,2]>=0,1.,-1.)*weights[:,1]]
        return d,n

class CapsuleSDF:
    def __init__(self,radius,half_length):self.radius=radius;self.half_length=half_length
    def query(self,points):
        p=np.array(points,dtype=float,copy=True)
        p[:,2]-=np.clip(p[:,2],-self.half_length,self.half_length)
        r=np.linalg.norm(p,axis=1);n=p/np.maximum(r[:,None],1e-12)
        n[r<1e-12]=(1,0,0)
        return r-self.radius,n

def geom_mesh(model,gid):
    typ=int(model.geom_type[gid]);size=model.geom_size[gid]
    if typ==int(mujoco.mjtGeom.mjGEOM_MESH):
        mid=int(model.geom_dataid[gid])
        v=int(model.mesh_vertadr[mid]);nv=int(model.mesh_vertnum[mid])
        f=int(model.mesh_faceadr[mid]);nf=int(model.mesh_facenum[mid])
        return trimesh.Trimesh(model.mesh_vert[v:v+nv],model.mesh_face[f:f+nf],process=False).convex_hull
    if typ==int(mujoco.mjtGeom.mjGEOM_BOX):return trimesh.creation.box(extents=2*size)
    if typ==int(mujoco.mjtGeom.mjGEOM_SPHERE):return trimesh.creation.icosphere(subdivisions=3,radius=size[0])
    if typ==int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return trimesh.creation.cylinder(radius=size[0],height=2*size[1],sections=64)
    if typ==int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        mesh=trimesh.creation.capsule(radius=size[0],height=2*size[1],count=[24,48])
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        return mesh
    raise ValueError(f"unsupported collision geometry type{typ}, geom{gid}")

def geom_sdf(model,gid):
    typ=int(model.geom_type[gid]);size=model.geom_size[gid]
    if typ==int(mujoco.mjtGeom.mjGEOM_BOX):return BoxSDF(size.copy())
    if typ==int(mujoco.mjtGeom.mjGEOM_SPHERE):return SphereSDF(float(size[0]))
    if typ==int(mujoco.mjtGeom.mjGEOM_CYLINDER):return CylinderSDF(float(size[0]),float(size[1]))
    if typ==int(mujoco.mjtGeom.mjGEOM_CAPSULE):return CapsuleSDF(float(size[0]),float(size[1]))
    if typ==int(mujoco.mjtGeom.mjGEOM_MESH):
        return MeshSDF.from_mesh(geom_mesh(model,gid),voxel_size=.002,padding=.02)
    raise ValueError(f"unsupported object SDF geom type{typ}")

def collision_geoms(model,body):
    return [i for i in range(model.ngeom) if model.geom_bodyid[i]==body and
            (model.geom_contype[i] or model.geom_conaffinity[i])]

def descendants(model,root):
    result=[root]
    for body in range(root+1,model.nbody):
        if int(model.body_parentid[body]) in result:result.append(body)
    return result

def export_native_hand(model,data,*,seed=260930082,density_per_cm2=2.,draw_per_link=2000):
    """Derive two21-link Allegro hands and native46D state mapping, without object assets."""
    hands=[];bodies=[];link_bodies=[];point_local=[];normal_local=[];point_link=[]
    rng=np.random.default_rng(seed)
    for hand_index,side in enumerate(("right","left")):
        root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f"allegro_palm_{side}")
        if root<0:raise ValueError("native bimanual Allegro palm missing")
        tcp=np.r_[data.sensor(f"franka/flange_pos_{side}").data,
                  data.sensor(f"franka/flange_quat_{side}").data]
        tcp_R=quaternion_matrix(tcp[3:])
        hand_bodies=descendants(model,root)
        hand_joint_ids=[]
        for bid in hand_bodies:
            hand_joint_ids.extend(range(int(model.body_jntadr[bid]),
                                        int(model.body_jntadr[bid]+model.body_jntnum[bid])))
        if len(hand_joint_ids)!=16:raise ValueError("expected16independent hand joints")
        # Native sensor order equals the source body's DOF order on each side;
        # caller verifies every sensor against this map before using labels.
        state_joint={j:14+16*hand_index+i for i,j in enumerate(hand_joint_ids)}
        hands.append({"side":side,"root_body":root,"tcp_slice":[7*hand_index,7*hand_index+7],
            "palm_in_tcp_pos":(tcp_R.T@(data.xpos[root]-tcp[:3])).tolist(),
            "palm_in_tcp_rot":(tcp_R.T@data.xmat[root].reshape(3,3)).tolist(),
            "joint_ids":hand_joint_ids,"joint_state_indices":[state_joint[j] for j in hand_joint_ids]})
        for bid in hand_bodies:
            joints=[]
            for j in range(int(model.body_jntadr[bid]),int(model.body_jntadr[bid]+model.body_jntnum[bid])):
                if model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_HINGE:raise ValueError("hinge-only exported hand")
                joints.append({"name":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j),
                    "state_index":state_joint[j],"axis":model.jnt_axis[j].tolist(),
                    "pos":model.jnt_pos[j].tolist(),"reference":float(model.qpos0[model.jnt_qposadr[j]])})
            bodies.append({"id":bid,"name":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,bid),
                "parent":int(model.body_parentid[bid]),"pos":model.body_pos[bid].tolist(),
                "rot":quaternion_matrix(model.body_quat[bid]).tolist(),"joints":joints})
            geoms=collision_geoms(model,bid)
            if not geoms:continue
            meshes=[geom_mesh(model,g) for g in geoms]
            areas=np.array([mesh.area for mesh in meshes])
            which=rng.choice(len(geoms),size=draw_per_link,p=areas/areas.sum())
            cloud=[];normals=[]
            for k,(gid,mesh) in enumerate(zip(geoms,meshes)):
                count=int(sum(which==k))
                if not count:continue
                p,face=trimesh.sample.sample_surface(mesh,count,seed=rng)
                R=quaternion_matrix(model.geom_quat[gid])
                cloud.append(p@R.T+model.geom_pos[gid])
                normals.append(mesh.face_normals[face]@R.T)
            cloud=np.concatenate(cloud);normals=np.concatenate(normals)
            keep=farthest_point_subsample(cloud,max(1,round(areas.sum()*density_per_cm2*10000)))
            ids=cKDTree(cloud).query(keep,k=1)[1]
            index=len(link_bodies);link_bodies.append(bid)
            point_local.append(cloud[ids]);normal_local.append(normals[ids])
            point_link.extend([index]*len(ids))
    metadata={"version":"native_allegro_bimanual_surface_fk_v1","robot_state_dim":46,
        "robot_state_layout":"rightTCP7,leftTCP7,rightHand16,leftHand16; wxyz quaternion",
        "hands":hands,"bodies":bodies,"link_bodies":link_bodies,
        "link_names":[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b) for b in link_bodies],
        "surface":"MuJoCo compiled collision geometry, convex hull for mesh geoms",
        "density_per_cm2":density_per_cm2,"draw_per_link":draw_per_link,"seed":seed,
        "object_geometry_included":False}
    return HandKinematics(metadata,np.concatenate(point_local),np.concatenate(normal_local),point_link)

class RigidObjectGrid:
    def __init__(self,body_id,name,grid,metadata):
        self.body_id=body_id;self.name=name;self.grid=grid;self.metadata=metadata
    @classmethod
    def from_body(cls,model,body,*,voxel=.002,padding=.02):
        geoms=collision_geoms(model,body)
        if not geoms:raise ValueError("object has no collision geometry")
        components=[];bounds=[];meta=[]
        for gid in geoms:
            mesh=geom_mesh(model,gid);R=quaternion_matrix(model.geom_quat[gid]);p=model.geom_pos[gid].copy()
            components.append((geom_sdf(model,gid),p,R))
            vertices=mesh.vertices@R.T+p;bounds.append(np.array([vertices.min(0),vertices.max(0)]))
            meta.append({"geom_id":gid,"type":int(model.geom_type[gid]),"size":model.geom_size[gid].tolist(),
                         "position":p.tolist(),"rotation":R.tolist()})
        lo=np.min(np.stack(bounds)[:,0],axis=0)-padding;hi=np.max(np.stack(bounds)[:,1],axis=0)+padding
        axes=[np.linspace(a,b,int(np.ceil((b-a)/voxel))+1) for a,b in zip(lo,hi)]
        points=np.stack(np.meshgrid(*axes,indexing="ij"),axis=-1);flat=points.reshape(-1,3)
        distances=np.full(len(flat),np.inf)
        for sdf,p,R in components:
            d,_=sdf.query((flat-p)@R);distances=np.minimum(distances,d)
        grid=MeshSDF(axes,distances.reshape(points.shape[:-1]).astype(np.float32))
        return cls(body,mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body),grid,
                   {"components":meta,"voxel_m":voxel,"padding_m":padding,
                    "distance":"min of component SDFs; exact exterior union distance, approximate inside overlaps"})
    def query_world(self,points,data):
        R=data.xmat[self.body_id].reshape(3,3)
        p=data.xpos[self.body_id]
        d,n=self.grid.query((points-p)@R)
        return d,n@R.T
