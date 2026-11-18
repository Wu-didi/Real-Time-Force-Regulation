"""Deployment hand FK using robot TCP/joint state only. No simulator/object inputs."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def quaternion_matrix(wxyz):
    q=np.asarray(wxyz,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>.01:
        raise ValueError("robot TCP quaternion must be normalized wxyz")
    return Rotation.from_quat(q[[1,2,3,0]]).as_matrix()

class HandKinematics:
    def __init__(self,metadata,points_local,normals_local,point_link):
        self.metadata=metadata
        self.points_local=np.asarray(points_local,dtype=np.float64)
        self.normals_local=np.asarray(normals_local,dtype=np.float64)
        self.point_link=np.asarray(point_link,dtype=np.int32)
        self.body_order=[b["id"] for b in metadata["bodies"]]
        self.body_specs={b["id"]:b for b in metadata["bodies"]}
        self.link_bodies=np.array(metadata["link_bodies"],dtype=int)
        self.point_body=self.link_bodies[self.point_link]
        if len(self.points_local)!=len(self.point_link):raise ValueError("surface sample schema")
        self.num_links=len(self.link_bodies)
        self.num_points=len(self.point_link)

    @property
    def schema_sha256(self):
        digest=hashlib.sha256(json.dumps(self.metadata,sort_keys=True,separators=(",",":")).encode())
        for a in (self.points_local,self.normals_local,self.point_link):
            digest.update(str(a.dtype).encode());digest.update(a.tobytes())
        return digest.hexdigest()

    def forward(self,robot_state):
        state=np.asarray(robot_state,dtype=float)
        if state.shape!=(46,) or not np.isfinite(state).all():
            raise ValueError("native robot-only state must be exactly46D; object poses forbidden")
        transforms={}
        for hand in self.metadata["hands"]:
            tcp=state[hand["tcp_slice"][0]:hand["tcp_slice"][1]]
            R=quaternion_matrix(tcp[3:7])
            transforms[hand["root_body"]]=(tcp[:3]+R@np.asarray(hand["palm_in_tcp_pos"]),
                                          R@np.asarray(hand["palm_in_tcp_rot"]))
        for bid in self.body_order:
            if bid in transforms:continue
            spec=self.body_specs[bid]
            parent_pos,parent_rot=transforms[spec["parent"]]
            position=parent_pos+parent_rot@np.asarray(spec["pos"])
            rotation=parent_rot@np.asarray(spec["rot"])
            for j in spec["joints"]:
                angle=state[j["state_index"]]-j["reference"]
                after=rotation@Rotation.from_rotvec(np.array(j["axis"])*angle).as_matrix()
                anchor=np.asarray(j["pos"])
                position+=rotation@anchor-after@anchor
                rotation=after
            transforms[bid]=(position,rotation)
        p=np.empty_like(self.points_local);n=np.empty_like(self.normals_local)
        link_R=np.empty((self.num_links,3,3));link_p=np.empty((self.num_links,3))
        for k,bid in enumerate(self.link_bodies):
            pos,rot=transforms[int(bid)]
            keep=self.point_link==k
            p[keep]=self.points_local[keep]@rot.T+pos
            n[keep]=self.normals_local[keep]@rot.T
            link_R[k]=rot;link_p[k]=pos
        return {"points_world":p,"normals_world":n,"link_rotation":link_R,
                "link_position":link_p,"point_link":self.point_link}

    def query_features(self,robot_state):
        fk=self.forward(robot_state)
        # meters retained in dataset; explicit scaling only at model inputs.
        features=np.concatenate((fk["points_world"],self.points_local/.1,
                                  fk["normals_world"]),axis=-1).astype(np.float32)
        return features,fk

    def save(self,path):
        path=Path(path)
        np.savez_compressed(path,metadata_json=np.array(json.dumps(self.metadata,sort_keys=True)),
              points_local=self.points_local,normals_local=self.normals_local,
              point_link=self.point_link,schema_sha256=np.array(self.schema_sha256))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as a:
            instance=cls(json.loads(str(a["metadata_json"])),a["points_local"],a["normals_local"],a["point_link"])
            if instance.schema_sha256!=str(a["schema_sha256"]):raise ValueError("hand geometry identity")
            return instance
