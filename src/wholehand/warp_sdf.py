"""NVIDIA Warp signed distance grid construction from public CAD geometry."""
import numpy as np
import warp as wp

from .geometry import MeshSDF

@wp.kernel
def mesh_distances(mesh:wp.uint64,points:wp.array(dtype=wp.vec3),values:wp.array(dtype=float)):
    i=wp.tid();p=points[i]
    q=wp.mesh_query_point_sign_winding_number(mesh,p,100.)
    if q.result:
        nearest=wp.mesh_eval_position(mesh,q.face,q.u,q.v)
        values[i]=wp.length(p-nearest)*q.sign
    else:values[i]=100.

def make_grid(mesh,voxel_size=.002,padding=.03,device='cuda:0'):
    wp.init()
    points=wp.array(np.asarray(mesh.vertices,np.float32),dtype=wp.vec3,device=device)
    indices=wp.array(np.asarray(mesh.faces,np.int32).ravel(),dtype=wp.int32,device=device)
    wm=wp.Mesh(points=points,indices=indices,support_winding_number=True)
    bounds=mesh.bounds+np.array([[-padding]*3,[padding]*3])
    axes=[np.linspace(a,b,int(np.ceil((b-a)/voxel_size))+1) for a,b in bounds.T]
    xyz=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1)
    query=wp.array(xyz.reshape(-1,3).astype(np.float32),dtype=wp.vec3,device=device)
    values=wp.zeros(len(query),dtype=float,device=device)
    wp.launch(mesh_distances,dim=len(query),inputs=[wm.id,query,values],device=device);wp.synchronize_device(device)
    distances=values.numpy().reshape(xyz.shape[:-1]).astype(np.float64)
    return MeshSDF(axes,distances)
