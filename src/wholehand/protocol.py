"""Paper simulation protocol utilities. No RGB-policy integration or contact training."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import ConvexHull,QhullError
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

LOAD_AXES=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],float)

def hull_intersects_box(points,lo,hi,tolerance=1e-9):
    """Closed convex fingertip/palm hull versus closed axis-aligned box.

    Caller transforms the hull into object coordinates for an oriented CAD box.
    SAT uses all hull faces, box faces and hull-edge/box-edge cross products.
    Degenerate hulls use a bounded convex-combination feasibility problem.
    """
    p=np.asarray(points,float);lo=np.asarray(lo,float);hi=np.asarray(hi,float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or np.any(hi<lo):raise ValueError('finite hull and ordered box required')
    if not len(p):return False
    if np.any(p.max(0)<lo-tolerance) or np.any(p.min(0)>hi+tolerance):return False
    try:
        hull=ConvexHull(p)
    except QhullError:
        r=linprog(np.zeros(len(p)),A_ub=np.r_[p.T,-p.T],b_ub=np.r_[hi+tolerance,-lo+tolerance],A_eq=np.ones((1,len(p))),b_eq=[1],bounds=(0,None),method='highs')
        if r.status not in [0,2]:raise RuntimeError('hull-box feasibility indeterminate')
        return r.status==0
    edges=set()
    for face in hull.simplices:
        for a,b in [(face[0],face[1]),(face[1],face[2]),(face[2],face[0])]:edges.add(tuple(sorted((a,b))))
    directions=np.array([p[b]-p[a] for a,b in sorted(edges)])
    cross=np.cross(directions[:,None,:],np.eye(3)[None,:,:]).reshape(-1,3)
    axes=np.r_[np.eye(3),hull.equations[:,:3],cross]
    norm=np.linalg.norm(axes,axis=1);axes=axes[norm>1e-12]/norm[norm>1e-12,None]
    q=p@axes.T;center=(lo+hi)/2;half=(hi-lo)/2
    middle=axes@center;radius=np.abs(axes)@half
    return bool(not np.any((q.max(0)<middle-radius-tolerance)|(q.min(0)>middle+radius+tolerance)))

def grasp_is_lost(hand_points_world,object_position,object_rotation,object_bounds):
    local=(np.asarray(hand_points_world)-object_position)@object_rotation
    return not hull_intersects_box(local,*np.asarray(object_bounds))

def sampled_directions(seed,count):
    v=np.random.default_rng(seed).normal(size=(count,3));return v/np.linalg.norm(v,axis=1)[:,None]

def perturbed_pose(position,rotation,translation_m,rotation_deg,translation_axis,rotation_axis):
    """Constant world-frame offsets, fixed for a trial; convention is disclosed."""
    return np.asarray(position)+translation_m*np.asarray(translation_axis),Rotation.from_rotvec(np.deg2rad(rotation_deg)*np.asarray(rotation_axis)).as_matrix()@rotation

def grasp_level_statistics(successes,grasp_ids):
    """Initial failures included; standard error across grasp-level means."""
    x=np.asarray(successes,float);ids=np.asarray(grasp_ids)
    if len(x)!=len(ids) or not len(x) or not np.isfinite(x).all():raise ValueError('complete trial outcomes required')
    rates=np.array([x[ids==g].mean() for g in np.unique(ids)])
    return {'trial_count':len(x),'grasp_count':len(rates),'grasp_mean_success_rate':float(rates.mean()),'trial_mean_success_rate':float(x.mean()),'grasp_standard_error':float(rates.std(ddof=1)/np.sqrt(len(rates))) if len(rates)>1 else None}

@dataclass(frozen=True)
class Schedule:
    # These durations/gains are not reported by the authors. Keep explicit.
    settle_s:float=1.
    perturb_s:float=1.5
    release_s:float=.5
    @property
    def first_load_start(self):return self.settle_s
    @property
    def perturb_start(self):return self.settle_s+6.
    @property
    def second_load_start(self):return self.perturb_start+self.perturb_s+self.release_s
    @property
    def end(self):return self.second_load_start+6.
    def load(self,t,mass):
        for begin in [self.first_load_start,self.second_load_start]:
            if begin<=t<begin+6:return .1*mass*9.81*LOAD_AXES[min(int(t-begin),5)]
        return np.zeros(3)
