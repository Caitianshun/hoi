"""Fixed-template, locally constrained shared surface attachments (CPU geometry).

No observation, reference, rendering, GPU, or file data dependencies. A fixed
portal graph supplies conservative surface-path certificates. The graph is
rebuilt only at construction around original q0, never around updated points.
Torch differentiation is confined to current triangle barycentric coordinates;
chart changes happen explicitly after optimizer steps.
"""
from __future__ import annotations
import copy
import heapq
import math
import numpy as np

EPS = 1e-10


def barycentric(point, triangle):
    """Affine coordinates in the triangle plane; not clamped."""
    A = (triangle[1:] - triangle[0]).T
    uv = np.linalg.lstsq(A, np.asarray(point) - triangle[0], rcond=None)[0]
    return np.array([1. - uv.sum(), uv[0], uv[1]])


def _rotation_between_faces(vector, edge_unit, normal_a, normal_b):
    # Orient normals consistently around the shared edge, even for meshes with
    # inconsistent input winding. The desired neighbor interior is checked by
    # the caller, so the two possible folds are selected geometrically.
    angle = math.atan2(np.dot(edge_unit, np.cross(normal_a, normal_b)),
                       np.dot(normal_a, normal_b))
    c, s = math.cos(angle), math.sin(angle)
    return vector*c + np.cross(edge_unit, vector)*s + edge_unit*np.dot(edge_unit, vector)*(1-c)


class SurfaceConstraintSet:
    """One immutable surface neighborhood and mutable chart per shared q.

    Parameters use metres. initial_bary is [N,3], in face-vertex order.
    Optimizable bary2 [N,2] contains weights of face vertices 1 and 2;
    weight 0 is 1-bary2.sum(). Faces and mesh arrays are never modified.

    project_step is a mutation: call ONCE after a proposed optimizer step.
    Passing the same current values is a zero move, suitable for validation.
    clone() shares immutable geometry/neighborhoods but resets mutable state.
    """
    def __init__(self, vertices, faces, initial_face, initial_bary,
                 radius_m=.05, portal_spacing_m=.005, initial_q=None):
        self.vertices = np.array(vertices, dtype=np.float64, copy=True)
        self.faces = np.array(faces, dtype=np.int64, copy=True)
        self.initial_face = np.atleast_1d(initial_face).astype(np.int64).copy()
        self.initial_bary = np.atleast_2d(initial_bary).astype(np.float64).copy()
        self.radius_m = float(radius_m)
        self.portal_spacing_m = float(portal_spacing_m)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError('vertices [V,3] and triangular faces [F,3] required')
        if not np.isfinite(self.vertices).all() or self.faces.min(initial=0) < 0 or self.faces.max(initial=0) >= len(self.vertices):
            raise ValueError('invalid mesh arrays')
        if self.initial_bary.shape != (len(self.initial_face), 3) or not np.isfinite(self.initial_bary).all():
            raise ValueError('initial_bary must be finite [N,3]')
        if np.any(self.initial_face < 0) or np.any(self.initial_face >= len(self.faces)):
            raise ValueError('invalid initial faces')
        if np.min(self.initial_bary, initial=0) < -1e-7 or not np.allclose(self.initial_bary.sum(1), 1., atol=1e-7):
            raise ValueError('initial barycentric coordinates outside surface')
        self.initial_bary = np.maximum(self.initial_bary, 0.)
        self.initial_bary /= self.initial_bary.sum(1, keepdims=True)
        if self.radius_m <= 0 or self.portal_spacing_m <= 0:
            raise ValueError('positive radius/portal spacing required')
        self.triangles = self.vertices[self.faces]
        normal = np.cross(self.triangles[:,1]-self.triangles[:,0], self.triangles[:,2]-self.triangles[:,0])
        area2 = np.linalg.norm(normal, axis=1)
        self.valid_face = area2 > 1e-14
        if not self.valid_face[self.initial_face].all():
            raise ValueError('initial face is degenerate')
        self.normals = normal / np.maximum(area2[:,None], 1e-30)
        self.initial_q = np.einsum('ni,nij->nj', self.initial_bary, self.triangles[self.initial_face])
        if initial_q is not None and not np.allclose(np.atleast_2d(initial_q), self.initial_q, atol=1e-6, rtol=0):
            raise ValueError('initial q must equal fixed face/bary point')
        edge_map = {}
        self.face_edges = np.full((len(self.faces),3), -1, dtype=np.int64)
        edge_vertices=[];edge_faces=[]
        for fi, face in enumerate(self.faces):
            if not self.valid_face[fi]:continue
            for opp in range(3):
                e=tuple(sorted([int(face[(opp+1)%3]),int(face[(opp+2)%3])]))
                if e not in edge_map:
                    edge_map[e]=len(edge_vertices);edge_vertices.append(e);edge_faces.append([])
                ei=edge_map[e];edge_faces[ei].append(fi);self.face_edges[fi,opp]=ei
        self.edge_vertices=np.asarray(edge_vertices,dtype=np.int64).reshape(-1,2)
        self.edge_faces=edge_faces
        self.edge_a=self.vertices[self.edge_vertices[:,0]]
        self.edge_b=self.vertices[self.edge_vertices[:,1]]
        self.neighbor=np.full((len(self.faces),3),-1,dtype=np.int64)
        for f in range(len(self.faces)):
            for opp,ei in enumerate(self.face_edges[f]):
                if ei>=0 and len(edge_faces[ei])==2:self.neighbor[f,opp]=next(x for x in edge_faces[ei] if x!=f)
        self.neighborhoods=[self._build_neighborhood(i) for i in range(len(self.initial_face))]
        for a in [self.vertices,self.faces,self.initial_face,self.initial_bary,self.initial_q]:a.flags.writeable=False
        self.reset()

    def reset(self):
        self.face=self.initial_face.copy();self.bary=self.initial_bary.copy()
        n=len(self.face);self.total_move_m=np.zeros(n);self.face_switch_count=np.zeros(n,dtype=int)
        self.boundary_hit_count=np.zeros(n,dtype=int);self.update_count=0
        return self

    def clone(self):
        """Reuse prebuilt fixed neighborhoods, with fresh mutable q state."""
        obj=copy.copy(self);return obj.reset()

    @property
    def bary2(self):return self.bary[:,1:].copy()

    def points_numpy(self, bary2=None):
        b=self.bary if bary2 is None else np.c_[1.-np.asarray(bary2).sum(1),np.asarray(bary2)]
        return np.einsum('ni,nij->nj',b,self.triangles[self.face])

    def torch_points(self, bary2):
        """Autograd-safe q[N,3] on CURRENT faces; bary2 is the optimizer tensor."""
        import torch
        tri=torch.as_tensor(self.triangles[self.face],dtype=bary2.dtype,device=bary2.device)
        return tri[:,0]+bary2[:,0,None]*(tri[:,1]-tri[:,0])+bary2[:,1,None]*(tri[:,2]-tri[:,0])

    def _build_neighborhood(self, k):
        q0=self.initial_q[k];radius=self.radius_m;ab=self.edge_b-self.edge_a
        ll=np.sum(ab*ab,axis=1);u=np.sum((q0-self.edge_a)*ab,axis=1)/np.maximum(ll,1e-30)
        near=self.edge_a+np.clip(u,0,1)[:,None]*ab
        candidate=np.flatnonzero(np.linalg.norm(near-q0,axis=1)<=radius+EPS)
        nodes=[q0.copy()];node_faces=[[int(self.initial_face[k])]]
        memberships={int(self.initial_face[k]):[0]}
        # All graph edges are straight segments inside one fixed triangle.
        # Coincident spatial points with different topology are NOT merged.
        for ei in candidate:
            if ll[ei]<=1e-24:continue
            foot=self.edge_a[ei]+u[ei]*ab[ei]
            delta=max(0.,radius**2-float(np.sum((foot-q0)**2)))
            half=math.sqrt(delta/ll[ei]);lo=max(0.,u[ei]-half);hi=min(1.,u[ei]+half)
            if hi<lo-EPS:continue
            count=max(1,int(math.ceil((hi-lo)*math.sqrt(ll[ei])/self.portal_spacing_m)))
            values=np.unique(np.r_[np.linspace(lo,hi,count+1),np.clip(u[ei],lo,hi)])
            # Nonmanifold shared edges are conservatively split per face.
            groups=[self.edge_faces[ei]] if len(self.edge_faces[ei])<=2 else [[f] for f in self.edge_faces[ei]]
            for group in groups:
                for v in values:
                    idx=len(nodes);nodes.append(self.edge_a[ei]+v*ab[ei]);node_faces.append(group)
                    for f in group:memberships.setdefault(int(f),[]).append(idx)
        nodes=np.asarray(nodes);dist=np.full(len(nodes),np.inf);dist[0]=0
        prev=np.full(len(nodes),-1,int);prev_face=np.full(len(nodes),-1,int);heap=[(0.,0)]
        while heap:
            d,i=heapq.heappop(heap)
            if d!=dist[i] or d>radius+EPS:continue
            for f in node_faces[i]:
                js=np.asarray(memberships[int(f)],dtype=int)
                dd=d+np.linalg.norm(nodes[js]-nodes[i],axis=1)
                for j,nd in zip(js,dd):
                    if nd<dist[j]-1e-13 and nd<=radius+EPS:
                        dist[j]=nd;prev[j]=i;prev_face[j]=int(f);heapq.heappush(heap,(float(nd),int(j)))
        by_face={}
        for f,ids in memberships.items():
            ids=np.asarray(ids);ids=ids[np.isfinite(dist[ids])]
            if len(ids):by_face[f]={'nodes':nodes[ids], 'distance':dist[ids], 'node_ids':ids}
        return {'by_face':by_face,'nodes':nodes,'distance':dist,'predecessor':prev,'predecessor_face':prev_face,
                'node_count':len(nodes),'reachable_nodes':int(np.isfinite(dist).sum()),'reachable_faces':len(by_face)}

    def surface_distance_bound(self,k,face,q):
        """Length of an explicit q0->portal graph path->q; +inf if absent."""
        seeds=self.neighborhoods[k]['by_face'].get(int(face))
        if seeds is None:return float('inf')
        return float(np.min(seeds['distance']+np.linalg.norm(seeds['nodes']-q,axis=1)))

    def path_certificate(self,k,face=None,q=None):
        """Polyline and segment face IDs proving a legal path from ORIGINAL q0."""
        face=int(self.face[k] if face is None else face)
        q=self.points_numpy()[k] if q is None else np.asarray(q)
        nb=self.neighborhoods[k];s=nb['by_face'].get(face)
        if s is None:raise ValueError('face outside fixed neighborhood')
        local=int(np.argmin(s['distance']+np.linalg.norm(s['nodes']-q,axis=1)))
        node=int(s['node_ids'][local]);ids=[];fs=[]
        while node>=0:
            ids.append(node)
            if nb['predecessor'][node]>=0:fs.append(int(nb['predecessor_face'][node]))
            node=int(nb['predecessor'][node])
        pts=np.vstack([nb['nodes'][ids[::-1]],q]);segment_faces=fs[::-1]+[face]
        return {'points':pts,'segment_faces':np.asarray(segment_faces,int),'length_m':float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum())}

    def _clip_neighborhood_segment(self,k,face,start,end):
        """Follow the connected legal interval from start through a union of disks."""
        d=end-start;A=float(d@d)
        if A<1e-26:return end.copy(),False
        seeds=self.neighborhoods[k]['by_face'].get(int(face))
        if seeds is None:return start.copy(),True
        intervals=[]
        for c,plen in zip(seeds['nodes'],seeds['distance']):
            r=max(0.,self.radius_m-float(plen));v=start-c;B=2*float(v@d);C=float(v@v)-r*r;disc=B*B-4*A*C
            if disc < -1e-16:continue
            root=math.sqrt(max(0.,disc));lo=max(0.,(-B-root)/(2*A));hi=min(1.,(-B+root)/(2*A))
            if hi>=lo-1e-10:intervals.append((lo,hi))
        covered=0.
        for lo,hi in sorted(intervals):
            if lo>covered+1e-9:break
            covered=max(covered,hi)
        if covered>=1-1e-10:return end.copy(),False
        return start+max(0.,min(1.,covered))*d,True

    @staticmethod
    def _first_edge(bstart,bend):
        ids=np.flatnonzero(bend < -1e-12)
        if not len(ids):return 1.,None
        choices=[]
        for j in ids:
            den=bstart[j]-bend[j]
            if den>1e-15:choices.append((max(0.,float(bstart[j]/den)),int(j)))
        if not choices:return 0.,int(ids[0])
        return min(choices)

    def project_step(self,proposed_bary2):
        """Project a batch proposed chart update; at most ONE neighbor per q.

        Returns dict with bary2/face/q and per-point diagnostics. A boundary hit
        includes fixed neighborhood, mesh boundary, or second-edge truncation.
        Non-finite proposals raise; they are never silently turned into zero.
        """
        proposal=np.asarray(proposed_bary2,dtype=float)
        if proposal.shape!=(len(self.face),2) or not np.isfinite(proposal).all():raise ValueError('finite proposed_bary2 [N,2] required')
        oldq=self.points_numpy();oldface=self.face.copy();hits=np.zeros(len(self.face),bool);moves=np.zeros(len(self.face));reasons=[]
        for k in range(len(self.face)):
            f=int(self.face[k]);b=self.bary[k];bend=np.r_[1-proposal[k].sum(),proposal[k]];end=bend@self.triangles[f]
            frac,opp=self._first_edge(b,bend);edgeq=oldq[k]+frac*(end-oldq[k]);mid,hit=self._clip_neighborhood_segment(k,f,oldq[k],edgeq)
            moves[k]=np.linalg.norm(mid-oldq[k]);q=mid;reason=[]
            if hit:reason.append('fixed_neighborhood')
            if opp is not None and not hit:
                neighbor=int(self.neighbor[f,opp])
                if neighbor<0:hit=True;reason.append('mesh_boundary_or_nonmanifold')
                else:
                    ei=self.face_edges[f,opp];axis=self.edge_b[ei]-self.edge_a[ei];axis/=np.linalg.norm(axis)
                    remain=end-edgeq
                    rotated=_rotation_between_faces(remain,axis,self.normals[f],self.normals[neighbor])
                    nb=barycentric(edgeq,self.triangles[neighbor]);target=edgeq+rotated
                    nbend=barycentric(target,self.triangles[neighbor])
                    # Winding may differ: choose the fold into neighbor interior.
                    shared=set(self.edge_vertices[ei]);nopp=next(j for j,v in enumerate(self.faces[neighbor]) if int(v) not in shared)
                    if nbend[nopp]<-1e-10:
                        rotated=_rotation_between_faces(remain,axis,self.normals[f],-self.normals[neighbor]);target=edgeq+rotated;nbend=barycentric(target,self.triangles[neighbor])
                    frac2,opp2=self._first_edge(np.maximum(nb,0),nbend)
                    neighbor_end=edgeq+frac2*(target-edgeq)
                    nxt,hit2=self._clip_neighborhood_segment(k,neighbor,edgeq,neighbor_end)
                    # If the fixed conservative neighbor offers no nonzero continuation,
                    # remain on old chart boundary rather than create a false switch.
                    if np.linalg.norm(nxt-edgeq)>1e-12:
                        q=nxt;f=neighbor;moves[k]+=np.linalg.norm(nxt-edgeq)
                    if hit2:hit=True;reason.append('fixed_neighborhood')
                    if opp2 is not None:hit=True;reason.append('one_neighbor_step_limit')
            bary=barycentric(q,self.triangles[f]);bary[np.abs(bary)<1e-9]=0
            if np.min(bary)<-1e-7:raise AssertionError('projection left triangle')
            bary=np.maximum(bary,0);bary/=bary.sum();self.face[k]=f;self.bary[k]=bary
            bound=self.surface_distance_bound(k,f,bary@self.triangles[f])
            if bound>self.radius_m+1e-7:raise AssertionError(('projection outside fixed path neighborhood',bound))
            hits[k]=hit;reasons.append(reason)
        switched=self.face!=oldface
        self.total_move_m+=moves;self.face_switch_count+=switched;self.boundary_hit_count+=hits;self.update_count+=1
        q=self.points_numpy();bound=np.array([self.surface_distance_bound(k,self.face[k],q[k]) for k in range(len(q))])
        return {'bary2':self.bary2,'projected_bary2':self.bary2,'bary':self.bary.copy(),'face':self.face.copy(),'q':q,'boundary_hit':hits,
                'boundary_reason':reasons,'face_changed':switched,'previous_face':oldface,'step_move_m':moves,
                'surface_distance_bound_m':bound,'displacement_from_initial_m':np.linalg.norm(q-self.initial_q,axis=1),
                'total_move_m':self.total_move_m.copy(),'face_switch_count':self.face_switch_count.copy()}

    def project_torch_(self,parameter,optimizer=None):
        """In-place post-step projection. Reset optimizer chart moments on switches."""
        import torch
        result=self.project_step(parameter.detach().cpu().numpy())
        with torch.no_grad():parameter.copy_(torch.as_tensor(result['bary2'],dtype=parameter.dtype,device=parameter.device))
        if optimizer is not None and result['face_changed'].any():
            changed=torch.as_tensor(result['face_changed'],device=parameter.device)
            for value in optimizer.state.get(parameter,{}).values():
                if torch.is_tensor(value) and value.shape==parameter.shape:value[changed]=0
        return result

    def state_np(self):
        q=self.points_numpy()
        return {'face':self.face.copy(),'bary':self.bary.copy(),'q':q,'initial_face':self.initial_face.copy(),
                'initial_bary':self.initial_bary.copy(),'initial_q':self.initial_q.copy(),
                'total_move_m':self.total_move_m.copy(),'face_switch_count':self.face_switch_count.copy(),
                'boundary_hit_count':self.boundary_hit_count.copy(),'update_count':self.update_count,
                'surface_distance_bound_m':np.array([self.surface_distance_bound(k,self.face[k],q[k]) for k in range(len(q))])}

    def metadata(self):
        return {'radius_m':self.radius_m,'portal_spacing_m':self.portal_spacing_m,'center':'original initial_q, never reset',
                'distance':'conservative shortest graph path through fixed edge portals plus straight final segment in current triangle',
                'local_set':'union over reachable portals of in-triangle disks of radius (0.05 - fixed graph distance), plus original-face disk',
                'portal_rule':'all edge intervals inside Euclidean q0 radius ball, uniform spacing <= portal_spacing, plus q0 nearest point',
                'topology':'only same-face straight segments and shared edges; no nearest-surface jumps; nonmanifold edges not crossed',
                'max_neighbor_crossings_per_step':1,'mesh_vertices_preserved':len(self.vertices),'mesh_faces_preserved':len(self.faces),
                'neighborhoods':[{k:n[k] for k in ['node_count','reachable_nodes','reachable_faces']} for n in self.neighborhoods]}


def self_test():
    """Small topology, distance, sparse-face and autograd tests, CPU only."""
    import torch
    results={}
    # Huge sparse triangle: local movement must not require reaching vertices.
    v=np.array([[0.,0,0],[1,0,0],[0,1,0],[9,9,9]])
    m=SurfaceConstraintSet(v,[[0,1,2]],[0],[[.6,.2,.2]])
    initial=m.initial_q.copy()
    r=m.project_step([[.4,.2]])
    assert abs(r['step_move_m'][0]-.05)<1e-8
    assert r['boundary_hit'][0] and len(m.vertices)==4
    assert np.array_equal(m.initial_q,initial)
    for i in range(5):m.project_step([[.6,.2]])
    assert np.linalg.norm(m.points_numpy()[0]-initial[0])<=.05+1e-9
    results['sparse_large_triangle_moves_5cm_not_frozen']=True
    # Two coplanar faces, with disconnected duplicate surface 0.1 mm away.
    v=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,.0001],[1,0,.0001],[0,1,.0001],[1,1,.0001]])
    f=np.array([[0,1,2],[1,3,2],[4,5,6],[5,7,6]])
    m=SurfaceConstraintSet(v,f,[0],[[.02,.49,.49]])
    r=m.project_step([[.52,.52]])
    assert r['face_changed'][0] and r['face'][0]==1
    assert np.allclose(r['q'][0],[.52,.52,0],atol=1e-9)
    assert set(m.neighborhoods[0]['by_face']).issubset({0,1})
    cert=m.path_certificate(0)
    assert cert['length_m']<=.05+1e-8
    for a,b,fi in zip(cert['points'][:-1],cert['points'][1:],cert['segment_faces']):
        assert min(barycentric(a,m.triangles[fi]))>=-1e-7
        assert min(barycentric(b,m.triangles[fi]))>=-1e-7
    results['adjacent_face_switch_and_no_disconnected_sheet_jump']=True
    results['surface_path_certificate_legal_and_below_5cm']=True
    # Reverse winding preserves crossing behavior.
    m=SurfaceConstraintSet(v,[[0,1,2],[1,2,3]],[0],[[.02,.49,.49]])
    r=m.project_step([[.52,.52]])
    assert r['face'][0]==1 and np.allclose(r['q'][0],[.52,.52,0],atol=1e-8)
    results['inconsistent_winding_handled']=True
    # Folded neighbor: cross xy triangle onto xz triangle, continuously on mesh.
    vf=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,-1.]])
    m=SurfaceConstraintSet(vf,[[0,1,2],[1,0,3]],[0],[[.49,.5,.01]])
    r=m.project_step([[.5,-.01]])
    assert r['face'][0]==1 and np.allclose(r['q'][0],[.5,0,-.01],atol=1e-8)
    assert r['surface_distance_bound_m'][0]<=.05+1e-8
    results['folded_face_surface_update']=True
    # Dense strip: very long proposed step truncates at second edge.
    vs=np.array([[x,y,0.] for x in np.arange(5)*.01 for y in [0,.01]])
    fs=[]
    for j in range(4):fs.extend([[2*j,2*j+2,2*j+1],[2*j+2,2*j+3,2*j+1]])
    m=SurfaceConstraintSet(vs,fs,[0],[[.5,.25,.25]])
    r=m.project_step([[7.,.25]])
    assert r['face_changed'][0] and r['face'][0] in m.neighbor[0]
    assert r['face_switch_count'][0]==1 and r['boundary_hit'][0]
    assert 'one_neighbor_step_limit' in r['boundary_reason'][0]
    results['one_step_crosses_at_most_one_adjacent_face']=True
    # Repeated random motion never shifts neighborhood center or exits surface.
    rng=np.random.default_rng(12345)
    for i in range(80):
        oldface=m.face.copy();r=m.project_step(m.bary2+rng.normal(size=(1,2))*.6)
        assert r['surface_distance_bound_m'][0]<=.05+1e-7
        assert min(r['bary'][0])>=-1e-10 and abs(r['bary'].sum()-1)<1e-10
        assert r['face'][0]==oldface[0] or r['face'][0] in m.neighbor[oldface[0]]
    results['repeated_updates_preserve_fixed_neighborhood']=True
    # Differentiability of the current face point, independently finite differenced.
    m=SurfaceConstraintSet(v,[[0,1,2],[1,3,2]],[0],[[.02,.49,.49]])
    x=torch.tensor(m.bary2,dtype=torch.float64,requires_grad=True)
    q=m.torch_points(x);loss=(q*torch.tensor([[2.,3.,5.]],dtype=torch.float64)).sum();loss.backward()
    grad=x.grad.numpy().copy();fd=np.empty_like(grad)
    for j in range(2):
        plus=m.bary2;minus=m.bary2;plus[0,j]+=1e-6;minus[0,j]-=1e-6
        fd[0,j]=float(((m.points_numpy(plus)-m.points_numpy(minus))*np.array([2,3,5])).sum()/(2e-6))
    assert np.max(abs(grad-fd))<1e-8
    m.project_step([[.52,.52]])
    z=torch.tensor(m.bary2,dtype=torch.float64)
    assert np.allclose(m.torch_points(z).numpy(),m.points_numpy())
    results['torch_gradient_matches_finite_difference']=True
    results['torch_points_reads_updated_current_face_every_call']=True
    results['all_passed']=True
    return results

if __name__=='__main__':
    import json
    print(json.dumps(self_test(),indent=2))
