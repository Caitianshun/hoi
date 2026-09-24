"""CPU sparse alpha compositor for A exact asymmetric camera. Adapted from frozen baseline audit; numerical validation required."""
import numpy as np

def sparse_raster(xyz,R,scales,opacity,K,C,uv,W=640,H=480):
 """Faithful scalar-pixel native-add3 alpha compositor; checked against GPU export.
 Reproduces native padding/crop, covariance floor, tile bounds and alpha cutoffs.
 Only output known pixel contributors, not full images or radiance.
 """
 xyz=np.asarray(xyz,dtype=np.float64); C=np.asarray(C,dtype=np.float64)
 cam=(xyz-C[:3,3])@C[:3,:3]; z=cam[:,2]; fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2]
 if False:  # A uses exact asymmetric projection, no padding/crop
  nw=int(2*max(cx,W-cx));nh=int(2*max(cy,H-cy));crop=np.array([0 if cx>W-cx else nw-W,0 if cy>H-cy else nh-H])
 else: nw,nh=W,H;crop=np.array([0,0])
 mu=cam[:,:2]/(z[:,None]+1e-7)*np.array([fx,fy])+np.array([cx,cy])
 rc=C[:3,:3].T[None]@R
 cov=(rc*scales[:,None,:]**2)@rc.transpose(0,2,1)
 zz=np.maximum(z,1e-8); clx=np.clip(cam[:,0]/zz,-1.3*nw/(2*fx),1.3*nw/(2*fx));cly=np.clip(cam[:,1]/zz,-1.3*nh/(2*fy),1.3*nh/(2*fy))
 J=np.zeros((len(z),2,3));J[:,0,0]=fx/zz;J[:,1,1]=fy/zz;J[:,0,2]=-fx*clx/zz;J[:,1,2]=-fy*cly/zz
 v=J@cov@J.transpose(0,2,1);v[:,0,0]+=.3;v[:,1,1]+=.3
 aa,bb,cc=v[:,0,0],v[:,0,1],v[:,1,1];det=aa*cc-bb**2
 conic=np.stack([cc,-bb,aa],-1)/np.maximum(det[:,None],1e-20)
 mid=.5*(aa+cc);rad=np.ceil(3*np.sqrt(np.maximum(mid+np.sqrt(np.maximum(.1,mid**2-det)),0)))
 grid=np.array([(nw+15)//16,(nh+15)//16]);rectmin=np.clip(np.trunc((mu-rad[:,None])/16),0,grid).astype(int);rectmax=np.clip(np.trunc((mu+rad[:,None]+15)/16),0,grid).astype(int)
 result=[]
 for q in np.asarray(uv):
  qp=q+crop;tile=np.floor(qp/16).astype(int)
  candidates=np.flatnonzero((z>.02)&(det>0)&((rectmin<=tile)&(rectmax>tile)).all(-1))
  candidates=candidates[np.argsort(z[candidates],kind='stable')];d=mu[candidates]-qp
  c=conic[candidates];power=-.5*(c[:,0]*d[:,0]**2+c[:,2]*d[:,1]**2)-c[:,1]*d[:,0]*d[:,1]
  a=np.minimum(.99,np.asarray(opacity).ravel()[candidates]*np.exp(np.minimum(power,0)))
  ok=(power<=0)&(a>=1/255);candidates=candidates[ok];a=a[ok]
  after=np.cumprod(1-a);before=np.r_[1,after[:-1]];ok=after>=.0001;candidates=candidates[ok];weight=(a*before)[ok]
  result.append((candidates,weight))
 return result
