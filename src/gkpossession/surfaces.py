"""Frozen expected remaining shot-xG and possession-start threat surfaces."""
import numpy as np

def zone(x,y,grid=(16,12)):
    gx,gy=grid;return np.clip((np.asarray(x)/120*gx).astype(int),0,gx-1)*gy+np.clip((np.asarray(y)/80*gy).astype(int),0,gy-1)

def fit_surface(count,total,k=30):
    count=np.asarray(count);total=np.asarray(total);mean=total.sum()/max(count.sum(),1)
    return (total+k*mean)/(count+k)

def fit_bundle(frame,matches,grid=(16,12),k=30):
    g=frame[frame.match_id.isin(matches)].copy();n=np.prod(grid);z=zone(g.x,g.y,grid)
    counts=np.bincount(z,weights=g.own.astype(float),minlength=n);sums=np.bincount(z,weights=g.rem*g.own,minlength=n);v=fit_surface(counts,sums,k)
    a=g[g.start & g.own];tcount=np.bincount(zone(a.x,a.y,grid),minlength=n);tsum=np.bincount(zone(a.x,a.y,grid),weights=a.poss_xg,minlength=n);t=fit_surface(tcount,tsum,k)
    vp=[]
    for p in [0,1]:
        mask=(g.own & (g.pressure==p)).to_numpy();c=np.bincount(z[mask],minlength=n);s=np.bincount(z[mask],weights=g.rem.to_numpy()[mask],minlength=n)
        vp.append((s+60*v)/(c+60))
    return {'grid':grid,'k':k,'V':v,'Vp':np.stack(vp,axis=1),'T':t,'count':counts,'total':sums,'t_count':tcount,'t_total':tsum}

def lookup(surface,x,y,name='V',pressure=None):
    z=zone(x,y,surface['grid'])
    return surface['Vp'][z,np.asarray(pressure,dtype=int)] if pressure is not None else surface[name][z]
