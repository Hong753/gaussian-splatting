"""Read-only size-branch and footprint diagnostics. No training policy changes."""
import math
import numpy as np

COLORS = np.array([[130,140,155], [35,190,150], [255,145,25], [220,65,195]], np.uint8)
LABELS = ('outside_raster', 'smaller_footprint', 'large_clone_size', 'large_split_size')


def training_extent(records, selected):
    """Reproduce getNerfppNorm using the actual selected camera centers."""
    from utils.graphics_utils import getWorld2View2
    centers = [np.linalg.inv(getWorld2View2(np.array(records[n]['R']), np.array(records[n]['T'])))[:3,3]
               for n in sorted(selected)]
    if not centers:
        raise ValueError('No training cameras for scene extent')
    centers = np.asarray(centers)
    return float(np.linalg.norm(centers-centers.mean(0), axis=1).max()*1.1)


def classify_sizes(max_sigma, radii, cutoff, large_radius):
    max_sigma, radii = np.asarray(max_sigma), np.asarray(radii)
    if (max_sigma.shape != radii.shape or max_sigma.ndim != 1 or
            not np.isfinite([cutoff,large_radius]).all() or cutoff < 0 or large_radius <= 0 or
            not np.isfinite(max_sigma).all() or not np.isfinite(radii).all() or (max_sigma <= 0).any() or (radii < 0).any()):
        raise ValueError('Invalid Gaussian scales or projected radii')
    labels = np.zeros(len(radii), np.uint8)
    labels[radii > 0] = 1
    large = radii > large_radius
    labels[large & (max_sigma <= cutoff)] = 2
    labels[large & (max_sigma > cutoff)] = 3
    return labels


def project_centers(points, view, projection, width, height, chunk=131072):
    """Native ndc2Pix convention; chunking bounds temporary allocations."""
    result = np.empty((len(points),3), np.float32)
    for start in range(0,len(points),chunk):
        xyz = points[start:start+chunk]
        clip = xyz@projection[:3,:] + projection[3,:]
        z = xyz@view[:3,2] + view[3,2]
        with np.errstate(divide='ignore', invalid='ignore'):
            result[start:start+len(xyz),0] = ((clip[:,0]/(clip[:,3]+1e-7)+1)*width-1)*.5
            result[start:start+len(xyz),1] = ((clip[:,1]/(clip[:,3]+1e-7)+1)*height-1)*.5
        result[start:start+len(xyz),2] = z
    return result


def roi_pixels(roi, width, height):
    x0,y0,x1,y1 = roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError('ROI must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1')
    return (int(x0*width),int(y0*height),min(width,math.ceil(x1*width)),min(height,math.ceil(y1*height)))


def footprint_roi(projected, radii, rectangle):
    """Conservative overlap of CUDA bounding squares, not an opacity visibility test."""
    u,v,z = np.asarray(projected).T
    x0,y0,x1,y1 = rectangle
    return ((radii > 0) & (z > .2) & np.isfinite(projected).all(1) &
            (u+radii >= x0) & (u-radii < x1) & (v+radii >= y0) & (v-radii < y1))


def center_frustum(projected, width, height):
    u,v,z=np.asarray(projected).T
    return (z > .2) & (u >= -.5) & (u < width-.5) & (v >= -.5) & (v < height-.5)


def size_report(max_sigma, radii, labels, mask, weights, rectangle):
    x0,y0,x1,y1=rectangle
    names=('large_clone_size','large_split_size','other')
    mass=weights[:,y0:y1,x0:x1].sum(axis=(1,2),dtype=np.float64)
    total=float(mass.sum())
    subset=np.asarray(radii)[mask]
    scales=np.asarray(max_sigma)[mask]
    return dict(roi_pixels=list(rectangle), raster_footprints_overlapping_roi=int(mask.sum()),
        size_branch_counts={name:int((mask & (labels==i)).sum()) for i,name in enumerate(LABELS)},
        radius_pixels_percentiles=dict(zip(('p50','p90','p99','max'),map(float,np.percentile(subset,[50,90,99,100])))) if len(subset) else {},
        max_sigma_percentiles=dict(zip(('p50','p90','p99','max'),map(float,np.percentile(scales,[50,90,99,100])))) if len(scales) else {},
        compositing_mass_fraction={name:float(mass[i]/total) if total else None for i,name in enumerate(names)},
        mean_accumulated_alpha=total/((x1-x0)*(y1-y0)),
        pixels_with_large_clone_weight_above_half=float((weights[0,y0:y1,x0:x1] > .5).mean()))


def rotations(quaternions):
    q=np.asarray(quaternions,dtype=np.float64)
    norms=np.linalg.norm(q,axis=1,keepdims=True)
    if not np.isfinite(q).all() or (norms < 1e-12).any():
        raise ValueError('Invalid Gaussian rotation')
    w,x,y,z=(q/norms).T
    return np.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                     2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                     2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),axis=1).reshape(-1,3,3)


def ellipsoid_segments(xyz, scales, quaternion, samples=32):
    """Three principal 3-sigma rings per selected Gaussian, in world units."""
    rotation=rotations(quaternion)
    angle=np.linspace(0,2*np.pi,samples+1)
    circles=[]
    for a,b in ((0,1),(0,2),(1,2)):
        ring=np.zeros((len(angle),3)); ring[:,a]=np.cos(angle); ring[:,b]=np.sin(angle)
        circles.append(ring)
    local=np.stack(circles)[None,:,:,:]*np.asarray(scales)[:,None,None,:]*3
    world=np.einsum('nij,nrsj->nrsi',rotation,local)+np.asarray(xyz)[:,None,None,:]
    return np.stack((world[:,:,:-1],world[:,:,1:]),axis=-2).reshape(-1,2,3).astype(np.float32)


def projected_ellipses(xyz, scales, quaternion, view, width, height, fovx, fovy, samples=48):
    """3-sigma projected covariance contours with the native 0.3px low-pass term."""
    rotation=rotations(quaternion)
    cov=np.einsum('nij,nj,nkj->nik',rotation,np.asarray(scales)**2,rotation)
    camera=np.asarray(xyz)@view[:3,:3]+view[3,:3]
    r=view[:3,:3].T
    cov=np.einsum('ij,njk,lk->nil',r,cov,r)
    fx,fy=width/(2*np.tan(fovx/2)),height/(2*np.tan(fovy/2))
    z=camera[:,2]
    tx=np.clip(camera[:,0]/z,-1.3*np.tan(fovx/2),1.3*np.tan(fovx/2))*z
    ty=np.clip(camera[:,1]/z,-1.3*np.tan(fovy/2),1.3*np.tan(fovy/2))*z
    j=np.zeros((len(xyz),2,3));j[:,0,0]=fx/z;j[:,0,2]=-fx*tx/z**2
    j[:,1,1]=fy/z;j[:,1,2]=-fy*ty/z**2
    cov2=np.einsum('nij,njk,nlk->nil',j,cov,j)+.3*np.eye(2)
    values,axes=np.linalg.eigh(cov2)
    angle=np.linspace(0,2*np.pi,samples+1)
    unit=np.stack((np.cos(angle),np.sin(angle)),1)
    return np.einsum('nij,nsj->nsi',axes,unit[None]*3*np.sqrt(np.maximum(values,0))[:,None])


def growth_window_stats(scales, clone_grad, split_grad, max_radii, extent, percent_dense, clone_threshold, split_threshold):
    """Counts from the real accumulated training gradients before growth resets them."""
    import torch
    sigma=scales.amax(dim=1)
    clone=(torch.norm(clone_grad,dim=-1) >= clone_threshold) & (sigma <= percent_dense*extent)
    split=(split_grad.reshape(-1) >= split_threshold) & (sigma > percent_dense*extent)
    return dict(scene_extent=float(extent), percent_dense=float(percent_dense),
        split_sigma_threshold=float(percent_dense*extent), eligible_clone_parents=int(clone.sum().item()),
        eligible_split_parents=int(split.sum().item()),
        clone_parents_radius_gt20_in_window=int((clone & (max_radii > 20)).sum().item()),
        split_parents_radius_gt20_in_window=int((split & (max_radii > 20)).sum().item()))
