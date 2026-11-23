"""Mesh state, UV-preserving topology and surface supervision."""

from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage

from .data import nearest_track_faces
from .render import clamp_to_triangle, edge_neighbors, normals, render


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray
    colors: np.ndarray
    opacity: np.ndarray
    hard: np.ndarray
    background: np.ndarray
    neighbors: np.ndarray
    transmitting: np.ndarray
    vertex_color: bool = False
    shader: str = "atlas"

    def clone(self, **fields):
        return replace(self, **{k: v.copy() if isinstance(v, np.ndarray) else v
                                for k, v in vars(self).items()} | fields)

    def write(self, path):
        np.savez_compressed(path, **vars(self))

    @classmethod
    def read(cls, path):
        with np.load(path) as state:
            return cls(**{k: state[k] for k in state.files if k not in ("vertex_color","shader")},
                       vertex_color=bool(state["vertex_color"]), shader=str(state["shader"]))


def initial_mesh(vertices, faces, camera, background):
    uv = camera.project(vertices[faces]) / [camera.width-1, camera.height-1]
    return Mesh(vertices.copy(), faces.copy(), uv, np.full((len(vertices),3),.47),
                np.full(len(faces),.90), np.zeros(len(faces),bool), np.asarray(background),
                edge_neighbors(faces), np.zeros(len(faces),bool), True)


def disconnect(mesh):
    faces = np.arange(mesh.faces.size).reshape(mesh.faces.shape)
    colors = mesh.colors[mesh.faces].reshape(-1, 3) if mesh.vertex_color else mesh.colors.copy()
    return mesh.clone(vertices=mesh.vertices[mesh.faces].reshape(-1,3), faces=faces,
                      neighbors=edge_neighbors(faces), colors=colors)


def connect(soup, template, image, size):
    vertices = template.vertices.copy()
    count = np.bincount(template.faces.ravel(), minlength=len(vertices))
    for axis in range(3):
        merged = np.bincount(template.faces.ravel(), weights=soup.vertices[:,axis], minlength=len(vertices))
        vertices[:,axis] = np.divide(merged, count, out=vertices[:,axis], where=count>0)
    # Seed a low-pass atlas using only the reference fitting image.
    h,w = size
    y,x = np.meshgrid(np.linspace(0,image.shape[0]-1,h),
                      np.linspace(0,image.shape[1]-1,w), indexing="ij")
    blurred = ndimage.gaussian_filter(image, (1.25,1.25,0))
    colors = np.stack([ndimage.map_coordinates(blurred[...,c],[y,x],order=1) for c in range(3)],axis=-1)
    return template.clone(vertices=vertices, colors=colors, opacity=soup.opacity.copy(), vertex_color=False)


def subset_faces(mesh, keep):
    remap = np.full(len(mesh.faces),-1,int)
    remap[keep] = np.arange(keep.sum())
    faces = mesh.faces[keep]
    return mesh.clone(faces=faces, uv=mesh.uv[keep], opacity=mesh.opacity[keep], hard=mesh.hard[keep],
                      transmitting=mesh.transmitting[keep], neighbors=edge_neighbors(faces)), remap


def split_face(mesh, face_id):
    """Interior centroid split. Outer edges and adjacent topology stay conforming."""
    face = mesh.faces[face_id]
    center = len(mesh.vertices)
    children = np.stack((face,np.roll(face,-1),np.repeat(center,3)), axis=1)
    uv = mesh.uv[face_id]
    child_uv = np.stack((uv,np.roll(uv,-1,axis=0),np.broadcast_to(uv.mean(0),uv.shape)),axis=1)
    faces = np.concatenate((mesh.faces.copy(),children[1:]))
    faces[face_id] = children[0]
    uvs = np.concatenate((mesh.uv.copy(),child_uv[1:]))
    uvs[face_id] = child_uv[0]
    colors = mesh.colors
    if mesh.vertex_color:
        colors = np.vstack((colors,colors[face].mean(0)))
    return mesh.clone(vertices=np.vstack((mesh.vertices,mesh.vertices[face].mean(0))), faces=faces,
                      uv=uvs, colors=colors, neighbors=edge_neighbors(faces),
                      opacity=np.r_[mesh.opacity,np.repeat(mesh.opacity[face_id],2)],
                      hard=np.r_[mesh.hard,np.repeat(mesh.hard[face_id],2)],
                      transmitting=np.r_[mesh.transmitting,np.repeat(mesh.transmitting[face_id],2)])


def fit_evidence(model, tracks, cams, assignments=None):
    if assignments is None:
        assignments = nearest_track_faces(model, tracks)
    free = []
    for j,i in enumerate(tracks["cameras"]):
        keep = tracks["obs"][:,j]
        xyz = tracks["points"][keep]
        free.append((i,cams[i].project(xyz),np.linalg.norm(xyz-cams[i].center,axis=1),
                     tracks["tolerance"][keep]))
    return assignments, free


def point_triangle_dist(points, triangles):
    a,b,c = np.moveaxis(triangles,1,0)
    e,g,p = b-a,c-a,points-a
    ee, eg, gg = (e*e).sum(1), (e*g).sum(1), (g*g).sum(1)
    pe, pg = (p*e).sum(1), (p*g).sum(1)
    det = np.maximum(ee*gg-eg*eg, 1e-16)
    u,v = (pe*gg-pg*eg)/det,(pg*ee-pe*eg)/det
    beta = clamp_to_triangle(np.stack((1-u-v,u,v),axis=1),triangles)
    return np.linalg.norm(points-(beta[...,None]*triangles).sum(1),axis=1)


def anchor_errors(model, xyz, assignments, scale, cap):
    valid = assignments >= 0
    err = np.full(len(xyz),cap)
    err[valid] = np.minimum(cap,point_triangle_dist(xyz[valid],model.vertices[model.faces[assignments[valid]]])/scale)
    return err


def area(model):
    a,b,c = np.moveaxis(model.vertices[model.faces],1,0)
    return .5*np.linalg.norm(np.cross(b-a,c-a),axis=1)


def smooth_edges(mesh, angle=28.):
    f,e = np.nonzero(mesh.neighbors >= 0)
    g = mesh.neighbors[f,e]
    keep = f < g
    f,g = f[keep],g[keep]
    n = normals(mesh.vertices,mesh.faces)
    # Winding is set by the connected screen triangulation. Treat doubtful normals as creases.
    keep = (n[f]*n[g]).sum(1) >= np.cos(np.deg2rad(angle))
    return f[keep],g[keep]


def regularizers(model, edges):
    f,g = edges
    n = normals(model.vertices,model.faces)
    bend = np.minimum(1-(n[f]*n[g]).sum(1),.30)
    v = model.vertices[model.faces]
    squared_edge_sum = ((v-np.roll(v,1,axis=1))**2).sum(axis=(1,2))
    quality = np.clip(squared_edge_sum/(4*np.sqrt(3)*np.maximum(area(model),1e-9))-1,0,12)
    return bend,quality


def gray_patch(images, i, coords):
    gray = images[i] @ [.25,.5,.25]
    return ndimage.map_coordinates(gray,[coords[...,1],coords[...,0]],order=1,mode="nearest")


def intersect_assigned(vertices, faces, center, dirs):
    tri = vertices[faces]
    a,b,c = np.moveaxis(tri,1,0)
    n = np.cross(b-a,c-a)
    den = (dirs*n[:,None,:]).sum(-1)
    depth = ((a-center)*n).sum(-1)[:,None] / np.where(abs(den)>1e-11,den,1e-11)
    return center + depth[...,None]*dirs


def patch_correspondences(model, images, cams, i, j, coords, tau):
    stencil = np.stack(np.meshgrid(np.arange(-1,2),np.arange(-1,2)),axis=-1).reshape(-1,2)
    p = coords[:,None]+stencil
    source = gray_patch(images,i,p)
    a,b = cams[i],cams[j]
    if np.linalg.norm(a.center-b.center) < .20:
        return None
    pixels = p[:,[0,2,4,6,8]].reshape(-1,2)
    refs = render(model,a,pixels,tau)
    winner = refs["weight"].argmax(1).reshape(-1,5)
    ids = winner[:,2]
    support = refs["weight"].max(1).reshape(-1,5)
    same = np.all(winner == ids[:,None],axis=1) & (support.min(1)>.45)
    dirs = a.rays(p)
    points = intersect_assigned(model.vertices,model.faces[ids],a.center,dirs)
    target = b.project(points)
    good = ((target > 1) & (target < [b.width-2,b.height-2])).all(axis=(1,2))
    good &= np.isfinite(target).all(axis=(1,2)) & (b.local(points)[...,2]>.05).all(axis=1)
    good &= same & (source.var(1)>.0011) & (source.min(1)>.035) & (source.max(1)<.965)
    if not good.any():
        return None
    ids,source,dirs,target = ids[good],source[good],dirs[good],target[good]
    p = target[:,[0,2,4,6,8]].reshape(-1,2)
    target_hit = render(model,b,p,tau)
    same = (target_hit["weight"].argmax(1).reshape(-1,5) == ids[:,None]).all(1)
    same &= target_hit["weight"].max(1).reshape(-1,5).min(1)>.45
    dest = gray_patch(images,j,target)
    same &= (dest.var(1)>.0011) & (dest.min(1)>.035) & (dest.max(1)<.965)
    return (ids[same],source[same],dirs[same],i,j) if same.any() else None


def warp_errors(model, images, cams, patches):
    if patches is None:
        return np.empty(0)
    ids,source,dirs,i,j = patches
    points = intersect_assigned(model.vertices,model.faces[ids],cams[i].center,dirs)
    target = gray_patch(images,j,cams[j].project(points))
    a = source-source.mean(1,keepdims=True)
    b = target-target.mean(1,keepdims=True)
    corr = (a*b).sum(1)/np.sqrt(np.maximum((a*a).sum(1)*(b*b).sum(1),1e-10))
    return np.minimum(1-np.clip(corr,-1,1),1.35)
