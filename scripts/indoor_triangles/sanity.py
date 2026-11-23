"""Precision checks with an independent scalar tracer and analytic test fixtures."""

import numpy as np

from .model import initial_mesh, split_face
from .render import edge_neighbors, render
from .synthetic import camera, scalar_hit


def quad(x, xx, y, yy, z):
    return np.array([[x,y,z],[xx,y,z],[xx,yy,z],[x,yy,z]],float)


def fixtures():
    z=np.vstack((quad(-2,0,-1.5,1.5,3.),quad(-3,3,-2,2,5.),
                 quad(.25,.9,-.45,.45,2.)))
    f=np.vstack([np.array([[0,1,2],[0,2,3]])+s for s in (0,4,8)])
    cam=camera(0.)
    # Uncovered doorway is x>0. The pane covers only part of that opening.
    model=initial_mesh(z,f,cam,[.04,.04,.04])
    model.colors[:4]=[.21,.22,.24]
    model.colors[4:8]=[.82,.13,.33]
    model.colors[8:]=[.20,.54,.66]
    model.hard[:4]=True
    model.transmitting[4:]=True
    model.opacity[4:]=.38
    return cam,model


def check_elastic(cam, room, coords, exact):
    projected = np.array([[10., 10.], [46., 10.], [10., 46.]])
    rays = cam.rays(projected)
    vertices = cam.center + rays * (3. / (rays @ cam.rotation[2]))[:, None]
    smooth = initial_mesh(vertices, np.array([[0, 1, 2]]), cam, room.background)
    elastic = smooth.clone(elastic_sigma=.45)
    samples = np.array([[22., 22.], [22., 10.], [22., 9.], [22., -4.]])
    alpha = render(elastic, cam, samples)["weight"][:, 0]
    expected = elastic.opacity[0] * np.array([1., np.exp(-.45),
                                               np.exp(-.45 * (1. + 3. / 36.)**(2. / .45)), 0.])
    np.testing.assert_allclose(alpha, expected, atol=1e-12, rtol=0.)
    sharp = render(elastic.clone(elastic_sigma=.055), cam, samples)["weight"][:, 0]
    assert sharp[1] > alpha[1] and sharp[2] < alpha[2]

    # Move the boundary toward a pixel beyond the smoothstep support.
    delta = .001
    outside = samples[2:3]
    motion = np.array([0., delta, 0.])
    gradients = []
    for mesh in (smooth, elastic):
        plus = render(mesh.clone(vertices=vertices-motion), cam, outside)["weight"][0, 0]
        minus = render(mesh.clone(vertices=vertices+motion), cam, outside)["weight"][0, 0]
        gradients.append(float((plus-minus) / (2. * delta)))
    assert gradients[0] == 0. and gradients[1] > 0.

    mixed = render(room.clone(elastic_sigma=.45), cam, coords)
    np.testing.assert_array_equal(mixed["hard_hit"], exact["hard_hit"])
    # Near-wall opaque hits cannot be influenced by the farther wall.
    blocked = (exact["hard_hit"] >= 0) & (exact["hard_hit"] < 2)
    np.testing.assert_array_equal(mixed["weight"][blocked, 2:4], 0.)
    seam = cam.project(np.array([[-1., 0., 3.]]))
    connected = room.clone(hard=np.zeros(len(room.faces), bool), elastic_sigma=.45)
    seam_alpha = render(connected, cam, seam)["weight"][0, :2]
    assert np.count_nonzero(seam_alpha) == 1
    np.testing.assert_allclose(seam_alpha.sum(), connected.opacity[0])
    return dict(edge_weight=float(alpha[1]/elastic.opacity[0]),
                sharpened_edge_weight=float(sharp[1]/elastic.opacity[0]),
                exterior_alpha_gradient_per_m=gradients,
                mixed_exact_hit_checks=len(coords))


def run_checks():
    cam,model=fixtures()
    cx,cy=cam.intrinsics[2:]
    r=np.random.default_rng(290)
    coords=r.uniform([4,7],[cam.width-4,cam.height-7],size=(704,2))
    pred=render(model,cam,coords)
    d,b=[],[]
    triangles=model.vertices[model.faces]
    for j,ray in enumerate(cam.rays(coords)):
        results=[(h[0],i,h[1]) for i,triangle in enumerate(triangles)
                 if (h:=scalar_hit(cam.center,ray,triangle)) is not None and model.hard[i]]
        if results:
            t,i,beta=min(results,key=lambda item:(item[0],item[1]))
            assert pred["hard_hit"][j]==i
            d.append(abs(pred["depth"][j,i]-t))
            b.append(float(np.max(np.abs(pred["beta"][j,i]-beta))))
        else:
            assert pred["hard_hit"][j]<0
    assert max(d)<1e-10 and max(b)<1e-10

    # Select the two sides of a wall edge at subpixel distance.
    adjacent=np.array([[cx-.09,cy+4.],[cx+.09,cy+4.]])
    p=render(model,cam,adjacent)
    hit=p["hard_hit"]
    assert hit[0] in (0,1) and hit[1] in (2,3)
    observed=p["depth"][np.arange(2),hit]
    exact=np.array([3.,5.])/cam.rays(adjacent)[:,2]
    assert np.max(np.abs(observed-exact))<1e-10

    # Intersect exact shared seams. Soft edges use one owner too.
    seam=cam.project(np.array([[-1,0,3.]]))
    ph=render(model,cam,seam)
    ps=render(model.clone(hard=np.zeros(len(model.faces),bool)),cam,seam)
    assert ph["weight"][0,:2].sum()==1 and np.count_nonzero(ph["weight"][0,:2])==1
    assert np.count_nonzero(ps["weight"][0,:2])==1

    # Range is off-axis Euclidean, distinct from camera-z.
    px=np.array([[15.,25.]])
    hit=render(model,cam,px)["hit"][0]
    offaxis=render(model,cam,px)["depth"][0,hit]
    true_range=3/cam.rays(px)[0,2]
    assert abs(offaxis-true_range)<1e-10 and offaxis-3>.1

    rays=np.array([[cx-10,cy],[cx+5,cy],[cx+23,cy]])
    eps=1e-5
    increased=model.clone()
    decreased=model.clone()
    increased.colors[4:8,0]+=eps
    decreased.colors[4:8,0]-=eps
    diff=(render(increased,cam,rays)["rgb"][:,0]-render(decreased,cam,rays)["rgb"][:,0])/(2*eps)
    assert np.allclose(diff,[0.,1.,.62],atol=1e-8)
    no_pane=slice(0,2)
    nearly_opaque=model.clone(hard=np.zeros(len(model.faces),bool))
    nearly_opaque.opacity[:4]=1.-1e-6
    soft_far=render(nearly_opaque,cam,rays[no_pane],.005)["rgb"]
    hard_far=render(model,cam,rays[no_pane],.005)["rgb"]
    residual=float(np.abs(soft_far-hard_far).max())
    assert residual<2e-6

    # Integrate a sharp moving edge across a pixel. The expected derivative
    # of red is the radiance jump times fx/z per metre of lateral movement.
    samples=np.stack((cx+(np.arange(4096)+.5)/4096-.5,np.full(4096,cy+4)),axis=1)
    delta=.006
    plus_geometry,minus=model.clone(),model.clone()
    plus_geometry.vertices[[1,2],0]+=delta
    minus.vertices[[1,2],0]-=delta
    geom=float((render(plus_geometry,cam,samples)["rgb"][:,0].mean()-
                render(minus,cam,samples)["rgb"][:,0].mean())/(2*delta))
    analytic=float((model.colors[0,0]-model.colors[4,0])*cam.intrinsics[0]/3)
    assert abs(geom/analytic-1)<.004
    lo,hi=model.clone(),model.clone()
    lo.vertices[[1,2],0]-=.002
    hi.vertices[[1,2],0]+=.002
    four=np.array([[cx-.25,cy+3.75],[cx+.25,cy+3.75],
                   [cx-.25,cy+4.25],[cx+.25,cy+4.25]])
    unresolved=float((render(hi,cam,four)["rgb"][:,0].mean()-
                       render(lo,cam,four)["rgb"][:,0].mean())/.004)
    assert abs(unresolved)<1e-12

    # Use a nonconstant atlas so interpolation invariance is also tested.
    atlas=model.clone(vertex_color=False,colors=r.random((19,25,3)))
    subdivided=split_face(atlas,0)
    a=render(atlas,cam,coords,color_basis=True)
    bsub=render(subdivided,cam,coords,color_basis=True)
    subdivision=float(np.max(abs(a["rgb"]-bsub["rgb"])))
    assert subdivision<1e-10 and np.array_equal(a["hit"]>=0,bsub["hit"]>=0)
    plus=atlas.clone(colors=atlas.colors+.003)
    subplus=split_face(plus,0)
    gradient_error=float(np.max(abs((render(plus,cam,coords)["rgb"]-a["rgb"])-
                                    (render(subplus,cam,coords)["rgb"]-bsub["rgb"])))/.003)
    assert gradient_error<1e-10
    # Propagate the same parent vertex move to the centroid as a constrained child.
    sub_lo = split_face(minus,0)
    sub_hi = split_face(plus_geometry,0)
    moved_subdivided = float((render(sub_hi,cam,samples)["rgb"][:,0].mean() -
                              render(sub_lo,cam,samples)["rgb"][:,0].mean())/(2*delta))
    assert abs(moved_subdivided-geom)<1e-10

    # One corner behind the near clip. Compare only reference hits above it.
    near=model.clone(vertices=np.array([[-.18,-.08,-.05],[1.,-.4,1.],[.1,.8,1.2]]),
                     faces=np.array([[0,1,2]]),opacity=np.array([.8]),hard=np.array([True]),
                     transmitting=np.array([False]),neighbors=edge_neighbors(np.array([[0,1,2]])),
                     colors=np.ones((3,3)),uv=np.zeros((1,3,2)))
    p=render(near,cam,coords,near=.05)
    ref=[scalar_hit(cam.center,u,near.vertices,near=.05/u[2]) is not None for u in cam.rays(coords)]
    assert np.count_nonzero(ref)>0 and np.array_equal(ref,p["hit"]>=0)
    two_faces = np.array([[0, 1, 2], [2, 1, 3]])
    clipped_neighbors = near.clone(
        vertices=np.vstack((near.vertices, [1.1, .85, -.05])),
        faces=two_faces, opacity=np.array([.8, .8]), hard=np.array([True, True]),
        transmitting=np.array([False, False]), neighbors=edge_neighbors(two_faces),
        colors=np.ones((4, 3)), uv=np.zeros((2, 3, 2)))
    shared_point = clipped_neighbors.vertices[[1, 2]].mean(0)
    at_seam = cam.project(shared_point[None])
    exact = render(clipped_neighbors,cam,at_seam,near=.05)
    assert exact["hit"][0] == 0 and np.count_nonzero(exact["weight"]) == 1
    return dict(scalar_reference_hits=len(d),max_range_error_m=max(d),max_barycentric_error=max(b),
                near_far_adjacent_range_m=observed.tolist(),offaxis_range_m=float(offaxis),
                offaxis_camera_z_m=3.,seam_contributing_foreground_faces=1,
                rear_rgb_fd=diff.tolist(),hard_soft_max_rgb_difference=residual,
                hard_silhouette_rgb_per_m=geom,analytic_silhouette_rgb_per_m=analytic,
                four_sample_unresolved_silhouette_rgb_per_m=unresolved,
                subdivision_max_rgb_delta=subdivision,subdivision_rgb_gradient_delta=gradient_error,
                subdivided_silhouette_rgb_per_m=moved_subdivided,
                clipped_hits=int(np.count_nonzero(ref)),clipped_shared_edge_owner=0,
                elastic=check_elastic(cam,model,coords,pred),passed=True)
