"""Connected, bounded LiDAR triangles. No predicted-depth acceptance mask is used.

Voxelization selects measured points (it never averages opposite sides of a wall).
Ball pivoting is local and overlapping, with bounded triangle edges and measured
support at edge midpoints and centroids. Unobserved openings are not closed.
This is CPU preprocessing; the resulting fixed mesh is rasterized on CUDA.
"""
import numpy as np


def measured_representatives(points, voxel_size):
    ids = np.flatnonzero(np.isfinite(points).all(axis=1))
    if not len(ids):
        raise ValueError('LiDAR cloud contains no finite points')
    finite = np.asarray(points[ids], dtype=np.float64)
    cells = np.floor((finite-finite.min(axis=0))/voxel_size).astype(np.int64)
    _, first = np.unique(cells, axis=0, return_index=True)
    ids = ids[np.sort(first)]
    return np.asarray(points[ids], dtype=np.float64), ids


def spatial_blocks(points, block_size, max_points):
    """Split by occupied spatial extent; no dense scene-sized voxel allocation."""
    pending = [np.arange(len(points))]
    while pending:
        ids = pending.pop()
        p = points[ids]
        extent = np.ptp(p, axis=0)
        if len(ids) <= max_points and extent.max() <= block_size:
            yield ids
            continue
        axis = int(extent.argmax())
        if extent[axis] > block_size:
            left = p[:, axis] < (p[:, axis].min()+p[:, axis].max())/2
            pending.extend((ids[~left], ids[left]))
        else:
            order = np.argsort(p[:, axis], kind='stable')
            middle = len(ids)//2
            pending.extend((ids[order[middle:]], ids[order[:middle]]))


def orient_normals(points, normals, radius, threads):
    """Local kNN spanning forest, avoiding a full-scene 3D Delaunay allocation.

    This also handles exactly planar clouds, for which 3D QHull orientation is
    degenerate. Sign propagation changes neither point positions nor normals'
    tangent planes. Double-sided rendering does not require an outward side.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    distances, neighbors = cKDTree(points).query(points, k=min(13, len(points)), workers=threads)
    rows = np.repeat(np.arange(len(points)), neighbors.shape[1]-1)
    cols, distance = neighbors[:, 1:].ravel(), distances[:, 1:].ravel()
    use = distance <= radius
    rows, cols, distance = rows[use], cols[use], distance[use]
    weights = 1-np.abs(np.einsum('ij,ij->i', normals[rows], normals[cols])) + 1e-3*distance/radius + 1e-6
    graph = coo_matrix((weights, (rows, cols)), shape=(len(points), len(points))).tocsr()
    tree = minimum_spanning_tree(graph)
    tree = (tree+tree.T).tocsr()
    seen = np.zeros(len(points), bool)
    for root in range(len(points)):
        if seen[root]:
            continue
        stack, seen[root] = [root], True
        while stack:
            parent = stack.pop()
            for child in tree.indices[tree.indptr[parent]:tree.indptr[parent+1]]:
                if seen[child]:
                    continue
                seen[child] = True
                if normals[parent] @ normals[child] < 0:
                    normals[child] *= -1
                stack.append(child)
    return normals


def build_lidar_mesh(points, voxel_size=.05, max_edge=.20, support_distance=.075,
                     normal_radius=.15, block_size=4., max_block_points=100000,
                     threads=8, progress=None):
    import open3d as o3d
    from scipy.spatial import cKDTree
    values = [voxel_size, max_edge, support_distance, normal_radius, block_size]
    if not np.isfinite(values).all() or min(values) <= 0 or max_block_points < 32 or threads < 1:
        raise ValueError('Invalid LiDAR meshing settings')
    representatives, original_ids = measured_representatives(points, voxel_size)
    if len(representatives) < 4:
        raise ValueError('At least four finite LiDAR representatives are required')
    tree = cKDTree(representatives)
    blocks = list(spatial_blocks(representatives, block_size, max_block_points))
    radii = np.asarray([1.5, 2., 3.])*voxel_size
    halo = max(2*radii[-1], normal_radius, max_edge)
    pieces, before_filter, rejected = [], 0, 0
    for index, core in enumerate(blocks):
        lo, hi = representatives[core].min(0), representatives[core].max(0)
        local = np.asarray(tree.query_ball_point((lo+hi)/2, np.max(hi-lo)/2+halo,
                                                p=np.inf, workers=threads), dtype=np.int64)
        p = representatives[local]
        local = np.sort(local[((p >= lo-halo) & (p <= hi+halo)).all(1)])
        p = representatives[local]
        if len(p) < 4:
            continue
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=12),
                               fast_normal_computation=False)
        normals = orient_normals(p, np.asarray(cloud.normals).copy(), 2*normal_radius, threads)
        candidates = []
        for sign in (1., -1.):
            cloud.normals = o3d.utility.Vector3dVector(sign*normals)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                cloud, o3d.utility.DoubleVector(radii))
            candidates.append(np.asarray(mesh.triangles, dtype=np.int64).copy())
        # A neighboring wall can obstruct the pivot ball on one side. Both
        # orientations are necessary to reconstruct *both* measured sheets.
        # This does not displace or thicken either surface.
        faces = np.concatenate(candidates)
        before_filter += len(faces)
        # Keep overlapping halo triangles too: independent block boundaries must
        # not cut new cracks into a continuous wall. Duplicate faces are removed.
        for start in range(0, len(faces), 65536):
            f = faces[start:start+65536]
            triangles = p[f]
            edges = triangles[:, [1, 2, 0]]-triangles
            cross = np.cross(edges[:, 0], -edges[:, 2])
            area = np.linalg.norm(cross, axis=1)
            direction = cross/np.maximum(area[:, None], 1e-15)
            normal_agreement = np.abs(np.einsum('ni,nji->nj', direction, normals[f])).min(1)
            keep = (np.linalg.norm(edges, axis=2).max(1) <= max_edge) & (area > 1e-12) & (normal_agreement >= .5)
            # A triangle must remain close to each vertex's local tangent
            # plane. This rejects slanted bridges between nearby wall sheets.
            offsets = np.maximum(np.abs(np.einsum('nji,nji->nj', edges, normals[f])),
                                 np.abs(np.einsum('nji,nji->nj', edges[:, [2, 0, 1]], normals[f])))
            keep &= offsets.max(1) <= voxel_size
            samples = np.concatenate(((triangles+triangles[:, [1, 2, 0]])/2,
                                      triangles.mean(1)[:, None]), axis=1)
            distance, _ = tree.query(samples.reshape(-1, 3), workers=threads)
            keep &= (distance.reshape(-1, 4) <= support_distance).all(1)
            rejected += int((~keep).sum())
            if keep.any():
                pieces.append(local[f[keep]])
        if progress:
            progress(index+1, len(blocks), len(local), len(faces))
    if not pieces:
        raise RuntimeError('No supported LiDAR triangles; inspect point spacing and meshing scales')
    faces = np.concatenate(pieces)
    _, unique = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[np.sort(unique)]
    used, inverse = np.unique(faces, return_inverse=True)
    missing = np.ones(len(representatives), bool)
    missing[used] = False
    report = dict(method='overlapping_local_two_sided_ball_pivoting', input_points=len(points),
        finite_points=int(np.isfinite(points).all(1).sum()), representatives=len(representatives),
        meshed_representatives=len(used), meshed_representative_fraction=len(used)/len(representatives),
        unmeshed_representatives=int(missing.sum()), blocks=len(blocks), halo=float(halo),
        pivot_radii=radii.tolist(), triangles_before_filter=before_filter,
        triangles_rejected=rejected, vertices=len(used), triangles=len(faces),
        coordinates='unchanged initialization LiDAR coordinates; vertices are measured points',
        policy='bounded measured triangles; no DA3 mask, global closure, or occupied-volume dilation')
    return (representatives[used].astype(np.float32), inverse.reshape(-1, 3).astype(np.int32),
            original_ids[used], original_ids[missing], report)
