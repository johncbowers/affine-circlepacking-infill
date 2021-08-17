#! /usr/bin/python3 
# Author: John Bowers
# Version: 0.1 Aug 3, 2021

from koebe.algorithms.incrementalConvexHull import incrConvexHull, orientationPointE3
from koebe.geometries.triangleFunctions import barycentricTransformE3, barycentricTransformE2, triangleBasis, barycentricCoordinatesOfE3
from koebe.geometries.euclidean3 import PointE3, VectorE3, SegmentE3
from koebe.geometries.euclidean2 import PointE2, SegmentE2
from koebe.geometries.commonOps import orientation2, Orientation

import numpy as np
from scipy.spatial import Delaunay

from vedo import Plotter
from vedo.shapes import Lines, Plane, Points, Arrows, Spheres, Circle
from vedo.mesh import Mesh

from koebe.algorithms.hypPacker import maximalPacking

from convex_hull import convexHullE3
from affine_sampling import adaptiveAffinePoissonSampling2D, radiusBasisHelper

from affine_mapping import vectorTransport3Dto2D

from graphics import show_mesh, show_mesh_and_samples, show_points, mesh_lines, dual_mesh_lines

from animated_plotter import AnimatedPlotter

####
# Helper Code
####

def face_orientation(f):
    p1, p2, p3 = [v.data for v in f.vertices()]
    return orientation2(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y)

def label_areas(mesh):
    def calc_area(face):
        A, B, C = [v.data for v in face.vertices()]
        u = B - A
        v = C - A
        return 0.5 * u.cross(v).norm()
    for f in mesh.faces:
        if f != mesh.outerFace:
            f.area = calc_area(f)

def main():

    print("Reading mesh data...")

    # open the point set file: 
    f = open('./data/ovalloaf/SurfaceStress_coarseData.txt', 'r')

    # Read the relevant lines (the first line is documentation) and split
    # into tuples (xi, yi, zi, si)
    data = [[float(x.strip()) for x in line.split(', ')] for line in f.readlines()[1:]]

    # Extract the points and stresses as their own lists: 
    pts       = [PointE3(x, y, z) for x, y, z, s in data]
    pts_np    = np.array([[x, y, z] for x, y, z, _ in data])
    stresses = [s for x, y, z, s in data]

    # Extract the minimum and maximum stress values
    max_stress = max(stresses)
    min_stress = min(stresses)

    min_radius = 0.02#0.0075 #0.01
    max_radius = 0.04#0.03 #0.03

    norm_stresses = [(s - min_stress) / (max_stress - min_stress) for s in stresses]
    desired_radii = [min_radius + (1 - s) * (max_radius - min_radius) for s in norm_stresses]

    pts_color  = np.array([[s * 255, 0, (1-s)*255] for s in norm_stresses])

    # Mesh the initial sample points

    orig_mesh = convexHullE3(pts + [PointE3(0,0,-0.5)])
    orig_mesh.outerFace = orig_mesh.verts[-1].remove()

    orig_mesh.markIndices()

    # Add the stress data to each vertex object
    for i in range(len(orig_mesh.verts)):
        orig_mesh.verts[i].desired_radius = desired_radii[i]

    print("Computing a circle packing...")

    # Find the closest vertex to the origin in 2D in order to have the circle packing
    # layout engine center that vertex. 
    dists = [(PointE3(*tuple(v.data)[0:2],0) - PointE3.O).normSq() for v in orig_mesh.verts]
    closestToOriginIdx = dists.index(min(dists))

    # Compute a circle packing of orig_mesh
    packing, _ = maximalPacking(
        orig_mesh, 
        num_passes=1000, 
        centerDartIdx = orig_mesh.darts.index(orig_mesh.verts[closestToOriginIdx].aDart)
    )
    packing.markIndices()

    # The coordinates of packing are given as hyperbolic coordinates, convert these
    # to PointE2 types for convenience. 
    packingMesh = packing.duplicate(
        vdata_transform=lambda d: PointE3(d.center.coord.real, d.center.coord.imag, 0)
    )
    packingMesh.markIndices()

    label_areas(orig_mesh)
    label_areas(packingMesh)

    for fIdx in range(len(orig_mesh.faces)):
        if orig_mesh.faces[fIdx] != orig_mesh.outerFace:
            packingMesh.faces[fIdx].area_ratio = packingMesh.faces[fIdx].area / orig_mesh.faces[fIdx].area

    # Compute the adaptive affine sampling
    print("Computing an anisotropic Poisson sampling...")
    accepted_samples = adaptiveAffinePoissonSampling2D(
        0.075, # Controls the hash-grid size, which may change speed the algorithm runs in
        packingMesh, # 2D circle packing mesh
        orig_mesh, # original 3D mesh
        k=10 # Quality control, bigger is higher quality, but slower. k=50 is probably fine for high quality.
    )

    # Lift the samples from 2D to 3D via the inverse packing map
    print("Lifting sampling back onto the original mesh...")

    lifted_points = []
    for sample in accepted_samples:
        p, _, _, _, _, face_idx, _ = sample
        result = radiusBasisHelper(PointE2(*p), face_idx, packingMesh, orig_mesh)
        _, Pp, _, _, _, _, _ = result
        lifted_points.append(Pp)


    # Remesh the sampling points.
    boundary_pos    = [v.data for v in orig_mesh.outerFace.vertices()]
    boundary_pos_2d = [v.data for v in packingMesh.outerFace.vertices()]
    boundary_radii  = [v.desired_radius for v in orig_mesh.outerFace.vertices()]

    # Use the Deluanay triangulation of the 2D points to obtain a new mesh. 
    dt_points = [PointE3(x, y, -(x*x + y*y)) for (x,y),_,_,_,_,_,_ in accepted_samples]
    dt_bdry_points = [PointE3(p.x, p.y, -(p.x*p.x + p.y*p.y)) for p in boundary_pos_2d]
    new_mesh = convexHullE3(dt_points + dt_bdry_points + [PointE3(0,0,-10000)])

    new_mesh.outerFace = new_mesh.verts[-1].remove()
    new_mesh_2D = new_mesh.duplicate()

    # Mark the new_mesh with .desired_radius, .is_interior, and .sample vertex attributes
    # and rest the .data field to be the lifted 3D positions. 
    for vIdx in range(len(accepted_samples)):
        (x, y), r, _, _, _, _, _ = accepted_samples[vIdx]
        new_mesh.verts[vIdx].desired_radius = 0.5 * r
        new_mesh.verts[vIdx].radius         = 0.5 * r
        new_mesh.verts[vIdx].data           = lifted_points[vIdx]
        new_mesh.verts[vIdx].sample         = accepted_samples[vIdx]
        new_mesh.verts[vIdx].found_face_idx = accepted_samples[vIdx][5]
        new_mesh.verts[vIdx].is_interior    = True
        new_mesh.verts[vIdx].update         = VectorE3(0,0,0)

        new_mesh_2D.verts[vIdx].data        = PointE3(x, y, 0)
        new_mesh_2D.verts[vIdx].radius      = 0.01
        new_mesh_2D.verts[vIdx].is_interior = True
        new_mesh_2D.verts[vIdx].found_face_idx = accepted_samples[vIdx][5]
        new_mesh_2D.verts[vIdx].update      = VectorE3(0,0,0)
        
    for rIdx in range(len(boundary_radii)):
        vIdx = rIdx + len(accepted_samples)
        new_mesh.verts[vIdx].desired_radius = 0.5 * boundary_radii[rIdx]
        new_mesh.verts[vIdx].radius         = 0.5 * boundary_radii[rIdx]
        new_mesh.verts[vIdx].data           = boundary_pos[rIdx]
        new_mesh.verts[vIdx].sample         = (tuple(boundary_pos_2d[rIdx]), None, None, None, None, None, None)
        new_mesh.verts[vIdx].is_interior    = False
        new_mesh.verts[vIdx].update         = VectorE3(0,0,0)

        new_mesh_2D.verts[vIdx].data        = PointE3(boundary_pos_2d[rIdx].x, boundary_pos_2d[rIdx].y, 0)
        new_mesh_2D.verts[vIdx].radius      = 0.01
        new_mesh_2D.verts[vIdx].is_interior = False
        new_mesh_2D.verts[vIdx].update      = VectorE3(0,0,0)

    # TODO, find non-delaunay edges in 3D and flip them.

    new_mesh.markIndices()
    new_mesh_2D.markIndices()

    for f in new_mesh_2D.faces:
        if f != new_mesh_2D.outerFace:
            f.orientation = face_orientation(f)
    

    show_points(lifted_points)
    show_mesh_and_samples(new_mesh, accepted_samples + [(None, r, None, None, None, None, None) for r in boundary_radii])
    show_mesh(new_mesh_2D)

    print("Running affine simulation...")

    def spring_update_2D():
        nonlocal new_mesh, new_mesh_2D, packingMesh, orig_mesh

        update_delta = 0.1
        prioritize_desired_radius_epsilon = 0

        pos_update = [VectorE3(0,0,0) for _ in new_mesh_2D.verts]
        rad_update = [0 for _ in new_mesh_2D.verts]

        nbr_count = [0 for _ in new_mesh_2D.verts]

        for e in new_mesh_2D.edges:
            u, v = e.endPoints()
            i, j = u.idx, v.idx
            p, q = u.data, v.data
            eVec = q - p
            eLen = eVec.norm()
            eVecN = (1 / eLen) * eVec
            disp = eLen - (u.radius + v.radius)

            pos_update[i] += disp * eVecN
            pos_update[j] += (-disp) * eVecN

            rad_update[i] += disp
            rad_update[j] += disp

            nbr_count[i] += 1
            nbr_count[j] += 1

        for v in new_mesh_2D.verts:
            if v.is_interior:
                old_vdata = v.data
#                v.data += update_delta * pos_update[v.idx]

                # Check each face's orientation, if one changed, then we have a problem
                face_orientations_are_valid = True
                for f in v.faces():
                    if f.orientation != face_orientation(f):
                        face_orientations_are_valid = False
                        break
                if not face_orientations_are_valid:
                    v.data = old_vdata
                else:
                    result = radiusBasisHelper(PointE2(v.data.x, v.data.y), v.found_face_idx, packingMesh, orig_mesh)
                    if result != None:
                        desired_radius, p3d, _, _, _, _, v.found_face_idx = result
                        new_mesh.verts[v.idx].data = p3d
                        v.update = pos_update[v.idx]
                        desired_radius_2D = 0.25*desired_radius * packingMesh.faces[v.found_face_idx].area_ratio
                        v.radius += update_delta * rad_update[v.idx] * (1 - prioritize_desired_radius_epsilon) + prioritize_desired_radius_epsilon * (desired_radius_2D - v.radius)
                    else:
                        v.data = old_vdata
            else:
                v.radius += update_delta * rad_update[v.idx]
            

    def spring_update_2D_v2():
        nonlocal new_mesh, new_mesh_2D, packingMesh, orig_mesh

        update_delta = 0.1
        prioritize_desired_radius_epsilon = 0

        pos_update = [VectorE3(0,0,0) for _ in new_mesh_2D.verts]
        pos_update_3d = [VectorE3(0,0,0) for _ in new_mesh.verts]

        rad_update = [0 for _ in new_mesh_2D.verts]
        rad_update_3d = [0 for _ in new_mesh.verts]

        nbr_count = [0 for _ in new_mesh_2D.verts]

        for e in new_mesh_2D.edges:
            u, v = e.endPoints()
            i, j = u.idx, v.idx
            p, q = u.data, v.data
            eVec = q - p
            eLen = eVec.norm()
            eVecN = (1 / eLen) * eVec
            disp = eLen - (u.radius + v.radius)

            pos_update[i] += disp * eVecN
            pos_update[j] += (-disp) * eVecN

            rad_update[i] += disp
            rad_update[j] += disp

            nbr_count[i] += 1
            nbr_count[j] += 1

        for e in new_mesh.edges:
            u, v = e.endPoints()
            i, j = u.idx, v.idx
            p, q = u.data, v.data
            eVec = q - p
            eLen = eVec.norm()
            eVecN = (1 / eLen) * eVec
            disp = eLen - (u.radius + v.radius)

            pos_update_3d[i] += disp * eVecN
            pos_update_3d[j] += (-disp) * eVecN

            rad_update_3d[i] += disp
            rad_update_3d[j] += disp

            # nbr_count[i] += 1
            # nbr_count[j] += 1
        
        for v in new_mesh_2D.verts:
            if v.is_interior:
                old_vdata = v.data
                v.data += update_delta * pos_update[v.idx]

                # Check each face's orientation, if one changed, then we have a problem
                face_orientations_are_valid = True
                for f in v.faces():
                    if f.orientation != face_orientation(f):
                        face_orientations_are_valid = False
                        break
                if not face_orientations_are_valid:
                    v.data = old_vdata
                else:
                    result = radiusBasisHelper(PointE2(v.data.x, v.data.y), v.found_face_idx, packingMesh, orig_mesh)
                    update2d = vectorTransport3Dto2D(pos_update_3d[v.idx], v.found_face_idx, packingMesh, orig_mesh)
                    if result != None:
                        desired_radius, p3d, _, _, _, _, v.found_face_idx = result
                        new_mesh.verts[v.idx].data = p3d
                        v.update = pos_update[v.idx]
                        desired_radius_2D = 0.25*desired_radius * packingMesh.faces[v.found_face_idx].area_ratio
                        v.radius += update_delta * rad_update[v.idx] * (1 - prioritize_desired_radius_epsilon) + prioritize_desired_radius_epsilon * (desired_radius_2D - v.radius)

                        v3d = new_mesh.verts[v.idx]
                        v3d.update = pos_update_3d[v.idx]
                        v.update = VectorE3(update2d.x, update2d.y, 0)
                    else:
                        v.data = old_vdata
            else:
                v.radius += update_delta * rad_update[v.idx]

        

    def spring_2D_update_multi_factory(k = 100):
        def repeater():
            nonlocal k
            for _ in range(k):
                spring_update_2D_v2()
        return repeater

    def spring_update():
        nonlocal new_mesh, new_mesh_2D

        update_delta = 0.1
        prioritize_desired_radius_epsilon = 0.5

        pos_update = [VectorE3(0,0,0) for _ in new_mesh.verts]
        rad_update = [0 for _ in new_mesh.verts]

        pos_velocity = [VectorE3(0,0,0) for _ in new_mesh.verts]
        rad_velocity = [0 for _ in new_mesh.verts]

        nbr_count  = [0 for _ in new_mesh.verts]

        # Compute the 3D position update and
        # radius update, plus track the neighbor counts
        # This is done over the edge structure, instead of the 
        # vertex to neighbors structure to compute once per
        # edge instead of twice per edge (should make it twice as fast).
        for e in new_mesh.edges:
            u, v    = e.endPoints()
            i, j    = u.idx, v.idx
            p, q    = u.data, v.data
            eVec    = q - p
            eLen    = eVec.norm()
            eVecN   = (1 / eLen) * eVec
            disp    = eLen - (u.desired_radius + v.desired_radius)

            pos_update[i] += disp * eVecN
            pos_update[j] += (-disp) * eVecN

            rad_update[i] += disp
            rad_update[j] += disp

            nbr_count[i] += 1
            nbr_count[j] += 1
        
        # Now apply the update to each vertex
        for v in new_mesh.verts:
            if v.is_interior:
                v.data += update_delta * pos_update[v.idx]
                v.desired_radius += update_delta * rad_update[v.idx]
                # v.desired_radius += update_delta * rad_update[v.idx]
        
        for v in new_mesh_2D.verts:
            if new_mesh.verts[v.idx].is_interior:
                face_idx = new_mesh.verts[v.idx].found_face_idx
                v2d = vectorTransport3Dto2D(pos_update[v.idx], face_idx, packingMesh, orig_mesh)
                new_pos = v.data + update_delta * v2d
                new_rad = new_mesh.verts[v.idx].desired_radius + update_delta * rad_update[v.idx]
                result = radiusBasisHelper(new_pos, face_idx, packingMesh, orig_mesh)
                if result != None:
                    r, p3d, _, _, _, _, idx = result
                    v.data = new_pos
                    new_mesh.verts[v.idx].data = p3d
                    new_mesh.verts[v.idx].desired_radius = (1-prioritize_desired_radius_epsilon) * new_rad + prioritize_desired_radius_epsilon * 0.25 * r
                    new_mesh.verts[v.idx].found_face_idx = idx

    def spring_draw(plotter):
        nonlocal new_mesh, new_mesh_2D
        plotter.clear()
        # plotter += mesh_lines(new_mesh)
        plotter += mesh_lines(new_mesh_2D)
        # plotter += Spheres([[v.data.x, v.data.y, v.data.z] for v in new_mesh.verts], 
        #                     [v.desired_radius for v in new_mesh.verts])
        # plotter += Spheres([[v.data.x, v.data.y, v.data.z] for v in new_mesh_2D.verts], 
        #                     [v.radius for v in new_mesh_2D.verts])
        plotter += Arrows([[v.data.x, v.data.y, v.data.z] for v in new_mesh_2D.verts], 
                          [[v.data.x + v.update.x, v.data.y + v.update.y, v.data.z + v.update.z] for v in new_mesh_2D.verts], 
                          c='red')        
        plotter += Arrows([[v.data.x, v.data.y, v.data.z] for v in new_mesh.verts], 
                          [[v.data.x + v.update.x, v.data.y + v.update.y, v.data.z + v.update.z] for v in new_mesh.verts], 
                          c='green')
        plotter += Points([[v.data.x, v.data.y, v.data.z + 0.2] for v in new_mesh.verts], 
                    c='green')
        for v in new_mesh_2D.verts:
            plotter += Circle(pos=[v.data.x, v.data.y, v.data.z], r=v.radius)
        plotter.render()


    plotter = Plotter(size=(1600, 1600))
    plotter.show(
        dual_mesh_lines(new_mesh)
    ).close()

    print("Running simulation...")
    
    #(spring_2D_update_multi_factory(k=100))()
    
    plotter = AnimatedPlotter(spring_2D_update_multi_factory(k=1), spring_draw)
    plotter.show()

    plotter = Plotter(size=(1600, 1600))
    plotter.show(
        dual_mesh_lines(new_mesh)
    ).close()

    # arrow_start_pts = []
    # arrow_end_pts = []
    # for v in new_mesh.verts:
    #     if v.is_interior:
    #         _, _, _, _, _, face_idx, _ = v.sample
    #         arrow_start_pts.append(tuple(v.data))
    #         arrow_end_pts.append(tuple(v.data + pos_update[v.idx]))
    #         v2d = 0.01 * vectorTransport3Dto2D(pos_update[v.idx], face_idx, packingMesh, orig_mesh)
    #         arrow_start_pts.append(tuple(new_mesh_2D.verts[v.idx].data))
    #         arrow_end_pts.append(tuple(new_mesh_2D.verts[v.idx].data + v2d))
    
    # plt = Plotter(size = (1600, 1600))
    # show_mesh(new_mesh, 
    #     [Arrows(arrow_start_pts, arrow_end_pts), mesh_lines(new_mesh_2D)]
    # )

    
    return orig_mesh, packing, packingMesh, new_mesh

orig_mesh, packing, packingMesh, new_mesh = main()