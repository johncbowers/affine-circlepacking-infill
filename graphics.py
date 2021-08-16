from koebe.datastructures.dcel import DCEL
from koebe.geometries.euclidean2 import PointE2, SegmentE2
from koebe.geometries.euclidean3 import PointE3

from vedo import Plotter
from vedo.shapes import Lines, Plane, Points, Arrows, Spheres, Circle
from vedo.mesh import Mesh

import numpy as np

def circlePackingEdgeSegmentsE2(packing):
    return [SegmentE2(PointE2(e.aDart.origin.data.center.coord.real, e.aDart.origin.data.center.coord.imag), 
                      PointE2(e.aDart.dest.data.center.coord.real, e.aDart.dest.data.center.coord.imag))
            for e in packing.edges]

def show_mesh(mesh: DCEL, additional_objs = []):
    meshEdgePoints  = [[v.data for v in e.endPoints()] for e in mesh.edges]
    meshStartPoints = [[p.x, p.y, p.z] if type(p) == PointE3 else [p.x, p.y, 0] for p, _ in meshEdgePoints]
    meshEndPoints   = [[p.x, p.y, p.z] if type(p) == PointE3 else [p.x, p.y, 0] for _, p in meshEdgePoints]
    plt = Plotter(size = (1600, 1600))
    if len(additional_objs) > 0:
        return plt.show(
            Lines(meshStartPoints, meshEndPoints, lw=4), 
            *additional_objs
        ).close()
    else:
        return plt.show(
            Lines(meshStartPoints, meshEndPoints, lw=4)
        ).close()

def mesh_lines(mesh:DCEL):
    meshEdgePoints  = [[v.data for v in e.endPoints()] for e in mesh.edges]
    meshStartPoints = [[p.x, p.y, p.z] if type(p) == PointE3 else [p.x, p.y, 0] for p, _ in meshEdgePoints]
    meshEndPoints   = [[p.x, p.y, p.z] if type(p) == PointE3 else [p.x, p.y, 0] for _, p in meshEdgePoints]
    return Lines(meshStartPoints, meshEndPoints, lw=4)

def dual_mesh_lines(mesh:DCEL):
    def barycenter_of_face(face):
        pts = [v.data for v in face.vertices()]
        return sum(pts, PointE3(0,0,0)) * (1/len(pts))
    def barycenter_of_edge(edge):
        pts = [v.data for v in edge.endPoints()]
        return 0.5 * sum(pts, PointE3(0,0,0))
    def dualSeg(edge):
        nonlocal mesh
        p1 = barycenter_of_face(edge.aDart.face) if edge.aDart.face != mesh.outerFace else barycenter_of_edge(edge)
        p2 = barycenter_of_face(edge.aDart.twin.face) if edge.aDart.twin.face != mesh.outerFace else barycenter_of_edge(edge)
        return [p1, p2]
    segs = [dualSeg(e) for e in mesh.edges]
    meshStartPoints = [[p.x, p.y, p.z] for p, _ in segs]
    meshEndPoints   = [[p.x, p.y, p.z] for _, p in segs]
    return Lines(meshStartPoints, meshEndPoints, lw=4, c='black')

def show_mesh_and_samples(mesh: DCEL, samples):
    
    locs  = [[v.data.x, v.data.y, v.data.z] for v in mesh.verts]
    radii = [r for _, r, _, _, _, _, _ in samples]
    
    plt = Plotter(size = (1600, 1600))
    return plt.show(
        mesh_lines(mesh), 
        Spheres(centers=locs, r=radii)
    ).close()

def show_dual_mesh(mesh):
    None

def show_points(points):
    plt = Plotter(size = (1600, 1600))
    return plt.show(
        Points(np.array([list(p) for p in points]), r=10, c='red')
    ).close()