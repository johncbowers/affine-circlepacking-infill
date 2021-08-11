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

import vedo
from vedo import Plotter
from vedo.shapes import Lines, Plane, Points, Arrows, Spheres, Circle
from vedo.mesh import Mesh

from koebe.algorithms.hypPacker import *

import random, math

####
# Helper Code
####

def circlePackingEdgeSegmentsE2(packing):
    return [SegmentE2(PointE2(e.aDart.origin.data.center.coord.real, e.aDart.origin.data.center.coord.imag), 
                      PointE2(e.aDart.dest.data.center.coord.real, e.aDart.dest.data.center.coord.imag))
            for e in packing.edges]


def leftHandTurn(p1: PointE2, p2: PointE2, p3: PointE2) -> bool:
    return (orientation2(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y) 
            != Orientation.NEGATIVE)

def face_orientation(f):
    p1, p2, p3 = [v.data for v in f.vertices()]
    return orientation2(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y)

def inTriangleFace(face, p):
    p1, p2, p3 = [v.data for v in face.vertices()]
    return leftHandTurn(p1, p2, p) and leftHandTurn(p2, p3, p) and leftHandTurn(p3, p1, p)

def getRandomPackingFace(packing):
    fIdx = random.randint(0, len(packing.faces) - 2) # The last face of a packing is the outer face
    return packing.faces[fIdx]


def pointLocateInMesh(p, packing, startFaceIndex = -1):
    """
    Uses a simple O(sqrt(F)) algorithm to find the traingle containing a point
    p in a packing mesh. Basic idea: select a starting face, then while the 
    current face does not include the point find an edge whose half-plane
    separates the point from the current face and move to the neighboring 
    face incident the edge. Stop when you find a face containing the point or
    hit the outer boundary of the mesh (this condition assumes the outer face of
    the mesh is convex and all faces are convex). 

    Pre-requisites: 
        The packing must be counter-clockwise oriented and the outer face must be the 
        last faces in the packing.faces list. 

        The outerface must be convex. 

    Parameters:
        p: PointE2 - The point to locate. 
        packing: DCEL - The packing. Each interior face should be a triangle and each vertex
                        must store a PointE2 as its .data. 
        startFaceIndex: int (OPTIONAL) - start the search from a specific face. If you are 
                        doing a lot of searches on a point set, ordering your points by proximity
                        and beginning each subsequent search from the previous speeds this algorithm
                        up to near O(1) for lots of queries. 
    Returns;
        A tuple (face, face_idx, search_faces) where face is the face of the mesh
        containing the point p and face_idx is its index or face = None and face_idx = -1 
        if no such face exists and search_faces is the list of face indices probed by
        the point location algorithm. 
    """
    if startFaceIndex == -1:
        startFaceIndex = random.randint(0, len(packing.faces) - 2)
    
    search_faces = [startFaceIndex]
    searchIdx = startFaceIndex
    
    while (searchIdx != packing.outerFace.idx and not inTriangleFace(packing.faces[searchIdx], p)):

        # Find a dart that makes a right hand turn
        found_face_idx = -1
        for d in packing.faces[searchIdx].darts():
            p0, p1 = d.origin.data, d.dest.data
            if not leftHandTurn(p0, p1, p):
                found_face_idx = d.twin.face.idx
                break
        
        if found_face_idx != -1:
            # We should have found one unless the mesh is oriented wrong
            search_faces.append(found_face_idx)
            searchIdx = found_face_idx
        else:
            raise ValueError("Mesh is not oriented properly.")

    if searchIdx == packing.outerFace.idx:
        return None, -1, search_faces
    else:
        return packing.faces[searchIdx], searchIdx, search_faces

def radiusBasisHelper(p2d: PointE3, 
                      search_idx: int, 
                      mesh2d: DCEL, 
                      mesh3d: DCEL):
    """
    Given a 2D point, p2d. First finds the triangle in mesh2d containing
    the point starting the search in mesh2d at the face with index search_idx. 
    Then lifts the point to the corresponding face in mesh3d, calculates a basis there
    and then pushes the basis back to 2D and returns the basis vectors. 

    Parameters:
        p2d: A point in the xy-plane given as a PointE2 object. 
        search_idx: an index of a face in mesh2d to start the point location search from
        mesh2d: The 2d mesh (the .data on each vertex should be PointE2 objects in the xy plane)
        mesh3d: The corresponding 3d mesh. The mesh should contain .desired_radius data at each vertex. 
    
    Returns:
        Either: 
        (r, p3d, e1, e2, e1p, e2p, idx)

        Where 
        r is the desired radius in 3D. 
        p3d is the lifted point in 3d
        (e1, e2) are the basis vectors in 2D that are the image of the triangle basis
        vectors of the desired radius at the point in 3D under the affine
        transformation mapping that triangle to its corresponding 2D triangle
        in mesh2d. 
        (e1p, e2p) are the orthonormal basis in 3D with lengths given by
        the desired radius. 

        idx is the index of the face of mesh2d containing p2d. 

        OR

        None if the point is not on the mesh2d. 
    """
    theFace, idx, _ = pointLocateInMesh(p2d, mesh2d, search_idx)

    if theFace == None or theFace == mesh2d.outerFace:
        return None
    
    u, v, w = [v.data for v in theFace.vertices()]
    mu, mv, mw = [v.data for v in mesh3d.faces[idx].vertices()]

    A, B, C = PointE3(u.x, u.y, 0), PointE3(v.x, v.y, 0), PointE3(w.x, w.y, 0)
    P = PointE3(p2d.x, p2d.y, 0)

    Pp = barycentricTransformE3(P, A, B, C, mu, mv, mw)
    
    alpha, beta, gamma = barycentricCoordinatesOfE3(P, A, B, C)
    r1, r2, r3 = [mesh3d.verts[v.idx].desired_radius for v in theFace.vertices()]
    r = alpha * r1 + beta * r2 + gamma * r3
    
    e1, e2 = triangleBasis(mu, mv, mw)
    e1p = barycentricTransformE3(mu + e1, mu, mv, mw, A, B, C) - A
    e2p = barycentricTransformE3(mu + e2, mu, mv, mw, A, B, C) - A

    return (r, Pp, e1, e2, e1p, e2p, idx)

def vectorTransport3Dto2D(v: VectorE3, face_idx: int, mesh2d: DCEL, mesh3d: DCEL):

    u, v, w     = [v.data for v in mesh2d.faces[face_idx].vertices()]
    mu, mv, mw  = [v.data for v in mesh3d.faces[face_idx].vertices()]

    return barycentricTransformE3(mu + v, mu, mv, mw, u, v, w) - u


############################################################
# Poisson sampling code
############################################################

def annulusSample2DAffine(Rmin, Rmax, e1, e2):
    from random import random
    rm = Rmin / Rmax
    m = rm * rm
    r = ((1 - m) * random() + m)**(1/2)
    x, y = random()*2-1, random()*2-1
    l = r*Rmax / (x*x + y*y)**(1/2)
    return tuple(l*x*e1 + l*y*e2)

def ptToBin2D(p, cellSize):
    return int(p[0] // cellSize), int(p[1] // cellSize)

def basisAABB(p, e1, e2):
    A = p + e1 + e2
    B = p + e1 - e2
    C = p - e1 + e2
    D = p - e1 - e2
    xs = [A.x, B.x, C.x, D.x]
    ys = [A.y, B.y, C.y, D.y]
    return (min(xs), max(xs), min(ys), max(ys))

def overlapBlocks2D(p, e1, e2, cellSize):
    minx, maxx, miny, maxy = basisAABB(p, e1, e2)
    i, j = int(minx // cellSize), int(miny // cellSize)
    I, J = int(maxx // cellSize), int(maxy // cellSize)
    return [(x, y) for x in range(i, I+1) for y in range(j, J+1)]

def distSq2D(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy

def inv_matrix(e1, e2):
    a, b = e1.x, e2.x
    c, d = e1.y, e2.y
    inv_det = 1.0 / (a * d - b * c)
    return ((d * inv_det, -b * inv_det), (-c * inv_det, a * inv_det))

def mat_mul(M, p):
    ((a, b), (c, d)), (x, y) = M, p
    return (a*x + b*y, c*x + d*y)

def self_dot_sq(p):
    (x, y) = p
    return x*x + y*y

def pt_sub(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return (x1 - x2, y1 - y2)

def pt_add(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return (x1 + x2, y1 + y2)

def samples_are_too_close(s1, s2):
    (p1, r1, _, _, M1, _, _) = s1
    (p2, r2, _, _, M2, _, _) = s2
    result = (self_dot_sq(mat_mul(M1, pt_sub(p2, p1))) < r1 * r1 or
            self_dot_sq(mat_mul(M2, pt_sub(p1, p2))) < r2 * r2)
    return result
    
rejected_samples = []

SAMPLE_TYPE_FACE = 0
SAMPLE_TYPE_EDGE = 1

def adaptiveAffinePoissonSampling2D(cellSize, mesh2d, mesh3d, k=100):
    global rejected_samples, SAMPLE_TYPE_FACE, SAMPLE_TYPE_EDGE

    # Data structures to track samples
    samples = [] # List of all samples in the output
    active  = [] # "Active" samples that can still be used to generate new samples
    grid    = {} # Geometrically hashed samples for fast collisision detection

    # Inline helper functions

    # Just doing this for a little help with error checking. 
    def sample_record(p, 
                      r, 
                      e1, 
                      e2, 
                      inv_mat, 
                      elem_idx, 
                      type):
        """
        Record constructor to create a sample record, since Python has no 
        record type or static typing (don't tell me "use type hints", they 
        are too poor for a variety of reasons) and is generally annoying to 
        debug as a result. 
        """
        return (p, r, e1, e2, inv_mat, elem_idx, type)

    def insertSample(sample) -> None:
        """ Inserts a sample into the grid structure. """
        nonlocal samples, active, grid, cellSize
        samples.append(sample)
        active.append(sample)
        # Find the grid ID. 
        ptBin = ptToBin2D(sample[0], cellSize)
        if ptBin not in grid: 
            grid[ptBin] = [sample]
        else: 
            grid[ptBin].append(sample)

    def isValidSample(sample) -> bool:
        nonlocal cellSize, grid
        (p, _, e1, e2, M, _, _) = sample
        x, y = p
        # We are only sampling within the unit square
        if x*x + y*y > 1:
            return False
        for qBin in overlapBlocks2D(PointE3(p[0], p[1], 0), e1, e2, cellSize):
            if qBin in grid:
                for grid_sample in grid[qBin]:
                    if samples_are_too_close(sample, grid_sample):
                        return False
        return True
    
    def sample_from_point(p, initial_search_face_idx):
        nonlocal mesh2d, mesh3d
        result = radiusBasisHelper(PointE2(p[0], p[1]), initial_search_face_idx, mesh2d, mesh3d)
        if result == None:
            return None
        r, _, _, _, e1, e2, face_idx  = result
        return sample_record((p[0], p[1]), r, e1, e2, inv_matrix(e1, e2), face_idx, SAMPLE_TYPE_FACE)

    p0 = (0.0001, 0.000001) # (TODO) this is the initial point, should be importance sampled. 
    s0 = sample_from_point(p0, 0)
    insertSample(s0)

    while len(active) > 0 and len(samples) < 10000:
        print(len(samples), len(active))
        # pick a random sample from the active list:
        sIdx = math.floor(random.random() * len(active))
        s    = active[sIdx]

        # Unpack the radius, basis vectors, and face index of s
        p, r, e1, e2, _, fIdx, _ = s

        r2 = 2 * r # Cache the value of 2*r

        # try to find a point between radius and 2*radius that does not
        # conflict with any other points. 
        found = False
        #print("Round")

        rejected_samples = []
        for _ in range(k):
            #print("\t sampling", r, r2, e1, e2)
            #print("p: ", p)
            x, y, _ = annulusSample2DAffine(r, r2, e1, e2)
            #print("theSample: ", theSample)
            s2 = sample_from_point(pt_add(p, (x,y)), fIdx)
            if s2 != None and isValidSample(s2):
                insertSample(s2)
                found = True
                break
            else:
                rejected_samples.append(s2)
            
        # If no point was found within k tries, we remove the current
        # point from the active list
        if not found:
            # O(1) removal of the point from the active list
            active[sIdx], active[-1] = active[-1], active[sIdx]
            active.pop()
    
    return samples
    

############################################################
# Helpers for dealing with scipy's Delaunay hull objects
############################################################

def hull_to_edge_map(hull):
    """
    Extracts from a scipy hull object (as returned by the Delaunay function
    in scipy.spatial) a list of edge indices with each edge represented exactly
    once by a pair (i, j). 

    Also returns a mapping of each edge to a list containing its neighboring faces. 
    """
    edges = set()
    edges_to_faces = {}

    def addEdge(i, j, face):
        if i > j:
            i, j = j, i
        edges.add((i, j))
        if (i, j) in edges_to_faces:
            edges_to_faces[(i, j)].append(face)
        else:
            edges_to_faces[(i, j)] = [face]

    for i, j, k in hull.simplices:
        face = (i, j, k)
        addEdge(i, j, face)
        addEdge(j, k, face)
        addEdge(k, i, face)
    
    return list(edges), edges_to_faces

def hull_bary_center_face(pts, i, j, k):
    """
    Computes the bary center of a triangle with vertices pts[i], pts[j], pts[k]. 
    pts should be a list of PointE2 or PointE3 objects and 
    """
    return 0.3333333333333333 * (pts[i] + pts[j] + pts[k])

def hull_bary_center_edge(pts, i, j):
    """
    Returns the center of a segment from pts[i] to pts[j]. 
    pts should be a list of PointE2 or PointE3 objects.
    """
    return 0.5 * (pts[i] + pts[j])

def hull_dual_segs(hull, pts):
    edges, edges_to_faces = hull_to_edge_map(hull)
    dual_segs = []
    for i, j in edges:
        if len(edges_to_faces[(i, j)]) == 2:
            f1 = edges_to_faces[(i, j)][0]
            f2 = edges_to_faces[(i, j)][1]
            dual_segs.append((hull_bary_center_face(pts, *f1), hull_bary_center_face(pts, *f2)))
        else:
            f = edges_to_faces[(i, j)][0]
            dual_segs.append((hull_bary_center_face(pts, *f), hull_bary_center_edge(pts, i, j)))
    return dual_segs

def oriented_simplices(hull):
    from collections import deque
    import numpy as np
    from scipy.spatial import ConvexHull
    # For some reason the scipy hull triangles are not oriented consistently
    # this method fixes that. 
    edge_to_faces = {}

    simplices = [(i, j, k) for i, j, k in hull.simplices]

    def name(i, j):
        if i < j:
            return (i, j)
        else:
            return (j, i)

    for i, j, k in simplices:
        edge_to_faces[name(i, j)] = set()
        edge_to_faces[name(j, k)] = set()
        edge_to_faces[name(k, i)] = set()
     
    for simplex in simplices:
        i, j, k = simplex
        edge_to_faces[name(i, j)].add(simplex)
        edge_to_faces[name(j, k)].add(simplex)
        edge_to_faces[name(k, i)].add(simplex)

    def edgeSet(simplex):
        i,j,k = simplex
        return set([(i, j), (j, k), (k, i)])
    
    def compatibleOrientation(simplex1, simplex2):
        set2 = edgeSet(simplex2)
        for e in edgeSet(simplex1):
            if e in set2:
                return False
        return True
    

    simplex_to_index = {}
    for sIdx in range(len(simplices)):
        simplex_to_index[tuple(simplices[sIdx])] = sIdx

    def neighborFaceIndices(simplex):
        neighbors = []
        for e in edgeSet(simplex):
            for f in (edge_to_faces[name(*e)] - set([tuple(simplex)])):
                neighbors.append(f)
        return [simplex_to_index[tuple(n)] for n in neighbors]


    orientation = [0 for _ in simplices]
    visited = [False for _ in simplices]

    orientation[0] = 1
    Q = deque([0])

    while len(Q) > 0:
        idx = Q.popleft()
        if not visited[idx]:
            visited[idx] = True
            for nIdx in neighborFaceIndices(simplices[idx]):
                if not visited[nIdx]:
                    if not compatibleOrientation(simplices[idx], simplices[nIdx]):
                        orientation[nIdx] = orientation[idx] * -1
                    else:
                        orientation[nIdx] = orientation[idx]
                    Q.append(nIdx)
    
    final_simplices = []
    for sIdx in range(len(simplices)):
        i, j, k = simplices[sIdx]
        if orientation[sIdx] == 1:
            final_simplices.append((i, j, k))
        else:
            final_simplices.append((k, j, i))
    
    return final_simplices
            
def convexHullE3(points):
    """
    Given a list of PointE3 objects, returns the convex hull as a DCEL object.
    """
    import numpy as np
    from scipy.spatial import ConvexHull
    from koebe.datastructures.dcel import DCEL, Vertex, Dart, Face, Edge

    hull = ConvexHull(np.array([[x,y,z] for x,y,z in points]))
    simplices = oriented_simplices(hull)

    dcel = DCEL()

    # Create the vertices
    for p in points:
        Vertex(dcel, data=p)
    
    # Create the faces
    for _ in simplices:
        Face(dcel)
    
    edgeIndToObjMap = {}
    edgeList = []
    # Helper to create the edge objects and set up the map from
    # edge indices _without_ creating duplicate edges.
    def createEdge(i, j):
        nonlocal dcel, edgeIndToObjMap
        if (i, j) not in edgeIndToObjMap:
            e = Edge(dcel)
            edgeIndToObjMap[(i, j)] = e
            edgeIndToObjMap[(j, i)] = e
            edgeList.append((i, j))
        
    # Check that each edge was doubled. 
    edgeSet = set(edgeList)
    for i, j in edgeList:
        if (j, i) not in edgeSet:
            raise RuntimeError("Edge incident a single simplex returned by scipy.spatial.ConvexHull.")
    
    # Create the edges
    for i, j, k in simplices:
        createEdge(i, j)
        createEdge(j, k)
        createEdge(k, i)
    
    # Create the darts
    dartMap = {}
    for sIdx in range(len(simplices)):
        i, j, k = simplices[sIdx]
        f = dcel.faces[sIdx]
        vi, vj, vk = dcel.verts[i], dcel.verts[j], dcel.verts[k]

        dij = Dart(dcel, edge=edgeIndToObjMap[(i, j)], origin=vi, face=f)
        djk = Dart(dcel, edge=edgeIndToObjMap[(j, k)], origin=vj, face=f, prev=dij)
        dki = Dart(dcel, edge=edgeIndToObjMap[(k, i)], origin=vk, face=f, prev=djk, next=dij)
        
        dartMap[(i, j)] = dij
        dartMap[(j, k)] = djk
        dartMap[(k, i)] = dki

    
    # Set the twins
    for i, j in edgeList:
        dartMap[(i, j)].makeTwin(dartMap[(j, i)])

    return dcel

testHullx = convexHullE3([PointE3(x,y,z) for x, y, z in [[0,0,0],[0,0,1],[0,1,0],[1,0,0]]])
testHull = convexHullE3([PointE3(x,y,z) for x, y, z in [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]])


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

def label_areas(mesh):
    def calc_area(face):
        A, B, C = [v.data for v in face.vertices()]
        u = B - A
        v = C - A
        return 0.5 * u.cross(v).norm()
    for f in mesh.faces:
        if f != mesh.outerFace:
            f.area = calc_area(f)

############################################################
# Loading and visualizing the stress file
############################################################

class AnimatedPlotter:

    def __init__(self, update_fn, draw_fn, dt=16, cam=dict(pos=(0, 0, 5), focalPoint=(0, 0, 0), viewup=(0, 1, 0), distance=5)):
        self.update_fn = update_fn
        self.draw_fn = draw_fn
        self.dt = dt
        self.cam = cam
        self.plotter = vedo.Plotter(size=(1600, 1600))
    
    def show(self):
        self.plotter.addCallback("timer", self._updateAndDraw)
        self.timerId = self.plotter.timerCallback("create", dt=self.dt)
        return self.plotter.show(camera=self.cam).close()

    def _updateAndDraw(self, evt):
        self.update_fn()
        self.draw_fn(self.plotter)

# viewer = Viewer(axes=1, dt=20).initialize()

# viewer.plotter += vedo.Cube()
# viewer.plotter += vedo.Sphere(r=0.1).x(1.5)
# viewer.plotter += "Sphere color is"

# viewer.plotter.show()

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

        

    def spring_2D_update_multi_factory(k = 1):
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