import math
import random

from koebe.geometries.euclidean2 import PointE2
from koebe.geometries.euclidean3 import PointE3

from koebe.datastructures.dcel import DCEL
from koebe.geometries.triangleFunctions import barycentricTransformE3, barycentricTransformE2, triangleBasis, barycentricCoordinatesOfE3

from point_locate import pointLocateInMesh

def annulusSample2DAffine(Rmin, Rmax, e1, e2):
    from random import random
    rm = Rmin / Rmax
    m = rm * rm
    r = ((1 - m) * random() + m)**(1/2)
    x, y = random()*2-1, random()*2-1
    l = r*Rmax / (x*x + y*y)**(1/2)
    return tuple(l*x*e1 + l*y*e2)

def distSq2D(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy

def inv_matrix(e1, e2):
    a, b = e1.x, e2.x
    c, d = e1.y, e2.y
    inv_det = 1.0 / (a * d - b * c)
    return ((d * inv_det, -b * inv_det), (-c * inv_det, a * inv_det))

rejected_samples = []

SAMPLE_TYPE_FACE = 0
SAMPLE_TYPE_EDGE = 1

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


def samples_are_too_close(s1, s2):
    (p1, r1, _, _, M1, _, _) = s1
    (p2, r2, _, _, M2, _, _) = s2
    result = (self_dot_sq(mat_mul(M1, pt_sub(p2, p1))) < r1 * r1 or
            self_dot_sq(mat_mul(M2, pt_sub(p1, p2))) < r2 * r2)
    return result
    
def pt_add(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return (x1 + x2, y1 + y2)


def ptToBin2D(p, cellSize):
    return int(p[0] // cellSize), int(p[1] // cellSize)


def overlapBlocks2D(p, e1, e2, cellSize):
    minx, maxx, miny, maxy = basisAABB(p, e1, e2)
    i, j = int(minx // cellSize), int(miny // cellSize)
    I, J = int(maxx // cellSize), int(maxy // cellSize)
    return [(x, y) for x in range(i, I+1) for y in range(j, J+1)]


def basisAABB(p, e1, e2):
    A = p + e1 + e2
    B = p + e1 - e2
    C = p - e1 + e2
    D = p - e1 - e2
    xs = [A.x, B.x, C.x, D.x]
    ys = [A.y, B.y, C.y, D.y]
    return (min(xs), max(xs), min(ys), max(ys))


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

def adaptiveAffinePoissonSampling2D(cellSize, mesh2d, mesh3d, k=100):
    global SAMPLE_TYPE_FACE, SAMPLE_TYPE_EDGE

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


