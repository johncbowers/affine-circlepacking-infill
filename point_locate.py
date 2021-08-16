from koebe.geometries.euclidean2 import PointE2
from koebe.geometries.commonOps import orientation2, Orientation

import random

def leftHandTurn(p1: PointE2, p2: PointE2, p3: PointE2) -> bool:
    return (orientation2(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y) 
            != Orientation.NEGATIVE)

def inTriangleFace(face, p):
    p1, p2, p3 = [v.data for v in face.vertices()]
    return leftHandTurn(p1, p2, p) and leftHandTurn(p2, p3, p) and leftHandTurn(p3, p1, p)

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