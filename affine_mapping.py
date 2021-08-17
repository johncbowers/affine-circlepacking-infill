from koebe.geometries.triangleFunctions import barycentricTransformE3, barycentricTransformE2, triangleBasis, barycentricCoordinatesOfE3
from koebe.datastructures.dcel import DCEL
from koebe.geometries.euclidean3 import VectorE3

def proj(a: VectorE3, b: VectorE3):
    """
    Projects vector a onto vector b. 
    """
    return (a.dot(b) / b.dot(b)) * b

def vectorProjAndTransportTriangles(theVec: VectorE3, A, B, C, a, b, c):
    """
    Projects theVec onto the subspace containing the triangle ABC, then pushes the vector 
    forwards through the affine map from ABC to abc to obtain a new vector in abc's tangent
    space with the same barycentric coordinates as theVec in ABC. 
    """
    u = B - A
    v = C - A
    vHat = v - proj(v, u)
    theProjVec = proj(theVec, u) + proj(theVec, vHat)
    return barycentricTransformE3(A + theProjVec, A, B, C, a, b, c) - a

def vectorTransport3Dto2D(theVec: VectorE3, face_idx: int, mesh2d: DCEL, mesh3d: DCEL):

    u, v, w     = [v.data for v in mesh2d.faces[face_idx].vertices()]
    mu, mv, mw  = [v.data for v in mesh3d.faces[face_idx].vertices()]

    # theProjVec = proj(theVec, mv - mu) + proj(theVec, mw - mu) + mu

    # return barycentricTransformE3(mu + theProjVec, mu, mv, mw, u, v, w) - u

    return vectorProjAndTransportTriangles(theVec, mu, mv, mw, u, v, w)