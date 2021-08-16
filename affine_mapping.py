from koebe.geometries.triangleFunctions import barycentricTransformE3, barycentricTransformE2, triangleBasis, barycentricCoordinatesOfE3
from koebe.datastructures.dcel import DCEL
from koebe.geometries.euclidean3 import VectorE3

def vectorTransport3Dto2D(v: VectorE3, face_idx: int, mesh2d: DCEL, mesh3d: DCEL):

    u, v, w     = [v.data for v in mesh2d.faces[face_idx].vertices()]
    mu, mv, mw  = [v.data for v in mesh3d.faces[face_idx].vertices()]

    return barycentricTransformE3(mu + v, mu, mv, mw, u, v, w) - u