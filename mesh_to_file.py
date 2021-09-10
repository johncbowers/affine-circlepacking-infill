from koebe.datastructures.dcel import DCEL
from koebe.geometries.euclidean3 import PointE3

def save_dual_mesh(mesh:DCEL, path:str):
    def barycenter_of_face(face):
        pts = [v.data for v in face.vertices()]
        return sum(pts, PointE3(0,0,0)) * (1/len(pts))
    def barycenter_of_edge(edge):
        pts = [v.data for v in edge.endPoints()]
        return 0.5 * sum(pts, PointE3(0,0,0))
    
    out_face_verts = [barycenter_of_face(f) for f in mesh.faces]
    out_edge_verts = [barycenter_of_edge(d.edge) for d in mesh.outerFace.darts()]

    for i in range(len(out_face_verts)):
        mesh.faces[i].out_idx = i

    outerFaceDarts = mesh.outerFace.darts()
    
    for i in range(len(out_edge_verts)):
        outerFaceDarts[i].out_idx = i + len(out_face_verts)

    out_verts = out_face_verts + out_edge_verts

    def out_idx(d):
        if d.face != mesh.outerFace:
            return d.face.out_idx
        else:
            return d.out_idx
        
    out_edges = [(out_idx(e.aDart), out_idx(e.aDart.twin)) for e in mesh.edges]

    with open(path, 'w') as f:
        f.write(f"{len(out_verts)} {len(out_edges)}\n")
        for x, y, z in out_verts:
            f.write(f"{x} {y} {z}\n")
        for i, j in out_edges:
            f.write(f"{i} {j}\n")
        
