import numpy as np
from scipy.spatial import ConvexHull
from koebe.datastructures.dcel import DCEL, Vertex, Dart, Face, Edge

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



# TODO: Are these used anywhere? ::: I DONT THINK SO


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