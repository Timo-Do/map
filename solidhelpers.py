import numpy as np
from matplotlib import pyplot as plt

def plot_solid(vertices, faces):
    fig = plt.figure(figsize=(8,5))
    ax = plt.axes(projection='3d')

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    ax.plot_trisurf(x, y, z, triangles=faces,
                        cmap='viridis', alpha=0.2, edgecolor='k')
    ax.set_box_aspect([np.ptp(i) for i in [x, y, z]])
    plt.show()


def getOutlineVertices(leaf : np.ndarray):
    outlineNodes = []
    for face in leaf:
        for idxEdge in range(3):
            edge = [face.nodes[idxEdge % 3], face.nodes[(idxEdge + 1) % 3]]
            isInOutline = False
            for e in [edge, edge[::-1]]:
                if(e in outlineNodes):
                    outlineNodes.remove(e)
                    isInOutline = True
            if(not isInOutline):
                outlineNodes.append(edge)
    
    outline = [outlineNodes[0][0], outlineNodes[0][1]]
    outlineNodes.remove(outlineNodes[0])
    while(len(outlineNodes) > 0):
        for line in outlineNodes:
            if(line[0] == outline[-1]):
                outline.append(line[1])
                outlineNodes.remove(line)
            elif(line[1] == outline[-1]):
                outline.append(line[0])
                outlineNodes.remove(line)
    outline.remove(outline[-1])
    outlineVertices = np.zeros((len(outline), 2))
    for idxOutlineNode, outlineNode in enumerate(outline):
        for face in leaf:
            for faceNodeIdx, faceNode in enumerate(face.nodes):
                if(faceNode == outlineNode):
                    outlineVertices[idxOutlineNode] = face.triangle_on_plane.vertices[faceNodeIdx]
    return outlineVertices