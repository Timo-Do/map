import numpy as np
import geometryhelpers
import maphelpers

layoutCharts = {
    # Faces
    'startLeft': [[0, 1], [1, 2], [2, 3], [0, 4], [1, 5], [3, 6]],
    'startRight': [[3, 2], [2, 1], [1, 0], [0, 4], [1, 5], [3, 6]],
    'startBottom': [[2, 3], [2, 1], [1, 0], [0, 4], [1, 5], [3, 6]],
    # Leaves
    'leafs': [(1, 0), (1, 5), (5, 6), (5, 4), (6, 7), (6, 2), (2, 3)],
}


def createOctahedron():
    vertices = np.array([
        # tips
        [  0,  0,  1],
        [  0,  0, -1],
        #base
        [  1,  0,  0],
        [  0,  1,  0],
        [ -1,  0,  0],
        [  0, -1,  0]
    ])
    return vertices

def getLeavesWithFaces(tips, base, params):
    s = params['s']
    a = params['a']
    levels = np.zeros(7, dtype = np.ndarray)
    for idxLevel in [0, -1]:
        levels[idxLevel] = s * tips[idxLevel]

    for idxLevel in [1, 2, -3, -2]:
        levels[idxLevel] = np.zeros((4, 3))
        for idxCorner in range(4):
            start = tips[int(idxLevel < 0)]
            end = base[idxCorner, :]
            if(idxLevel in (1, -2)):
                frac = a
            else:
                frac = (1 - a)
            levels[idxLevel][idxCorner] = start + frac * (end - start)

    levels[3] = np.zeros((12, 3))
    for leftSide in range(4):
        start = base[leftSide, :]
        end = base[(leftSide + 1) % 4]
        levels[3][leftSide * 3 + 0] = s * start
        levels[3][leftSide * 3 + 1] = start + a * (end - start)
        levels[3][leftSide * 3 + 2] = start + (1 - a) * (end - start)

    vertices = np.vstack([level for level in levels])

    leaves = np.zeros((8, 7, 3), dtype=np.uint8)
    def level2idx(level, num = 0):
        vertexCount = [1, 4, 4, 12, 4, 4, 1]
        idx = num
        for count in vertexCount[:level]:
            idx += count
        return idx
    
    for leaf in range(8):
    # The first 4 leaves are the "upper" faces, the last 4 are the "lower" faces
        if(leaf < 4):
            tip = 0
            upper = 1
            lower = 2
        else:
            tip = -1
            upper = -2
            lower = -3
        leftSide = leaf % 4
        rightSide = (leftSide + 1) % 4
        leftLev3 = leftSide * 3
        rightLev3 = (leftSide + 1) * 3 % 12
    
        # Face (0)
        leaves[leaf, 0, 0] = level2idx(upper, leftSide)
        leaves[leaf, 0, 1] = level2idx(lower, leftSide)
        leaves[leaf, 0, 2] = level2idx(3, leftLev3 + 1)

        # Face (1)
        leaves[leaf, 1, 0] = level2idx(upper, leftSide)
        leaves[leaf, 1, 1] = level2idx(upper, rightSide)
        leaves[leaf, 1, 2] = level2idx(3, leftLev3 + 1)

        # Face (2)
        leaves[leaf, 2, 0] = level2idx(3, leftLev3 + 1)
        leaves[leaf, 2, 1] = level2idx(upper, rightSide)
        leaves[leaf, 2, 2] = level2idx(3, leftLev3 + 2)

        # Face (3)
        leaves[leaf, 3, 0] = level2idx(lower, rightSide)
        leaves[leaf, 3, 1] = level2idx(upper, rightSide)
        leaves[leaf, 3, 2] = level2idx(3, leftLev3 + 2)

        # Face (4)
        leaves[leaf, 4, 0] = level2idx(3, leftLev3 + 0)
        leaves[leaf, 4, 1] = level2idx(3, leftLev3 + 1)
        leaves[leaf, 4, 2] = level2idx(lower, leftSide)

        # Face (5)
        leaves[leaf, 5, 0] = level2idx(upper, leftSide)
        leaves[leaf, 5, 1] = level2idx(upper, rightSide)
        leaves[leaf, 5, 2] = level2idx(tip)

        # Face (6)
        leaves[leaf, 6, 0] = level2idx(3, leftLev3 + 2)
        leaves[leaf, 6, 1] = level2idx(3, rightLev3)
        leaves[leaf, 6, 2] = level2idx(lower, rightSide)
    return vertices, leaves

def getMapFaces(params : dict):
    alpha = np.deg2rad(params['rotation']['alpha'])
    beta = np.deg2rad(params['rotation']['beta'])
    gamma = np.deg2rad(params['rotation']['gamma'])

    yaw, pitch, roll = geometryhelpers.getRotationMatrices(alpha, beta, gamma)

    octahedron = createOctahedron() @ pitch
    tips = octahedron[:2]
    base = octahedron[2:]

    vertices, leaves = getLeavesWithFaces(tips, base, params)
    vertices = vertices @ yaw @ roll
    mapFaces = np.zeros((8, 7), dtype=maphelpers.Face)
    for idxLeaf in np.arange(8):
        leaf = leaves[idxLeaf]
        for idxFace, face in enumerate(leaf):
            mapFaces[idxLeaf, idxFace] = maphelpers.Face(face, vertices[face])

    return mapFaces