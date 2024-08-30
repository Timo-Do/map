import numpy as np
import bases
import geometryhelpers
import maphelpers
import imagehelpers


s = 0.8
a = 0.24
PP1Leaf = 300
PP1Layout = 300
alpha = np.deg2rad(206)
beta = np.deg2rad(45)
gamma = np.deg2rad(0)

yaw, pitch, roll = geometryhelpers.getRotationMatrices(alpha, beta, gamma)

# Base Octahedron (pitched)
octahedron = bases.createOctahedron() @ pitch
tips = octahedron[:2, :]
base = octahedron[2:, :]


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

vertices = np.vstack([level for level in levels]) @ yaw @ roll



def level2idx(level, num = 0):
    vertexCount = [1, 4, 4, 12, 4, 4, 1]
    idx = num
    for count in vertexCount[:level]:
        idx += count
    return idx

leaves = np.zeros((8, 7, 3), dtype=np.uint8)
mapFaces = np.zeros((8, 7), dtype=maphelpers.Face)
for leaf in range(8):
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

    # Face (1)
    leaves[leaf, 0, 0] = level2idx(upper, leftSide)
    leaves[leaf, 0, 1] = level2idx(lower, leftSide)
    leaves[leaf, 0, 2] = level2idx(3, leftLev3 + 1)

    # Face (2)
    leaves[leaf, 1, 0] = level2idx(upper, leftSide)
    leaves[leaf, 1, 1] = level2idx(upper, rightSide)
    leaves[leaf, 1, 2] = level2idx(3, leftLev3 + 1)

    # Face (3)
    leaves[leaf, 2, 0] = level2idx(3, leftLev3 + 1)
    leaves[leaf, 2, 1] = level2idx(upper, rightSide)
    leaves[leaf, 2, 2] = level2idx(3, leftLev3 + 2)

    # Face (4)
    leaves[leaf, 3, 0] = level2idx(lower, rightSide)
    leaves[leaf, 3, 1] = level2idx(upper, rightSide)
    leaves[leaf, 3, 2] = level2idx(3, leftLev3 + 2)

    # Face (5)
    leaves[leaf, 4, 0] = level2idx(3, leftLev3 + 0)
    leaves[leaf, 4, 1] = level2idx(3, leftLev3 + 1)
    leaves[leaf, 4, 2] = level2idx(lower, leftSide)

    # Face (6)
    leaves[leaf, 5, 0] = level2idx(upper, leftSide)
    leaves[leaf, 5, 1] = level2idx(upper, rightSide)
    leaves[leaf, 5, 2] = level2idx(tip)

    # Face (7)
    leaves[leaf, 6, 0] = level2idx(3, leftLev3 + 2)
    leaves[leaf, 6, 1] = level2idx(3, rightLev3)
    leaves[leaf, 6, 2] = level2idx(lower, rightSide)

#bases.plot_solid(vertices, leaves.reshape(-1, 3))

basemap = imagehelpers.load_image("images/blue_marble_august_small.png")
mapFaces = np.zeros((8, 7), dtype=maphelpers.Face)

# alignment charts for each leaf
startLeft = [[0, 1], [1, 2], [2, 3], [0, 4], [1, 5], [3, 6]]
startRight = [[3, 2], [2, 1], [1, 0], [0, 4], [1, 5], [3, 6]]
startBottom = [[2, 3], [2, 1], [1, 0], [0, 4], [1, 5], [3, 6]]
# alignment chart for the entire layout
layoutChart =  [(1, 0), (1, 5), (5, 6), (5, 4), (6, 7), (6, 2), (2, 3)]
layoutGap = 0.01
for idxLeaf in np.arange(8):
    leaf = leaves[idxLeaf]
    for idxFace, face in enumerate(leaf):
        mapFaces[idxLeaf, idxFace] = maphelpers.Face(face, vertices[face])

# get basemap
def getBasemapFiles(col, row):
    colNames = ["A", "B", "C", "D"]
    rowNames = ["1", "2"]
    #return f"images/base/real/world.200408.3x21600x21600.{colNames[col]}{rowNames[row]}.jpg"
    return f"images/base/test/{colNames[col]}{rowNames[row]}.png"

basemap = imagehelpers.load_basemap(getBasemapFiles)


# align each leaf and render it for printing
for idxLeaf, leaf in enumerate(mapFaces):
    for fold in startBottom:
        leaf[fold[1]].align_to(leaf[fold[0]])
    canvas = imagehelpers.render(leaf.flatten(), basemap, PP1Leaf, background = [17, 17, 17, 255])
    imagehelpers.save_image(f"images/{idxLeaf}.png", np.rot90(canvas))

# now do the layout

startLeaf = layoutChart[0][0]

for fold in startBottom:
    mapFaces[startLeaf, fold[1]].align_to(mapFaces[startLeaf, fold[0]])

def isCommonEdge(mainNodes, neighborNodes):
    return len(np.intersect1d(mainNodes, neighborNodes)) > 1

for glueing in layoutChart:
    mainLeaf = mapFaces[glueing[0]]
    neighborLeaf = mapFaces[glueing[1]]
    if(isCommonEdge(mainLeaf[3].nodes, neighborLeaf[0].nodes)):
        neighborLeaf[0].align_to(mainLeaf[3], gap = layoutGap)
        chart = startLeft
    elif(isCommonEdge(mainLeaf[0].nodes, neighborLeaf[3].nodes)):
        neighborLeaf[3].align_to(mainLeaf[0], gap = layoutGap)
        chart = startRight
    elif(isCommonEdge(mainLeaf[2].nodes, neighborLeaf[2].nodes)):
        neighborLeaf[2].align_to(mainLeaf[2], gap = layoutGap)
        chart = startBottom
    else:
        raise ValueError("No common edge found!")
    for fold in chart:
        neighborLeaf[fold[1]].align_to(neighborLeaf[fold[0]])
        pass



mapFaces = mapFaces[mapFaces != 0]

canvas = imagehelpers.render(mapFaces, basemap, PP1Layout)
imagehelpers.save_image("images/layout.png", np.rot90(canvas))


