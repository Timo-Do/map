import numpy as np
import bases
import geometryhelpers
import maphelpers
import imagehelpers

import yaml
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if("RUNCONFIG" in os.environ):
    logging.debug(f"Found RUNCONFIG in environment: {os.environ['RUNCONFIG']}")
    configPath = f'configs/{os.environ["RUNCONFIG"]}.yaml'
else:
    logging.info("No RUNCONFIG found in environment, using default (quick) config")
    configPath = 'configs/quick.yaml'
logging.info(f"Using config: {configPath}")
with open(configPath, 'r') as file:
    config = yaml.safe_load(file)


params = config['parameters']
s = params['s']
a = params['a']
PP1Leaf = int(params['PP1']['Leafs'])
PP1Layout = int(params['PP1']['Layout'])
alpha = np.deg2rad(params['rotation']['alpha'])
beta = np.deg2rad(params['rotation']['beta'])
gamma = np.deg2rad(params['rotation']['gamma'])
basemap_path = str(params['Basemap'])

logging.info(f"Parameters: s = {s}, a = {a}, PP1Leaf = {PP1Leaf}, PP1Layout = {PP1Layout}, alpha = {alpha}, beta = {beta}, gamma = {gamma}, basemap = {basemap_path}")

yaw, pitch, roll = geometryhelpers.getRotationMatrices(alpha, beta, gamma)

logging.debug("Rotation matrix created")


octahedron = bases.createOctahedron() @ pitch
logging.debug("Base octahedron created, with rotation applied")
# In the base octahedron, the first two vertices are upper and lower tip (if you hold it like a diamond)
# We call the inner square the "base". See the function for details.
tips = octahedron[:2, :]
base = octahedron[2:, :]

# So one level is one slice of the octahedron (sliced horizontally)
# This is how we group the vertices. It will make sense when we draw the faces.
# We have 7 levels: 2 tips
#          * <- Level 0
#         / \
#        /   \  <- Level 1
#       /     \ 
#      /       \
#     /         \
#    /           \
#   /             \ <- Level 2
#  /               \
# *-----------------* <- Level 3
#  \               /
#   \             / <- Level -3
#    \           /
#     \         /
#      \       /
#       \     /
#        \   /  <- Level -2
#         \ /
#          * <- Level -1

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

# We name each face of the octahedron a leaf, because that's what it looks like in the end
# We have 8 leaves, each with 7 faces (which are triangles)
leaves = np.zeros((8, 7, 3), dtype=np.uint8)
mapFaces = np.zeros((8, 7), dtype=maphelpers.Face)
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
# check if basemap contains a $ character
if("$" in basemap_path):
    def getBasemapFiles(col, row):
        colNames = ["A", "B", "C", "D"]
        rowNames = ["1", "2"]
        return basemap_path.replace("$ROW", rowNames[row]).replace("$COL", colNames[col])
    basemap = imagehelpers.load_basemap(getBasemapFiles)
else:
    basemap = imagehelpers.load_image(basemap_path)
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




#basemap = np.array([[[255, 255, 255, 255]]])
#basemap = imagehelpers.load_basemap(getBasemapFiles)

def isCommonEdge(mainNodes, neighborNodes):
    return len(np.intersect1d(mainNodes, neighborNodes)) > 1

# align each leaf and render it for printing
for idxLeaf, leaf in enumerate(mapFaces):
    for fold in startBottom:
        leaf[fold[1]].align_to(leaf[fold[0]])

    outlineNodes = []
    idxOutline = 0
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

    center = np.mean(outlineVertices, axis = 0)
    outlineVertices -= center
    outlineVertices *= 1.01
    outlineVertices += center    


    canvas, shape = imagehelpers.render(leaf.flatten(), basemap, PP1Leaf, background = [17, 17, 17, 255], outline = outlineVertices)
    imagehelpers.save_image(f"images/{idxLeaf}.png", canvas)
    shape.save_svg(f"images/{idxLeaf}.svg")




# now do the layout

startLeaf = layoutChart[0][0]

for fold in startBottom:
    mapFaces[startLeaf, fold[1]].align_to(mapFaces[startLeaf, fold[0]])



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


