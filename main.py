import numpy as np
import solidhelpers
import imagehelpers
import butterfly
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
basemap_path = str(params['Basemap'])

mapFaces = butterfly.getMapFaces(params["Butterfly"])

# Load the basemap 
if("$" in basemap_path):
    def getBasemapFiles(col, row):
        colNames = ["A", "B", "C", "D"]
        rowNames = ["1", "2"]
        return basemap_path.replace("$ROW", rowNames[row]).replace("$COL", colNames[col])
    basemap = imagehelpers.load_basemap(getBasemapFiles)
else:
    basemap = imagehelpers.load_image(basemap_path)



layoutGap = 0.01

# ----> Leafs (What you need for printing)
for idxLeaf, leaf in enumerate(mapFaces):
    for fold in butterfly.layoutCharts["startBottom"]:
        leaf[fold[1]].align_to(leaf[fold[0]], gap = params["Leafs"]["GlueingGap"])
    outlineVertices = solidhelpers.getOutlineVertices(leaf)
    center = np.mean(outlineVertices, axis = 0)
    outlineVertices -= center
    outlineVertices *= 1 + params["Leafs"]["OutlineExpansion"]
    outlineVertices += center    


    canvas, shape = imagehelpers.render(leaf.flatten(),
                                        basemap, params["Leafs"]["PP1"],
                                        background = params["Leafs"]["Background"],
                                        outline = outlineVertices)
    imagehelpers.save_image(f"images/leafs/png/{idxLeaf}.png", canvas)
    shape.save_svg(f"images/leafs/svg/{idxLeaf}.svg")

# ----> Layout (Leafs glued together)
startLeaf = butterfly.layoutCharts["leafs"][0][0]

for fold in butterfly.layoutCharts["startBottom"]:
    mapFaces[startLeaf, fold[1]].align_to(mapFaces[startLeaf, fold[0]])

def isCommonEdge(mainNodes, neighborNodes):
    return len(np.intersect1d(mainNodes, neighborNodes)) > 1

for glueing in butterfly.layoutCharts["leafs"]:
    mainLeaf = mapFaces[glueing[0]]
    neighborLeaf = mapFaces[glueing[1]]
    if(isCommonEdge(mainLeaf[3].nodes, neighborLeaf[0].nodes)):
        neighborLeaf[0].align_to(mainLeaf[3], gap = layoutGap)
        chart = butterfly.layoutCharts["startLeft"]
    elif(isCommonEdge(mainLeaf[0].nodes, neighborLeaf[3].nodes)):
        neighborLeaf[3].align_to(mainLeaf[0], gap = layoutGap)
        chart = butterfly.layoutCharts["startRight"]
    elif(isCommonEdge(mainLeaf[2].nodes, neighborLeaf[2].nodes)):
        neighborLeaf[2].align_to(mainLeaf[2], gap = layoutGap)
        chart = butterfly.layoutCharts["startBottom"]
    else:
        raise ValueError("No common edge found!")
    for fold in chart:
        neighborLeaf[fold[1]].align_to(neighborLeaf[fold[0]])
        pass

mapFaces = mapFaces[mapFaces != 0]

canvas = imagehelpers.render(mapFaces, basemap,
                             params["Layout"]["PP1"],
                             background = params["Layout"]["Background"])
imagehelpers.save_image("images/layout.png", np.rot90(canvas))


