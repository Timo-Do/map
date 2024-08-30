import imagehelpers
import numpy as np

base = imagehelpers.load_image("images/blue_marble_august_small.png")
rows = ["1", "2"]
cols = ["A", "B", "C", "D"]
splitRows = np.split(base, len(rows), axis = 0)

for idxRow, row in enumerate(splitRows):
    for idxCol, part in enumerate(np.split(row, len(cols), axis = 1)):
        imagehelpers.save_image(f"images/base/test/{cols[idxCol]}{rows[idxRow]}.png", part)