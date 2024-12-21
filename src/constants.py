import numpy as np

# some useful constants for triangulation
outerFace = np.iinfo(int).max
noneFace = outerFace - 1
noneEdge = outerFace - 2
noneVertex = outerFace - 3
noneIntervalVertex = outerFace - 4