import coordinates
import matplotlib.pyplot as plt
import numpy as np

coords=coordinates.coordinates()

coords.loadCoordinates(path="K:\\Datasets\\sampleNiftiCoordinates\\",prefix="")

coords.initialize()
coords.meanArcLength()
