import math
import numpy as np

num_ray_16=16 # if 16 or 24 or 52, problem in the plot of BC Model
# num_ray_16=51 # after change this, should first MakeWeights.py under BoundaryCellmodel
num_ray_16plus2=num_ray_16+2 # considering the head direction and goal vector
# num_ray_51=51
# num_travel_dir will be used in perform_lookahead_directed under linearLookahead.py
# to check the 4 Direction.
num_travel_dir=int(num_ray_16/4)
hRes = 0.5
maxR = 16
maxX = 12.5
maxY = 6.25
minX = -12.5
minY = -12.5
polarDistRes = 1
resolution = 0.2  # I dont really know what that is for
# polarAngularResolution = (2 * math.pi) / num_ray_plus2  # 51 BCs distributed around the circle
polarAngularResolution = (2 * math.pi) / num_ray_16 # 16, 51 BCs distributed around the circle
hSig = 0.5
nrHDC = 100  # Number of HD neurons
nrSteps = 10000 # training steps
hdActSig = 0.1885
angularDispersion = 0.2236

radialRes = 1
maxRadius = 16
nrBCsRadius = round(maxRadius / radialRes)

transformationRes = math.pi / 10  # for 20 Layers
nrTransformationLayers = int((2 * math.pi) // transformationRes)

transformationAngles = np.linspace(0, 2*math.pi, 20)

nrBVCRadius = round(maxR / polarDistRes)
nrBVCAngle = ((2 * math.pi - 0.01) // polarAngularResolution) + 1
nrBVC = int(nrBVCRadius * nrBVCAngle)

######Simulation#########
#set sensor length from simulation the longer the narrower space appears in the same environment
rayLength = 2.5
scalingFactorK = maxRadius / rayLength
