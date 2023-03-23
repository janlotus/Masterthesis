import numpy as np
import os
# path=os.path.abspath(os.getcwd())
from system.bio_model.BoundaryCellModel import parametersBC
num_ray=parametersBC.num_ray_16
# ego2trans = np.load(os.path.join(path,"weights/ego2TransformationWts.npy"))
# heading2trans = np.load(os.path.join(path,"weights/heading2TransformationWts.npy"))
# trans2BVC = np.load(os.path.join(path,"weights/transformation2BVCWts.npy"))
ego2trans = np.load("system/bio_model/BoundaryCellModel/weights/ego2TransformationWts.npy")
heading2trans = np.load("system/bio_model/BoundaryCellModel/weights/heading2TransformationWts.npy")
trans2BVC = np.load("system/bio_model/BoundaryCellModel/weights/transformation2BVCWts.npy")
# clipping small weights makes activities sharper
#ego2trans = np.where(ego2trans >= np.max(ego2trans * 0.3), ego2trans, 0)

# rescaling as in BB-Model
ego2trans = ego2trans * 50
trans2BVC = trans2BVC * 35
heading2trans = heading2trans * 15

def calculateActivities(egocentricActivity, heading):
    '''
    calculates activity of all transformation layers and the BVC layer by multiplying with respective weight tensors
    :param egocentricActivity: 816x1 egocentric activity which was previously calculated in @BCActivity
    :param heading: 100 HDC networks activity,value under threhold is set to 0
    :return: activity of all TR layers and BVC layer
    '''
    # ego = np.reshape(egocentricActivity, 816)
    ego = np.reshape(egocentricActivity, num_ray*16)
    transformationLayers = np.einsum('i,ijk -> jk', ego, ego2trans) # 816*20

    maxTRLayers = np.amax(transformationLayers)
    transformationLayers = transformationLayers / maxTRLayers # 816*20
    headingIntermediate = np.einsum('i,jik -> jk ', heading, heading2trans) #816*20
    headingScaler = headingIntermediate[0, :]#20
    # scaledTransformationLayers = np.ones((816, 20))
    scaledTransformationLayers = np.ones((num_ray*16, 20))
    for i in range(20):
        scaledTransformationLayers[:, i] = transformationLayers[:, i] * headingScaler[i]
    bvcActivity = np.sum(scaledTransformationLayers, 1) #816
    maxBVC = np.amax(bvcActivity)
    if maxBVC!=0 and maxBVC!=None:
        bvcActivity = bvcActivity/maxBVC #816
    else:
        print("############################################")
        print("maxBVC is",maxBVC)
        print("bvcActivity is",bvcActivity)
        print("BVC is too much for computing")
        bvcActivity=0
    return transformationLayers, bvcActivity

