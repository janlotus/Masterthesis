import math
import numpy as np
from system.bio_model.BoundaryCellModel import parametersBC as p
from system.bio_model.BoundaryCellModel import subroutines

def headingCellsActivityTraining(heading):
    '''
    calculates HDC activity from current decoded heading direction
    :param heading: heading direction
    :return: activity vector of HDCs
    '''
    # the head direction at beginn is not 0, but 90,so minus 90 degree
    # if the head direciton at beginn is 0, then delete the heading=heading-np.pi/2
    # heading=heading-np.pi/2  # the head direction at beginn is not 0, but 90
    sig = 0.25 #0.1885  #higher value means less deviation and makes the boundray rep more accurate
    sig = p.nrHDC * sig / (2 * math.pi)
    amp = 1
    hdRes = 2 * math.pi / 100
    heading_vector = np.repeat(heading, 100)

    tuning_vector = np.linspace(0, 2 * math.pi, p.nrHDC)     # nrHdc = 100  np.arange(0, 2 * math.pi, hdRes)
    # tuning_vector = np.linspace(0.5*math.pi, 2.5 * math.pi, p.nrHDC) # to adjust the activity in allocentric frame
    # normal gaussian for hdc activity profile
    activity_vector = np.multiply(amp,
                                  (np.exp(-np.power((heading_vector - tuning_vector) / 2 * (math.pow(sig, 2)), 2))))
    return np.around(activity_vector, 5)



