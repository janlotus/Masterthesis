
import math
import os # adding this line
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/" # adding this line for pycuda
import time
import numpy as np
import matplotlib
matplotlib.use('tkagg')
from matplotlib import animation
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import random
import pickle
import matplotlib as mpl
from plotting.plotResults import layout_video, error_plot, plot_current_state
from plotting.plotThesis import *
from system.controller.pybulletenvironment import PybulletEnvironment
from system.bio_model.GridcellModel.gridcellModel import GridCellNetwork
from system.bio_model.PlaceCellModel.placecellModel import PlaceCellNetwork
from system.bio_model.CognitivemapModel.cognitivemapModel import CognitiveMapNetwork
from system.bio_model.HeadDirectionCellModel.headdirectionModel import generateHDC
from system.controller.explorationPhase import compute_exploration_goal_vector
from system.controller.navigationPhase import compute_navigation_goal_vector
from system.decoder.phaseOffsetDetector import PhaseOffsetDetectorNetwork
from system.decoder.spikeDetection import SpikeDetector
from system.bio_model.HeadDirectionCellModel import placeEncoding, helper, hdcNetwork
# from system.bio_model.HeadDirectionCellModel.polarPlotter import PolarPlotter
from system.bio_model.HeadDirectionCellModel.polarPlotter_withoutconj import PolarPlotter
from system.bio_model.HeadDirectionCellModel.params import n_hdc, weight_av_stim
from system.bio_model.HeadDirectionCellModel.hdcCalibrConnectivity import ACDScale
from system.bio_model.BoundaryCellModel import BCActivity,BCsimulation,HDCActivity
from system.bio_model.BoundaryCellModel.polarBCplotter import BCplotter
from system.bio_model.BoundaryCellModel import parametersBC
from system.helper import Printsimulationtimeresult
path = os.path.abspath(os.path.join(os.getcwd(), '..'))
mpl.rcParams['animation.ffmpeg_path'] = path + "/ffmpeg/ffmpeg"
mpl.rcParams.update(mpl.rcParamsDefault)

######## exploration-->navigation  #########
# run simulation original>exploration phase
#nr_steps = 23000 # 8000 for decoder test, 15000 for maze exploration, 8000 for maze navigation
#nr_steps_exploration = 15000  # 3500 for decoder test, nr_steps for maze exploration, 0 for maze navigation
##### exploration #####
# run_sim_from_data = False
# nr_steps = 15000
# nr_steps_exploration = nr_steps
##### navigation #####
run_sim_from_data = True
nr_steps=8000
nr_steps_exploration=0
# dt = 0.05 # not work for GC firing
dt = 1e-2 #must below 0.02,otherwise problem in GC initialization
# #######################################
# initialize Grid cell Model
M = 6  # 6 for default, number of modules
n = 40  # 40 for default, size of sheet -> nr of neurons is squared
# the accept g_m*v =[0,1, 1,2], the speed of robot is 0,5m/s, so, gmin=0,2,gmax=2,4
gmin = 0.2  # 0.2 for default, maximum arena size, 0.5 -> ~10m | 0.05 -> ~105m
gmax = 2.4  # 2.4 for default, determines resolution, dont pick to high (>2.4 at speed = 0.5m/s)
gc_network = GridCellNetwork(n, M, dt, gmin, gmax=gmax, from_data=run_sim_from_data)
print('-> GridCell Network generated')

# initialize Place Cell Model
pc_network = PlaceCellNetwork(from_data=run_sim_from_data)
print('-> PlaceCell Network generated')

# initialize Cognitive Map Model
cognitive_map = CognitiveMapNetwork(dt, from_data=run_sim_from_data)
print('-> CognitiveMap generated')
visualize = True

######## simulation  #########
# env_models, available models: "maze", "plus", "box",'linear_sunburst','curved',"obstacle"
# to choose the models, we have to consider the HeadDirection Cells, Border Cells, Grid Cells
# HD Cells need Turning of agent. Border Cells need Obstacles. Grid Cells need agent going only in x or y direction
env_model='linear_sunburst'#'linear_sunburst' for default
vector_model = "linear_lookahead"  # "linear_lookahead" for default, "phase_offset_detector", "spike_detection"
pod_network = PhaseOffsetDetectorNetwork(16, 9, n) if vector_model == "phase_offset_detector" else None
spike_detector = SpikeDetector() if vector_model == "spike_detection" else None
# dt = 1e-2
env = PybulletEnvironment(visualize, env_model, dt, pod=None)
realDir = env.euler_angle[2]
if run_sim_from_data:
    idx = np.argmax(cognitive_map.reward_cells)
    gc_network.set_as_target_state(pc_network.place_cells[idx].gc_connections)

#######################################
# initialize Head Direction Model
# Set calibration mode for the place encoding feedback model
# FGL "on" True:default Stores only the ACD, cue distance and the agent's position at the first glance at the cue/landmark
# to reset HD
# FGL "off" False: Associate ACDs in every newly discovered position, to reset HD when revisiting these positions.
FirstGlanceLRN = True
viewAngle = (np.pi) / 2

# Attention: The visualization of the vectors is very slow for large matrix dimensions.
# set place encoding parameters
matrixRealLengthCovered = 4 #default is 4
# size of space in the environment that is encoded by the position encoder
# matrixDim = 3 * matrixRealLengthCovered  # place encoding granularity
matrixDim = 18 * matrixRealLengthCovered  # higher resolution,more simulation time
nextplot = time.time()
##  matplotlib visualization/plotting                   ##
# rtplot = True
rtplot_hdc=True
rtplot_bc=True
# rtplot = False
plotfps = 1.0
simple_fdbk = False
place_enc_fdbk = True
hdc = generateHDC(InitHD=realDir, place_enc_fdbk=place_enc_fdbk, simpleFdbk=simple_fdbk)  # debug
PosEncoder = placeEncoding.PositionalCueDirectionEncoder(env.cuePos[0:2], matrixDim, matrixRealLengthCovered,
                                                         FirstGlanceLRN=FirstGlanceLRN)
print('-> HeadDirection Cell generated')
# minimum neuron model timestep
# dt_neuron_min = 0.0005  # (2000Hz)
dt_neuron_min = 0.0001  # (10000Hz)
timesteps_neuron = math.ceil(dt / dt_neuron_min)
dt_neuron = dt / timesteps_neuron
# t_episode = 25
# print("neuron timesteps: {}, dt={}".format(timesteps_neuron, dt_neuron))

r2d = (360 / (2 * np.pi))  # rad to deg factor

####### noisy angular velocity ########
# use_noisy_av = False
use_noisy_av = True
# gaussian noise
# create noise for angular velocity input
# relative standard deviation (standard deviation = rel. sd * av)
# the rel.noise with 0.4 not works
noisy_av_rel_sd = 0.4
# absolute standard deviation (deg)
noisy_av_abs_sd = 0.0

# noise spikes
# average noise spike frequency in Hz
noisy_av_spike_frequency = 0.0
# average magnitude in deg/s
noisy_av_spike_magnitude = 0.0
# standard deviation in deg/s
noisy_av_spike_sd = 0.0

# noise oscillation
noisy_av_osc_frequency = 0.0
noisy_av_osc_magnitude = 0.0
noisy_av_osc_phase = 0.0

avs = []
avs_active=[]#the angular velocity only for turning
errs = []
errs_signed = []
thetas = []
# hdcnetTimes = []
robotTimes = []
plotTimes = []
hdcplotTimes=[]
transferTimes = []
decodeTimes = []
# cue_view_pos = []
goalsearchTimes=[]
GCnetTimes=[]
PCnetTimes=[]
CognitivemapnetTimes=[]
errs_noisy_signed = []
noisyDir = 0.0
# t_before = time.time()
# t_ctr = 0
#######################################

# initialize Boundary Cell Model
##################   BC-Model ###############
bcPlotter = BCplotter()
# variables needed for plotting
eBCSummed = 0
bvcSummed = 0
eBCRates = []
bvcRates = []
eachRateDiff = []
xPositions = []
yPositions = []
sampleX = []
sampleY = []
sampleT = []
bcTimes = []
BC_FOV_360=True # if agent detect 360⁰ obstacle,else only front
################## end BC-model ##################
# In this work, we only consider vector_model=="linear_lookahead""

t_episode = nr_steps*dt
nr_plots = 5  # allows to plot during the run of a simulation
nr_trials = 1 # 1 for default, 50 for decoder test, 1 for maze exploration
# video = False  # False for default, set to True if you want to create a video of the run
# need to commentar the plot_sub_goal_localization in linearLookahead.py if video=True
video = False # default false cause problem for plot_sub_goal_localization in navigation,commit that or video=False

fps = 5  # number of frames per s
step = int((1 / fps) / dt)  # after how many simulation steps a new frame is saved

if video:
    [Fig, f_gc, f_t, f_mon] = layout_video()
else:
    Fig = None

plot_matching_vectors = False  # False for default, True if you want to match spikes in the grid cel spiking plots
# Save across frames
goal_vector_array = [np.array([0, 0])]  # array to save the calculated goal vector
# save real Orientation and decoded orientation
#initialization at 90⁰,for use estimated HD in env.computer movement
decoded_orientation=[ ]
# add start orientation for using estimated HD
decoded_orientation.append(np.pi/2)
# real_orientation.append(np.pi/2)
real_xy_speed_list=[ ]
#use estimated HD for goal searching
# UseHD=False
UseHD=True
# this function performs the simulation steps and is called by video creator or manually
def getStimL(ahv):
    if ahv < 0.0:
        return 0.0
    else:
        return ahv * weight_av_stim

def getStimR(ahv):
    if ahv > 0.0:
        return 0.0
    else:
        return - ahv * weight_av_stim

def getNoisyTheta(theta):
    noisy_theta = theta
    # gaussian noise
    if noisy_av_rel_sd != 0.0:
        noisy_theta = random.gauss(noisy_theta, noisy_av_rel_sd * theta)
    if noisy_av_abs_sd != 0.0:
        noisy_theta = random.gauss(noisy_theta, noisy_av_abs_sd * dt * (1 / r2d))
    # noise spikes
    if noisy_av_spike_frequency != 0.0:
        # simplified, should actually use poisson distribution
        probability = noisy_av_spike_frequency * dt
        if random.random() < probability:
            deviation = random.gauss(noisy_av_spike_magnitude * dt * (1 / r2d),
                                     noisy_av_spike_sd * dt * (1 / r2d))
            print(deviation)
            if random.random() < 0.5:
                noisy_theta = noisy_theta + deviation
            else:
                noisy_theta = noisy_theta - deviation
    return noisy_theta

ACD_PeakActive = False
nextplot = time.time()
t_before = time.time()
t_ctr = 0
noisyDir = 0.0

if rtplot_hdc:
    plotter = PolarPlotter(n_hdc, 0.0, False, PosEncoder) # plot of HDC

def animation_frame(frame):
    if video:
        # calculate how many simulations steps to do for current frame
        start = frame - step
        end = frame
        if start < 0:
            start = 0
    else:
        # run trough all simulation steps as no frames have to be exported
        start = 0
        end = frame
    noisyDir = 0.0
    eBCSummed = 0
    bvcSummed = 0
    global hdc_active, cue_active,ACD_PeakActive
    for i in range(start, end):
        # print('the current step is--->',i)
        # robot simulation step
        beforeStep=time.time()
        # perform one simulation step
        # compute the goal_vector from rodent to goal in global coordinate system
        exploration_phase = True if i < nr_steps_exploration else False
        if exploration_phase:
            compute_exploration_goal_vector(env, i)
        else:
            compute_navigation_goal_vector(gc_network, pc_network, cognitive_map, i - nr_steps_exploration, env,
                                           pod=pod_network, spike_detector=spike_detector, model=vector_model)
        goal_vector_array.append(env.goal_vector)
        afterStep=time.time()
        goalsearchTimes.append(afterStep-beforeStep)


        action = []
        beforeStep = time.time()
        if UseHD==True:
            currentHD=decoded_orientation[-1]
        else:
            currentHD=None

        theta, agentState,delta_distance = env.compute_movement(gc_network, pc_network, cognitive_map,
                                                 exploration_phase=exploration_phase,currentHD=currentHD)
        afterStep = time.time()
        robotTimes.append((afterStep - beforeStep))
        noisy_theta = getNoisyTheta(theta)
        angVelocity = noisy_theta * (1.0 / dt) if use_noisy_av else theta * (1.0 / dt)
        # angVelocity = theta * (1.0 / dt)
        # noisy_theta = getNoisyTheta(theta)

        realDir = agentState['Orientation']
        # decodedDir=realDir
        # to reduce compute power,we don't update HDC when the agent go straightly.
        # because of noise,the agent is treated as move straightly if abs(angVelocity) < 0.015
        if abs(angVelocity) < 0.015:
            # 0.015 is the threshold according to simulation
            angVelocity = 0
            theta = 0
            noisy_theta = getNoisyTheta(theta)
            av_net = noisy_theta * (1.0 / dt) if use_noisy_av else angVelocity
            # av_net = theta * (1.0 / dt)
            # hdc_active = False
            # print('agent not turn')
            stimL = getStimL(0)
            stimR = getStimR(0)
            hdc.setStimulus('hdc_shift_left', lambda i: 0) # update self.currents
            hdc.setStimulus('hdc_shift_right', lambda i: 0)
            mapping_hdc=0
            # hdc.setStimulus('ecd_ring', lambda i: 0)
        else:
            hdc_active = True
            noisy_theta = getNoisyTheta(theta)
            av_net = noisy_theta * (1.0 / dt) if use_noisy_av else angVelocity
            # av_net=theta * (1.0 / dt)
            stimL = getStimL(av_net)
            stimR = getStimR(av_net)
            hdc.setStimulus('hdc_shift_left', lambda _: stimL)
            hdc.setStimulus('hdc_shift_right', lambda _: stimR)
            # print('agent is turning')
            mapping_hdc=1
            avs_active.append(av_net)
        thetas.append(theta)
        avs.append(angVelocity)

        ACDir = helper.calcACDir(agentState['Position'], env.cuePos[0:2])
        ECDir = helper.calcECDir(realDir, ACDir)
        cueInRange = PosEncoder.checkRange(agentState['Position'])
        # ACDir = helper.calcACDir(agentState['Position'], env.cuePos[0:2])
        # ECDir = helper.calcECDir(realDir, ACDir)
        cueInSight = helper.cueInSight(viewAngle, ECDir)
        if (cueInRange == True) and (cueInSight == True):
            # Set ECD stimuli to ECD cells
            hdcNetwork.setPeak(hdc, 'ecd_ring', ECDir)
            if (ACD_PeakActive == True):
                # decode the ACD encoded by ACD cells
                # ACD_PeakActive takes care that ACD is only derived from a fully emerged activity peak
                decodedACDDir = helper.decodeRingActivity(list(hdc.getLayer('acd_ring')))
                # Set ACD = False if ACD learned in new position
                # Set ACD = restored ACD for calibration
                ACD = PosEncoder.get_set_ACDatPos(agentState['Position'], decodedACDDir)

                if (ACD != False):
                    # Set ACD stimuli to ACD cells
                    hdcNetwork.setPeak(hdc, 'acd_ring', ACD, scale=(1 - ACDScale))
            #print('cue is in sight')
            cue_active = True
            mapping_cue = 1
            ACD_PeakActive = True
        else:
            # Set stimuli = 0 when cue is out of sight
            hdc.setStimulus('ecd_ring', lambda i: 0)
            hdc.setStimulus('acd_ring', lambda i: 0)
            mapping_cue=0
            # cue_active = False
            ACD_PeakActive = False

        if i==0:
            hdc_active=True
            cue_active=True
            Bvc_active=True

        if hdc_active or cue_active:

            # beforeStep= time.time()
            hdc.step(dt_neuron, numsteps=timesteps_neuron)

            if mapping_hdc==0:
                hdc_active=False
            else:
                hdc_active=True
            if mapping_cue==0:
                cue_active=False
            else:
                cue_active=True
            # afterStep = time.time()
            # netTimes.append((afterStep - beforeStep) / timesteps_neuron)

            # Get layer rates
            # realDir = np.pi/2 # need to adjust if the agent nor face nord
            rates_hdc = list(hdc.getLayer('hdc_attractor'))
            rates_sl = list(hdc.getLayer('hdc_shift_left'))
            rates_sr = list(hdc.getLayer('hdc_shift_right'))
            rates_ecd = list(hdc.getLayer('ecd_ring'))
            # rates_conj = list(hdc.getLayer('Conj'))
            rates_acd = list(hdc.getLayer('acd_ring'))
            # rates_conj_2 = list(hdc.getLayer('Conj2'))
            # rates_hdc2 = list(hdc.getLayer('hdc_ring_2'))

            # Calculate & save errors
            beforeStep = time.time()

            # Decode layer activities
            decodedDir = helper.decodeRingActivity(rates_hdc)
            decodedECDDir = helper.decodeRingActivity(rates_ecd)
            decodedACDDir = helper.decodeRingActivity(rates_acd)

            # Calculate direction
            afterStep = time.time()
            decodeTimes.append(afterStep - beforeStep)
            beforeStep=time.time()
            if rtplot_hdc:
                plotter.plot(rates_hdc, rates_sl, rates_sr, rates_ecd,rates_acd,
                             stimL, stimR, realDir, decodedDir)
            else:
                print('real time plot is false')
            afterStep=time.time()
            hdcplotTimes.append(afterStep-beforeStep)
        else:
            decodedDir=decoded_orientation[-1] # if not turning, the decoded Dir keep the same
            # print('live plot is not active,step not active')
        # hdcafterStep = time.time()
        # hdcnetTimes.append(hdcafterStep-hdcbeforeStep)
        decoded_orientation.append(decodedDir)
        noisyDir = (realDir + noisy_theta) % (2 * np.pi)
        realDir = (realDir + theta) % (2 * np.pi)

        err_noisy_signed_rad = helper.angleDist(realDir, noisyDir)
        errs_noisy_signed.append(r2d * err_noisy_signed_rad)
        err_signed_rad = helper.angleDist(realDir, decodedDir)
        errs_signed.append(r2d * err_signed_rad)
        errs.append(abs(r2d * err_signed_rad))

        real_xy_speed = env.xy_speeds[-1]
        decoded_orientation.append(decodedDir)
        ################################################    BC-Model    ################################################
        beforeBCStep = time.time()
        # the following may be write in fun to Calculate egoBCActvity(env.)
        raysThatHit = env.getRays()  # if not hit, then -1, else distance
        # Navigation detection start 0,but BC detection starts from 0.5pi
        # need to turn the polar angels starting from 0,so minimal from -0.5pi
        # to reduce the error, -(0.5+2pi/16rays)=-0.625
        polar_angles = np.linspace(-0.625* math.pi, 1.375 * math.pi,
                                   parametersBC.num_ray_16)  # - parametersBC.polarAngularResolution
        # in egocentric frame
        rayDistances = np.array(raysThatHit)
        rayAngles = np.where(rayDistances == -1, rayDistances, polar_angles)  # if hit, then angles, else -1

        ############### Simulation of only 180° FOV##############
        # Silences everything that is detected behind the agent
        if BC_FOV_360==False:
            for lk in range(parametersBC.num_ray_16//2):
                rayAngles[lk + parametersBC.num_ray_16//4] = -1
                rayDistances[lk + parametersBC.num_ray_16//4] = -1

        noEntriesRadial = np.where(rayDistances == -1)  # index of no hit
        noEntriesAngular = np.where(rayAngles == -1)

        # get boundary points
        thetaBndryPts = np.delete(rayAngles, noEntriesAngular)  # angle of hits
        if env_model == "maze" or "curved" or "eastEntry":
            rBndryPts = np.delete(rayDistances,
                                  noEntriesRadial) * parametersBC.scalingFactorK  # scaling factor to match training environment 16/rayLen from pyBullet_environment
        else:
            rBndryPts = np.delete(rayDistances, noEntriesRadial) * 16
        # the egocentric activity
        egocentricBCActivity = BCActivity.boundaryCellActivitySimulation(thetaBndryPts,
                                                                          rBndryPts)  # activity of num_ray*16 neurons

        afterBCStep = time.time()
        bcTimes.append(afterBCStep - beforeBCStep)
        ## for difference and plotting values
        eBCSummed += np.sum(egocentricBCActivity)
        # bvcSummed += np.sum(bvcActivity)
        # diff = np.sum(bvcActivity) - np.sum(egocentricBCActivity)
        # eachRateDiff.append(diff)
        # current position
        position = env.getPosition()
        xPositions.append(position[0])
        yPositions.append(position[1])
        eBCRates.append(np.sum(egocentricBCActivity))
        # bvcRates.append(np.sum(bvcActivity))
        if i % 20 == 0:
            sampleX.append(position[0])
            sampleY.append(position[1])
            sampleT.append(str(int(round(i))))

            # plotting
        if rtplot_bc:
            beforeStep = time.time()

            ##############     BC-Model    ##############
            bcPlotter.BCPlotting(egocentricBCActivity,decodedDir) # without acitivity in Allocentric

            # This plots the decoded heading, simulation speed will decrease when plotting
            # plotter.plot(rates_hdc, rates_sl, rates_sr, stimL, stimR, realDir, decodedDir)
            afterStep = time.time()
            plotTimes.append((afterStep - beforeStep))
            ################    end BC-Model    ##########

        # grid cell network track movement,update spiking of each gc module,gc.s
        beforeStep=time.time()
        # gc_network.track_movement(xy_speed)
        gc_network.track_movement(real_xy_speed)
        # gc_network.track_movement(calculated_xy_speed)

        # place cell network track gc firing
        goal_distance = np.linalg.norm(env.xy_coordinates[-1] - env.goal_location)
        reward = 1 if goal_distance < 0.1 else 0
        reward_first_found = False
        if reward == 1 and (len(cognitive_map.reward_cells) == 0 or np.max(cognitive_map.reward_cells) != 1):
            reward_first_found = True
            gc_network.set_current_as_target_state()  # gc.t=gc.s
        GCnetTimes.append(time.time()-beforeStep)
        beforeStep=time.time()
        [firing_values, created_new_pc] = pc_network.track_movement(gc_network.gc_modules, reward_first_found)
        PCnetTimes.append(time.time()-beforeStep)
        if created_new_pc:
            pc_network.place_cells[-1].env_coordinates = np.array(env.xy_coordinates[-1])
            # each pc_network.place_cells has env_coordinates[x,y] and gc_connection[6*1600]

        # cognitive map track pc firing
        # firing is pc firing list and is value is average of 6 module gc spiking
        # update,reward_cell,topology_cell,Recency_cell
        beforeStep=time.time()
        cognitive_map.track_movement(firing_values, created_new_pc, reward)
        CognitivemapnetTimes.append(time.time()-beforeStep)
        # plot or print intermediate update in console
        if not video and i % int(nr_steps / nr_plots) == 0:
            progress_str = "Progress: " + str(int(i * 100 / nr_steps)) + "%"
            print(progress_str)
            # plotCurrentAndTarget(gc_network.gc_modules)

    # simulated steps until next frame
    if video:
        # export current state as frame
        exploration_phase = True if frame < nr_steps_exploration else False
        plot_current_state(env, gc_network.gc_modules, f_gc, f_t, f_mon,
                           pc_network=pc_network, cognitive_map=cognitive_map,
                           exploration_phase=exploration_phase, goal_vector=goal_vector_array[-1])
        progress_str = "Progress: " + str(int((frame * 100) / nr_steps)) + "% | Current video is: " + str(
            frame * dt) + "s long"
        print(progress_str)


if video:
    # initialize video and call simulation function within
    frames = np.arange(0, nr_steps, step)
    # frames = np.arange(nr_steps)
    # t_before = time.time()
    anim = animation.FuncAnimation(Fig, func=animation_frame, frames=frames, interval=1 / fps, blit=False)

    # Finished simulation
    # Export video
    directory = "videos/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    f = "videos/animation.mp4"
    video_writer = animation.FFMpegWriter(fps=fps)
    anim.save(f, writer=video_writer)
    env.end_simulation()
    # plot the orientation

else:
    # manually call simulation function
    # t_before = time.time()
    animation_frame(nr_steps)

    # Finished simulation
    ################## hdc plot######################
    # final calculations
    t_total = time.time() - t_before
    print('the t_total is {},t_episode{},dt{}'.format(t_total,t_episode,dt))
    X = np.arange(0.0, t_episode, dt)
    print('length of X',len(X))
    cahv = [avs[i] - avs[i - 1] if i > 0 else avs[0] for i in range(len(avs))]
    cerr = [errs_signed[i] - errs_signed[i - 1] if i > 0 else errs_signed[0] for i in range(len(errs_signed))]
    corr, _ = pearsonr(cahv, cerr)
    # error noisy integration vs. noisy HDC
    if use_noisy_av:
        plt.plot(X, errs_noisy_signed, label="Noisy integration")
        plt.plot(X, errs_signed, label="Noisy HDC")
        plt.plot([0.0, t_episode], [0.0, 0.0], linestyle="dotted", color="k")
        plt.xlabel("time (s)")
        plt.ylabel("error (deg)")
        plt.legend()
        plt.show()

    # print results
    # print("cue_view_pos",cue_view_pos)
    print("\n\n\n")
    print("############### Begin Simulation results ###############")
    # performance tracking
    print("Total time (real): {:.2f} s, Total time (simulated): {:.2f} s, simulation speed: {:.2f}*RT".format(t_total,
                                                                                                              t_episode,
                                                                                                              t_episode / t_total))
    # print("Average step time network of HDC:  {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(hdcnetTimes),
    #                                                                        1.0 / np.mean(hdcnetTimes)))
    print("Average step time network of GC:  {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(GCnetTimes),
                                                                           1.0 / np.mean(GCnetTimes)))
    print("Average step time network of PC:  {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(PCnetTimes),
                                                                           1.0 / np.mean(PCnetTimes)))
    print("Average step time network of Cognitivemap:  {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(CognitivemapnetTimes),
                                                                           1.0 / np.mean(CognitivemapnetTimes)))
    print("Average step time network of BC:  {:.4f} ms; {} it/s possible".format(
        1000.0 * np.mean(bcTimes),
        1.0 / np.mean(bcTimes)))

    print("Average step time robot:    {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(robotTimes),
                                                                           1.0 / np.mean(robotTimes)))
    print("Average step time for searching goal:    {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(goalsearchTimes),
                                                                           1.0 / np.mean(goalsearchTimes)))
    if rtplot_bc:
        print("Average step time plotting for BC: {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(plotTimes),
                                                                               1.0 / np.mean(plotTimes)))
    if rtplot_hdc:
        print("Average step time plotting for HDC: {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(hdcplotTimes),
                                                                               1.0 / np.mean(hdcplotTimes)))
    time_coverage = 0.0
    print("Average time decoding:      {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(decodeTimes),
                                                                           1.0 / np.mean(decodeTimes)))
    # print("Steps done network of HDC:  {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X) * timesteps_neuron,
    #                                                                               len(X) * timesteps_neuron * np.mean(
    #                                                                                   hdcnetTimes), 100 * len(
    #         X) * timesteps_neuron * np.mean(hdcnetTimes) / t_total))
    # time_coverage += 100 * len(X) * timesteps_neuron * np.mean(hdcnetTimes) / t_total
    print("Steps done robot:    {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X), len(X) * np.mean(robotTimes),
                                                                                  100 * len(X) * np.mean(
                                                                                      robotTimes) / t_total))
    time_coverage += 100 * len(X) * np.mean(robotTimes) / t_total
    if rtplot_hdc:
        print("Steps done HDCplotting: {}; Time: {:.3f} s; {:.2f}% of total time".format(int(t_episode / plotfps),
                                                                                      int(t_episode / plotfps) * np.mean(
                                                                                          plotTimes), 100 * int(
                t_episode / plotfps) * np.mean(plotTimes) / t_total))
        time_coverage += 100 * int(t_episode / plotfps) * np.mean(plotTimes) / t_total
    print("Steps done decoding: {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X), len(X) * np.mean(decodeTimes),
                                                                                  100 * len(X) * np.mean(
                                                                                      decodeTimes) / t_total))
    print("Steps done searching goal: {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X), len(X) * np.mean(goalsearchTimes),
                                                                                  100 * len(X) * np.mean(
                                                                                      goalsearchTimes) / t_total))
    print("Steps done GC network: {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X), len(X) * np.mean(GCnetTimes),
                                                                                  100 * len(X) * np.mean(
                                                                                      GCnetTimes) / t_total))
    print("Steps done PC network: {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X), len(X) * np.mean(PCnetTimes),
                                                                                  100 * len(X) * np.mean(
                                                                                      PCnetTimes) / t_total))
    print("Steps done Cognitive map : {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X), len(X) * np.mean(CognitivemapnetTimes),
                                                                                  100 * len(X) * np.mean(
                                                                                      CognitivemapnetTimes) / t_total))
    if rtplot_bc:
        print("Steps done BC : {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X), len(X) * np.mean(bcTimes),
                                                                                  100 * len(X) * np.mean(
                                                                                      bcTimes) / t_total))
    time_coverage += 100 * len(X) * np.mean(decodeTimes) / t_total
    print("Time covered by the listed operations: {:.3f}%".format(time_coverage))

    print("maximum overall angular velocity: {:.4f} deg/s".format(max(avs) * r2d))
    print("average overall angular velocity: {:.4f} deg/s".format(sum([r2d * (x / len(avs)) for x in avs])))
    print("median overall angular velocity:  {:.4f} deg/s".format(np.median(avs)))

    print("maximum turning angular velocity: {:.4f} deg/s".format(max(avs_active) * r2d))
    print("average turning angular velocity: {:.4f} deg/s".format(sum([r2d * (x / len(avs_active)) for x in avs_active])))
    print("median turning angular velocity:  {:.4f} deg/s".format(np.median(avs_active)))

    print("maximum error: {:.4f} deg".format(max(errs)))
    print("average error: {:.4f} deg".format(np.mean(errs)))
    print("median error:  {:.4f} deg".format(np.median(errs)))
    print("################ End Simulation results ################")
    print("\n\n\n")

    # close real-time plot
    plt.close()
    plt.ioff()

    # plot error and angular velocity
    fig, ax1 = plt.subplots()
    # ax1.set_xlim(200, 375)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("error (deg)")
    # ax1.set_ylim(-13.5, 13.5)

    ax1.set_ylim(-15, 15)
    ax1.plot(X, errs_signed, color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.set_ylabel("angular velocity (deg/s)")
    ax2.set_ylim(-100, 100)
    ax2.plot(X, [x * r2d for x in avs], color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax1.plot([0.0, t_episode], [0.0, 0.0], linestyle="dotted", color="k")
    plt.show()

    # plot only error
    plt.xlabel("time (s)")
    plt.ylabel("error (deg)")
    # plt.ylim(-1.6, 1.6)
    # plt.ylim(-3.6, 3.6)
    plt.ylim(-15, 15)
    plt.xlim(0.0, t_episode)
    plt.plot(X, errs_signed)
    plt.plot([0.0, t_episode], [0.0, 0.0], linestyle="dotted", color="k")
    plt.show()

    # plot only angular velocity
    plt.xlabel("time (s)")
    plt.ylabel("angular velocity (deg/s)")
    # plt.ylim(-50, 50)
    plt.ylim(-100, 100)
    plt.xlim(0.0, t_episode)
    plt.plot(X, [x * r2d for x in avs])
    plt.plot([0.0, t_episode], [0.0, 0.0], linestyle="dotted", color="k")
    plt.show()

    # plot total rotation
    totalMovements = [0.0] * len(thetas)
    for i in range(1, len(avs)):
        totalMovements[i] = totalMovements[i - 1] + abs(r2d * thetas[i - 1])
    plt.plot(X, totalMovements)
    plt.xlabel("time (s)")
    plt.ylabel("total rotation (deg)")
    plt.show()

    # plot relative error # only used for PI evaluation
    # begin after 20%
    begin_relerror = int(0.2 * len(X))
    plt.plot(X[begin_relerror:len(X)], [100 * errs[i] / totalMovements[i] for i in range(begin_relerror, len(errs))])
    plt.xlabel("time (s)")
    plt.ylabel("relative error (%)")
    plt.show()

    # plot change in angular velocity vs. change in error # only used for PI evaluation
    plt.scatter(cahv, cerr)
    plt.plot([min(cahv), max(cahv)], [corr * min(cahv), corr * max(cahv)],
             label="linear approximation with slope {:.2f}".format(corr), color="tab:red")
    plt.legend()
    plt.xlabel("change in angular velocity (deg/s)")
    plt.ylabel("change in error (deg)")
    plt.show()


    # Plot last state of
    # cognitive_map_plot(pc_network, cognitive_map, environment=env_model)
    cognitive_map_plot(env,pc_network, cognitive_map)

    # Save place network and cognitive map to reload it later
    # if run_sim_from_data==True:# if navigation,then provide filename
    #     # provide filename="_navigation" to avoid overwriting the exploration phase
    #     filename="_navigation_"+env.door_option
    #     pc_network.save_pc_network(filename=filename)
    #     cognitive_map.save_cognitive_map(filename=filename)
    # else:
    # if not run_sim_from_data:
    pc_network.save_pc_network()  # provide filename="_navigation" to avoid overwriting the exploration phase
    cognitive_map.save_cognitive_map()  # provide filename="_navigation" to avoid overwriting the exploration phase
    # Calculate the distance between goal and actual end position (only relevant for navigation phase)
    error = np.linalg.norm((env.xy_coordinates[-1] + env.goal_vector) - env.goal_location)


    # if not run_sim_from_data:
    #     env.close()

    # Data to save to perform analysis later on
    error_array = [error]
    gc_array = [gc_network.consolidate_gc_spiking()]
    position_array = [env.xy_coordinates]
    vector_array = [goal_vector_array]

    progress_str = "Progress: " + str(int(1 * 100 / nr_trials)) + "% | Latest error: " + str(error)
    print(progress_str)
    # Directly plot and print the errors (distance between goal and actual end position)
    error_plot(error_array)
    print(error_array)

    # Save the data of all trials in a dedicated folder
    directory = "experiments" + vector_model + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    np.save("experiments" + vector_model + "/error_array", error_array)
    np.save("experiments" + vector_model + "/gc_array", gc_array)
    np.save("experiments" + vector_model + "/position_array", position_array)
    np.save("experiments" + vector_model + "/vectors_array", vector_array)
    '''
    Printsimulationtimeresult(t_total, t_episode, dt, GCnetTimes,
                                  PCnetTimes, CognitivemapnetTimes, bcTimes,
                                  robotTimes, goalsearchTimes, plotTimes,
                                  hdcplotTimes, decodeTimes, timesteps_neuron,
                                  plotfps, avs, r2d, errs, errs_signed,
                                  errs_noisy_signed, use_noisy_av, thetas, rtplot_bc, rtplot_hdc)
    '''
    env.end_simulation()  # disconnect pybullet