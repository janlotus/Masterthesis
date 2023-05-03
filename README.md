# MA_bio_inspired_navigation_based on HDC,GC,PC and BC

Welcome to the Git Repo for the Master Thesis on

> Biologically Inspired Spatial Navigation Based on Head Direction Cells, Grid Cells, Place Cells and Border Cells

In this file we will cover the installation process to get the code running and touch upon the meaning of the different files.
For a detailed description of the functionality and thoughts behind this code, please refer to the thesis.

## Install packages
The code is based on Python3. You will need to install the following packages to run the code:

- pip
- pybullet
- matplotlib
- numpy
- scipy
- pycuda # if use Nividia GPU

Also make sure to have latex installed to be able to export plots.

## Some more setup
For exploration, make sure to set in main.py the variable **run_from_data = False,nr_steps=15000,nr_steps_exploration=nr_steps** to generate the cognitive map.
For navigation, make sure to set in main.py the variable **run_from_data = True,nr_steps=8000,nr_steps_exploration=0**.

When running experiments, double check where the data will be saved and avoid that it overwrites your previous runs.

## What to do with the code?
The code was designed to perform and analyze the experiments described in the thesis.
With the **main.py** the main experiments can be executed.

### set the parameters
- As environment choose **env_model = "single_line_traversal"**. 
- Pick a decoder of your choice via the variable **vector_model**.
- Set **nr_steps = 8000** and **nr_steps_exploration = 15000**.
- Double check what data you want to save of each trial.
- if use Video, then comment the two functions plot_sub_goal_localization in linearLookahead.py
- Nivida GPU RTX3060 and Pycuda is used for calculation.
- If no GPU,then change the system/Bio_model/HeadDirectionCellModel/network.py, use numpy, but maybe not work
- If change the parameters in system/Bio_model/HeadDirectionCellModel/params.py,generate weights using hdcCalibrConnectivity.py
- If change the parameters in system/Bio_model/BoundaryCellModel/parametersBC.py, such as number of sensors, generate weights using MakeWeights.py

### Maze test
Here it makes sense to decouple exploration and navigation phase.
- Set the number of trials you want to execute to 1.
- First do a run with **nr_steps = 15000** and **nr_steps_exploration = nr_steps**.
- Save the cognitive map at the end of the run (or use the data provided with lrz folder)
- Adapt the environment if wanted and do a run with **nr_steps = 8000** and **nr_steps_exploration = 0**.


You also have the option to create videos with the **main.py** file. This is why the loop is structured in this specific way.

## Code Structure
Note that some folders are git-ignored but will be created when running the script.

    ├── data                   		# Optional initialization data for gc, pc, cm
    ├── environment_map         		# Environment files for pybullet scene. Just use linear_sunburst_map
    ├── experiments             		# Here it will save plots, when you analyze run data 
    ├── experimentslinear_lookahead		# Result for errors
    ├── p3dx                    		# Agent files for pybullet
    ├── plots                   		# Here it will save plots during linear lookaheads if you want
    ├── plotting                		# Script to create plots
    │   ├── plotThesis.py       		# TUM themed and labeled plots used for the thesis
    │   ├── plotResults.py      		# WIP plots to check results and scripts to create video
    │   ├── plotHelper.py      		# Some helper functions used by plotResults.py
    │   └── tum_logo.png        		# TUM Logo    
    ├── system                  		# Scripts that make up the system
    │   ├── bio_model           		# Scripts modeling HDC,BC,GC,PC and cognitive map cells.
    │       ├──BoundaryCellModel        # Scripts for BC model
    │           ├──weights              # Files to stores the weights between neurons in BC model
    │           ├──BoundarycellModel.py # BC Activity calculation, allocentric information not used here
    │           ├──parametersBC.py      # Parameters,such as number of sensors
    │           ├──Makeweights.py       # Generate weights, if ParameterBC changes, run this again to update weights.
    │           ├──.......              # Other scripts.
    │       ├──CognitiveMapModel
    │       ├──GridcellModel
    │       ├──HeadDirectionCellModel
    │           ├──network.py           # Switch between numpy and Pycuda(Nivida GPU),if no GPU, change this script.
    │           ├──params.py            # Parameters for HDC model
    │           ├──hdcCalibrConnectivity.py# If parameter for HDC changes, run this to update the weights.
    │       ├──PlaceCellModel
    │   │── controller          		# Navigation phase, Exploration Phase controller and Agent environment
    │   ├── decoder             		# Scripts for different grid cell decoder mechanism,just use linearLookahead
    │   └── helper.py           	        # Some helper function used across scripts
    ├── videos                  		# Here it will save the video
    ├── main.py                 		# Execute this file for main experiments
    ├── README.md               		# You are here. Overview of project
    └── TUM_MA_...pdf           		# Master thesis explaining thoughts and theory behind code

## Questions left?
Any questions left or having troubles with the code? Feel free to reach out to hezhanlun@gmail.com
