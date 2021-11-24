## Run 1D PVS model on C0092 centerline
## To be compared with 3D results from "The mechanisms behind perivascular fluid flow"
## (https://doi.org/10.1371/journal.pone.0244442)

### Model B1 : Pressure gradient + No wall mvt ###

import sys
import os
import shutil

import numpy as np
from dolfin import *

### ------- Parametrization ------- ###

microm = 1e-3 # Convert [µm] to [mm]
meter = 1e3 # Convert [m] to [mm]

params = dict()

params["refine"] = False

params["c_vel"] = 0.8 #[mm/s]
params["nb_cycles"] = 3
params["dt"] = 0.5
L_PVS = 44e-3 # [mm]
params["coord_factor"] = 2.0/L_PVS #C0092

# Pressure gradient
p_static_gradient = 0.1995
params["p_static_gradient"] = 0.0

# Wall mvt
def f(t):
    return 7.5*np.sin(2.0*np.pi*t)

params["deltaR"] = f
#f = 0.1Hz
#lambda = 8 mm (Aldea)
#c = lambda*f

params["frequency"] = 0.1
params["wall_movement"] = True
params["traveling_wave"] = True

params["origin"] = [0.6895,0.056,1.0715]


# Path to centerline mesh and data
cwd = os.getcwd()
params["case_dir"] = cwd + "/../../C0092/"
params["case_prefix"] = "C0092_clip1_mesh1_0.95_ratio"
# Path to results directory
params["results_dir"] = cwd + "/../../../results/results_B3/"

print("Changing current dir to ", params["case_dir"])
os.chdir(params["case_dir"])

# Solve 3D model ?
solve_3D = True
### ------------------------------- ###

### --------- Run 1D model -------- ###

model1D_path = '../3D_to_1D'
sys.path.insert(1, model1D_path)
from pvs1D import *
pvs_model(params)

### ------------------------------- ###

### --------- Run 3D model -------- ###

model3D_path = '../mechanisms-behind-pvs-flow/3D' 

# Additional parameters needed for 3D

params["inlet_markers"] = [21]
params["outlet_markers"] = [22]

# Rigid motion
params["rigid_motion"] = False

# Data obtained from centerline should be used
params["p_oscillation_L"] = [1.96]

# Various wave speed
# Bilston
#phis = [0] # No bilston
params["p_oscillation_phi"] = 0

reponame_1D = params["results_dir"] + "1D"
reponame_3D = params["results_dir"] + "3D"

sys.path.insert(1, model3D_path)
import stokes_ale_artery_pvs as ref

if solve_3D:
    inflow_area = ref.pvs_model(params)
else:
    inflow_area = 1
### ------------------------------- ###

# Copy script to results_dir
shutil.copy(cwd + "/" + __file__, params["results_dir"] + __file__)

# Save params in the results_dir
import json
params.pop("deltaR") #we can't store the interp1d function in json
params_file = open(params["results_dir"] + 'params.json', 'w')
json.dump(params, params_file)
params_file.close()


### ------- Post processing ------- ###

from post_process import *
generate_inflow_data(reponame_3D, reponame_1D)
generate_wall_velocity_data(reponame_3D, reponame_1D)

compute_1D_avg_from_3D_solution(params["case_dir"], params["case_prefix"], params["results_dir"], params["coord_factor"], params["dt"])
