## Run 1D PVS model on line mesh along z axis.
## To be compared with 2D results from "The mechanisms behind perivascular fluid flow"
## (https://doi.org/10.1371/journal.pone.0244442)
import sys
import os
import shutil

### ------- Parametrization ------- ###

microm = 1e-3 # Convert [microm] to [mm]
meter = 1e3 # Convert [m] to [mm]

params = dict()

# ** Geometry
# mesh is in mm
params["Length"] = 1.0 # blood vessel length in millimeters
params["R1"] = 20 # arterial radius in micrometers
params["R2"] = 60 # pvs radius in micrometers

params["mesh_name"] = "pvs1D"
params["mesh_refinement"] = 1
params["reconstruct_sol"] = False # reconstruct 2D solution based on 1D solution (can be time-consuming)

# **Fluid parameters
params["rho"] = 1.0e-3  # Density of (CSF) water : 1000 [kg/m^3] -> 1e-3 [g/mm^3]
                        # Note : We use [g/mm^3] to make sur we obtain the pressure in [Pa]
params["nu"] = 0.697    # Kinematic viscosity

# ** Time
params["dt"] = 0.5
params["frequency"] = 1
params["nb_cycles"] = 1

# ** Driving forces
p_static_gradient = 0.1995
params["p_static_gradient"] = p_static_gradient
params["wall_movement"] = False
params["traveling_wave"] = False

# ** Additional parameters for 2D model
params["type_s"] = "axi"
params["origin"] = [0,0,0]
params["center"] = []
params["rigid_motion"] = False
# params["rigid_motion_dir"] =  [0,1]
# params["rigid_motion_amplitude"] = 6*microm

params["p_oscillation_phi"] = 0 #Bilston
params["c_vel"] = 1e-2
# params["rigid_motion_X0"] = params["Length"]*0.5
# params["periodic_BC"] = False

cwd = os.getcwd()
# Path to results directory
params["results_dir"] = cwd + "/../../../results/results_A1/"

model1D_path = cwd + '/../../2D_to_1D'
sys.path.insert(1, model1D_path)
print("Changing current dir to ", model1D_path)
os.chdir(model1D_path)

model2D_path = '../mechanisms-behind-pvs-flow/2Daxi'

reponame_1D = params["results_dir"] + "1D"
reponame_2D = params["results_dir"] + "2D"

### ------------------------------- ###


### --------- Build 1D mesh ---------- ###

from pvs1D_mesh import build_pvs_mesh

R_a, R_pv, Length = [params[key] for key in ["R1", "R2", "Length"]]
R_a *= microm # arterial radius in millimeters
R_pv *= microm # perivascular radius in millimeters

n = 32*Length*params["mesh_refinement"]

if not os.path.exists(params["results_dir"]):
    os.makedirs(params["results_dir"])

build_pvs_mesh(Length, R_a, R_pv, int(n), params["results_dir"] + params["mesh_name"] + ".xdmf")

### ------------------------------- ###



### ---------- Run 1D model ---------- ###

from pvs1D import *
pvs_model(params)

### ------------------------------- ###

# Copy script to results_dir
shutil.copy(cwd + "/" + __file__, params["results_dir"] + __file__)

### ------- Post processing ------- ###

# Comparison with 2D PVS model (Daversin-Catty & al.)
sys.path.insert(1, model2D_path)
import stokes_ale_artery_pvs as ref
from dolfin import *

# Run 2D model
inflow_area = ref.pvs_model(params)

# Read in the 2D mesh
mesh_2D = Mesh()
mesh_fname = "2D%s_L%.0f"%(params["type_s"], int(params["Length"]))
ref_meshfile = HDF5File(MPI.comm_world, reponame_2D + "/HDF5/mesh.h5", "r")
ref_meshfile.read(mesh_2D, "/mesh0", False)
ref_meshfile.close()

# Read solutions from 2D and 1D models
V = VectorFunctionSpace(mesh_2D, "CG", 2)
Q = FunctionSpace(mesh_2D, "CG", 1)

ref_u = Function(V) # Use 2D flux as reference
ref_p = Function(Q) # Same for pressure


# Read 1d mesh
mesh1D = Mesh()
with XDMFFile(MPI.comm_world, params["results_dir"] + "pvs1D.xdmf") as xdmf:
    xdmf.read(mesh1D)


q1D = Function(FunctionSpace(mesh1D, 'CG', 2))
p1D = Function(FunctionSpace(mesh1D, 'CG', 1))

# Load hdf5 files
ref_uhfile = HDF5File(MPI.comm_world, reponame_2D + "/HDF5/u.h5", "r")
ref_phfile = HDF5File(MPI.comm_world, reponame_2D + "/HDF5/p.h5", "r")
qhfile =     HDF5File(MPI.comm_world, reponame_1D + "/HDF5/q1D.h5", "r")
phfile =     HDF5File(MPI.comm_world, reponame_1D + "/HDF5/p1D.h5", "r")


attr = ref_uhfile.attributes("/function")
nsteps = attr['count']
for i in range(nsteps):
    dataset = "/function/vector_%d"%i
    attr = ref_uhfile.attributes(dataset)
    ref_uhfile.read(ref_u, dataset)
    ref_phfile.read(ref_p, dataset)
    qhfile.read(q1D, dataset)
    phfile.read(p1D, dataset)


# Average 2D flux and pressure

if params["reconstruct_sol"]: #then we read in our 2D reconstructed solutions as well
    u2D = Function(V) 
    p2D = Function(Q)
    uhfile = HDF5File(MPI.comm_world, reponame_1D + "/HDF5/u2D.h5", "r")
    phfile = HDF5File(MPI.comm_world, reponame_1D + "/HDF5/p2D.h5", "r")

    for i in range(nsteps):
        dataset = "/function/vector_%d"%i
        attr = ref_uhfile.attributes(dataset)
        uhfile.read(u2D, dataset)
        phfile.read(p2D, dataset)

    # Compute error
    ufile_err = XDMFFile(MPI.comm_world, reponame_1D + "/XDMF/error_u.xdmf")
    pfile_err = XDMFFile(MPI.comm_world, reponame_1D + "/XDMF/error_p.xdmf")

    error_u = Function(V)
    error_p = Function(Q)
    error_u.vector().set_local(u2D.vector() - ref_u.vector())
    error_p.vector().set_local(p2D.vector() - ref_p.vector())
    ufile_err.write(error_u)
    pfile_err.write(error_p)

    error_u_norm = errornorm(u2D, ref_u, norm_type='l2')
    error_p_norm = errornorm(p2D, ref_p, norm_type='l2')
    print("|error u|_L2 = ", error_u_norm)
    print("|error p|_L2 = ", error_p_norm)


## For comparison: Plot cross section fluxes obtained from 1D and 2D model

# Average 2D flux and pressure
x_coords_Q = q1D.function_space().tabulate_dof_coordinates()[:,0]
x_coords_P = p1D.function_space().tabulate_dof_coordinates()[:,0]

rs = np.linspace(R_a, R_pv, 500)
dr = rs[1]-rs[0]

# We find cross section flux q from 2D model via quadrature over cross section
A_pv = np.pi*(R_pv**2.0-R_a**2.0)
q_2D_vals = [(2.0*np.pi)*np.sum([ref_u(s, r)[0]*r*dr for r in rs]) for s in x_coords_Q]
p_2D_vals = [(2.0*np.pi)/(A_pv)*np.sum([ref_p(s, r)*r*dr for r in rs]) for s in x_coords_P]

q_2D = Function(q1D.function_space())
q_2D.vector()[:] = q_2D_vals

p_2D = Function(p1D.function_space())
p_2D.vector()[:] = p_2D_vals

# Compute error
error_q = errornorm(q1D, q_2D)
error_p = errornorm(p1D, p_2D)

# Now plot
plt.plot(x_coords_Q, q_2D_vals, '*', label='2D')
plt.plot(q1D.function_space().tabulate_dof_coordinates()[:,0],
        q1D.vector().get_local(), '*', label='1D')
plt.legend()
plt.xlabel('$s$')
plt.xlabel('$q$')
plt.title('Cross section flux')
plt.savefig('q_comparison.png')


# Now plot
plt.clf()
plt.plot(x_coords_P, p_2D_vals, '*', label='2D')
plt.plot(p1D.function_space().tabulate_dof_coordinates()[:,0],
        p1D.vector().get_local(), '*', label='1D')
plt.legend()
plt.xlabel('$s$')
plt.xlabel('$q$')
plt.title('Cross section pressure')
plt.savefig('p_comparison.png')

print('h: %1.3f, Error q: %1.1e, error p: %1.1e'%(mesh1D.hmin(), error_q, error_p))

### ------------------------------- ###
# Save params in the results_dir

import json
params.pop("deltaR", None) #we can't store the interp1d function in json
params_file = open(params["results_dir"] + 'params.json', 'w')
json.dump(params, params_file)
params_file.close()
