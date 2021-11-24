from stokes_ale_artery_pvs import *
from scipy.signal import find_peaks
import shutil

microm = 1e-3 # Convert [µm] to [mm]
meter = 1e3 # Convert [m] to [mm]

params = dict()
params["nb_cycles"] = 2

# C0075 fine
params["mesh_file"] = "../../C0092/C0092_clip1_mesh1_0.95_ratio_PVS.xdmf"
params["markers_file"] = "../../C0092/C0092_clip1_mesh1_0.95_ratio_PVS_mf.xdmf"

params["inlet_markers"] = [21]
params["outlet_markers"] = [22]

L_PVS = 44e-3 # [mm]
params["coord_factor"] = 2.0/L_PVS #C0092

params["frequency"] = 2.2
params["traveling_wave"] = True

# Rigid motion parameters -- Not used (we do not consider rigid motion here) 
params["rigid_motion"] = False
params["origin"] = [1,1,1]
params["center"] = [2,2,2]
x0,y0,z0 = params["origin"][0], params["origin"][1], params["origin"][2]
xc,yc,zc = params["center"][0], params["center"][1], params["center"][2]
# Direction
lp = [1,1,1]
rp = [2,2,2]
rigid_motion_dir = [lp[i] - rp[i] for i in range(len(lp))]
rigid_norm = sqrt(rigid_motion_dir[0]*rigid_motion_dir[0] \
                  + rigid_motion_dir[1]*rigid_motion_dir[1] \
                  + rigid_motion_dir[2]*rigid_motion_dir[2])
rigid_motion_dir = [rigid_motion_dir[i]/rigid_norm for i in range(len(rigid_motion_dir))]
params["rigid_motion_X0"] = sqrt( (xc-x0)**2 + (yc-y0)**2 + (zc-z0)**2)
params["rigid_motion_dir"] = rigid_motion_dir
params["rigid_motion_amplitude"] = 6*microm

# Pressure gradient
#p_static_gradient = 0
# (1) Infusion : We assume a cortex thickness of 2.5 mm and that the pressure drop can be 0.03 mmHg
# This gives p_grad = 0.03mmHg/2.5 mm = 0.03*133/2.5 Pa/mm = 1.60 Pa/mm
# p_static_gradient = 1.6
# (2) 3rd circulation : 0.01 mmHg/m = 0.00133 Pa/mm
# p_static_gradient = 0.00133
# (3) Respiration : 0.5 mmHg/m = 0.0665 Pa/mm
# p_static_gradient = 0.0665
# Movie
p_static_gradient = 0.1995
params["p_static_gradient"] = p_static_gradient
params["p_oscillation_L"] = [0.9, 1]

# Various wave speed
c_vels = [1e3]

# Bilston
phis = [0] # No bilston

# Time steps
dts = [0.02]

for i_c, c_vel in enumerate(c_vels):
    params["c_vel"] = c_vel
    for dt in dts:
        params["dt"] = dt
        for phi in phis:
            params["p_oscillation_phi"] = phi
    
            parameters = "c%.2f_dt%.3f_dp%.2f"%(c_vel, dt, p_static_gradient)
            if phi:
                parameters += "_Bilston_phi_%.2f"%(phi)

            fig, figs = plt.subplots(1, 3)
            fig.set_size_inches(18.5, 10.5)
            fig.suptitle(parameters)

            figs[0].set_title('u avg [mm/s]')
            figs[1].set_title('inflow [mm^3/s]')
            figs[2].set_title('Position [mm]')

            reponame, file_extension = os.path.splitext(params["mesh_file"])
            reponame = reponame.split('/')[-2] + "_results"

            inflow_area = pvs_model(params)

            ##  -- Plot u_avg
            t,uavg = plt.loadtxt(reponame + '/uavg.txt').transpose()
            ## Save plot to png
            figs[0].plot(t, uavg, label="uavg")
            figs[0].legend()
            figs[0].set_xlabel("Time [s]")

            ## -- Plot inflow
            t,v = plt.loadtxt(reponame + '/inflow.txt').transpose()
            header_v_avg = "T \t Vavg"
            data_v_avg = plt.array([t,v/inflow_area]).transpose()
            plt.savetxt(reponame + '/vavg.dat', data_v_avg, delimiter="\t", header=header_v_avg)
            ## Save plot to png
            figs[1].plot(t,v, label="vavg")
            figs[1].legend()
            figs[1].set_xlabel("Time [s]")

            ## -- Plot Q
            Q = plt.array([plt.trapz(v[:i],t[:i]) for i in range(1,len(t))])
            position = Q/inflow_area
            ## Save data to file with suitable formating
            header_Q = "T \t Q"
            header_position = "T \t Position"
            data_Q = plt.array([t[1:],Q]).transpose()
            data_position = plt.array([t[1:],position]).transpose()
            plt.savetxt(reponame + '/Q.dat', data_Q, delimiter="\t", header=header_Q)
            plt.savetxt(reponame + '/position.dat', data_position, delimiter="\t", header=header_position)

            ## Save plot to png
            figs[2].plot(t[1:], position, label="position")
            figs[2].legend()
            figs[2].set_xlabel("Time [s]")

            ## Export figure with the three subplots
            plt.savefig(reponame + "/" + parameters + ".png")

            ### --- Post processing files ---- ###
            ppfile_length_name = reponame + '/post_processing.dat'
            ppfile_length = open(ppfile_length_name, 'a+')
            if i_c is 0 and MPI.rank(MPI.comm_world) == 0:
                ppfile_length.truncate(0)
                ppfile_length.write('cvel \t avgvelocity \n')

            # Note : find_peaks doesn't work when peaks are not acute enough
            # If so, compute slope finding two peaks manually
            try:
                idx = find_peaks(position)[0]
                v0 = position[idx[-2]]
                v1 = position[idx[-1]]
                t0 = t[idx[-2]]
                t1 = t[idx[-1]]
                v_avg = (v1-v0)/(t1-t0)
                #print("v_avg (find_peaks) = ", v_avg)
            except:
                shift = len(position)/params["nb_cycles"]
                idx1 = np.argmax(position)
                idx0 = np.argmax(position[:idx1 - int(shift)])
                v0 = position[idx0]
                v1 = position[idx1]
                t0 = t[idx0]
                t1 = t[idx1]
                v_avg = (v1-v0)/(t1-t0)
                #print("v_avg (argmax) = ", v_avg)

            if MPI.rank(MPI.comm_world) == 0:
                ppfile_length.write('%g\t%g\n'%(c_vel, 1e3*v_avg))

            ppfile_length.close()

            if len(c_vels) > 1 or len(dts) > 1 or len(phis) > 1:
                current_reponame = os.getcwd() + "/" + reponame
                new_reponame = current_reponame + "_c_%.2f"%(c_vel) + "_mesh%i"%(refinement) + "_dt_%.4f"%(dt) + "_phi_%.2f"%(phi)
                print("Moving ", current_reponame, " to ", new_reponame)
                shutil.move(current_reponame, new_reponame)
                os.mkdir(current_reponame)

