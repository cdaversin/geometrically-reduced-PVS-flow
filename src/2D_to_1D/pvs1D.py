import numpy as np
from dolfin import *
import pvs1D_utils as utils
import pylab as plt
import numpy as np
import scipy.interpolate as sp_interpolate
import resource


microm = 1e-3 # Convert [µm] to [mm]
meter = 1e3 # Convert [m] to [mm]

def default_params():
    params = dict()
    params["Length"] = 1
    params["R1"] = 20
    params["R2"] = 60
    params["mesh_name"] = ""
    params["frequency"] = 10
    params["nb_cycles"] = 1
    params["p_static_gradient"] = 0
    params["dt"] = 0.001
    params["wall_movement"] = False
    params["rho"] = 1.0
    params["nu"] = 0.697
    params["results_dir"] = ""

    return params

def pvs_model(params = default_params()):

    # Collect params from dictionnary
    Length = params["Length"]
    R1 = params["R1"]*microm # arterial radius in mm
    R2 = params["R2"]*microm # pvs radius in mm
    frequency = params["frequency"]
    nb_cycles = params["nb_cycles"]
    p_static_gradient = params["p_static_gradient"]
    dt = params["dt"]
    use_wall_mvt = params["wall_movement"]

    nu = Constant(params["nu"])
    rho = Constant(params["rho"])
    mu = nu*rho #dynamic visosity

    results_dir = params["results_dir"] + "1D"
    
    # Cardiac cycle in mice :  550 to 725 bpm -> 80–110 ms
    # (source : "Cardiac electrophysiology in mice: a matter of size",
    # https://www.frontiersin.org/articles/10.3389/fphys.2012.00345/full)
    # cycle_duration = 0.1 # Duration of a cardiac cycle (in seconds)
    cycle_duration = 1/frequency # [s]

    ### -------- Create PVS mesh ---------------------------- ###
    # For now : 1D mesh along z direction (use pvs_mesh.py to generate the mesh)
    # Next step : Realistic geometry from PVS-meshiong-tools
    mesh_file = params["results_dir"] + "pvs1D.xdmf" # Need to be parametrized to make sure we have the right dimensions

    # 2D mesh whose centerline is our 1D_mesh
    m_ = 3*params["mesh_refinement"] # Mesh resoution - PVS width
    n_ = 20*params["mesh_refinement"]           
    print("Mesh size = ", Length, " x ", R2-R1, " [mm]")
    print("Mesh resolution = ", n_, " x ", m_)
    
    mesh_2D = RectangleMesh(Point(0.0, R1), Point(Length, R2), n_, m_) # along x

    ### ----------------------------------------------------- ###

    ### -------- Read PVS mesh ------------------------------ ###
    mesh = Mesh()
    with XDMFFile(MPI.comm_world, mesh_file) as xdmf:
        xdmf.read(mesh)
    
    ### ----------------------------------------------------- ###

    ### -------- Markers and Measures ----------------------- ###
    inlet_tag = 1
    outlet_tag = 2
    mf = MeshFunction('size_t',mesh, mesh.topology().dim()-1)
    mf[0] = inlet_tag
    mf[mesh.num_vertices()-1] = outlet_tag
    ds = Measure("ds", domain = mesh, subdomain_data=mf)

    # Export markers (just checking)
    mfile = XDMFFile(MPI.comm_world, params["results_dir"] + "XDMF/markers.xdmf")
    mfile.write(mf)
    mfile.close()
    ### ----------------------------------------------------- ###

    
    
    ### -------- Mesh displacement ------------------------ ###
    
    # The change in R_a over time is provided by the runscript    
    if "deltaR" in params:
        fdata = params["deltaR"] 
    else: 
        # if no deltaR is provided we assume zero change in R_a
        def foo(t): return 0.0
        fdata = foo
    
    
    radius_a, radius_pv = Constant(R1), Constant(R2) # cast these to fenics 

    # To begin with we evaluate quantities at t=0
    time = 0.0

    def RelDeltaD(_t):
        # Percentage change in diameter at given time _t
        # Data is given for one cardiac cycle , x \in [0,1]
        val = fdata((_t/cycle_duration)%1)
        # The data are given in percents
        val = val*1e-2
        return val

    R_pv = R2 # R_pv is constant

    traveling_wave = params["traveling_wave"]
    c_vel = params["c_vel"]
    frequency = params["frequency"]
    x0,y0,z0 = params["origin"][0], params["origin"][1], params["origin"][2]
    L_PVS = R2 - R1

    # Radius R_a is a fenics expression depending on time
    # By updating R_a.time we then have an expression giving the R_a(x) for that time
    class R_a(UserExpression):
        def __init__(self, time, **kwargs):
            self.time = time
            super().__init__(**kwargs)

        def eval(self, values, x):
            X = sqrt( (x[0] - x0)**2 + (x[1] - y0)**2 + (x[2] - z0)**2 )
            t = self.time - (X/c_vel)*(traveling_wave)
            
            delta_R = 0.5*RelDeltaD(t)*L_PVS*use_wall_mvt 
            values[0] = radius_a(x) + delta_R
            
        def value_shape(self):
            return ()

    # R_pv constant in time (but define as a function when moving with tissue)
    def R_pv(time):
        return radius_pv

    el = FiniteElement('CG', mesh.ufl_cell(), 1)

    # Cross-section area
    A_pv = utils.A_PV(R_a, R_pv, time, element=el, domain=mesh)
    
    # Cross-section diameter
    D_a = utils.D(R_a, time)
    
    # Arterial wall velocity
    w = utils.W(R_a, time, delta_t=0.01*dt, element=el, domain=mesh)
    
    # Coefficient alpha (as used in the 1D variational formulation)
    alpha = utils.Alpha(R_a, R_pv, time, degree=2)
    
    ### ----------------------------------------------------- ###

    
    ### ----------------------------------------------------- ###

    ### -------- Stokes formulation ----------------------- ###
    Q = FiniteElement("CG", mesh.ufl_cell(), 2)
    P = FiniteElement("CG", mesh.ufl_cell(), 1)
    QP = FunctionSpace(mesh, MixedElement(Q, P))

    # Velocity-pressure at time n on Omega_t_n
    qp_ = Function(QP)
    (q_, p_) = split(qp_)

    # Velocity-pressure test functions
    (v, phi) = TestFunctions(QP)
    # Velocity-pressure trial functions
    (q, p) = TrialFunctions(QP)

    def dds(f):
        T = Expression(("1", "0", "0"), degree=2) # centerline along x axis
        # T = Expression(("1/sqrt(2)", "1/sqrt(2)", "0"), degree=2) # TEST - centerline x=y
        T = project(T, VectorFunctionSpace(mesh, "CG", 1))
        return inner(grad(f), T)

    # Variational formulation
    a = ((rho/A_pv)*q*v*dx
         + dt*(mu/A_pv)*dds(q)*dds(v)*dx
         - dt*p*dds(v)*dx
         + dds(q)*phi*dx
         + dt*(alpha/A_pv)*mu*q*v*dx)

    # Impose p=p_static_gradient at ds(1) = inlet / p=0 at ds(2) = outlet
    p_inlet = Constant(Length*p_static_gradient)
    L = D_a*w*phi*dx + (rho/A_pv)*q_*v*dx + dt*A_pv*p_inlet*v*ds(1)

    # Time
    time = 0.0

    qp0 = Function(QP) 

    # Export
    qfile_1D = XDMFFile(MPI.comm_world, results_dir + "/XDMF/q1D.xdmf")
    pfile_1D = XDMFFile(MPI.comm_world, results_dir + "/XDMF/p1D.xdmf")
    ufile_2D = XDMFFile(MPI.comm_world, results_dir + "/XDMF/u2D.xdmf")
    pfile_2D = XDMFFile(MPI.comm_world, results_dir + "/XDMF/p2D.xdmf")
    uhfile_2D = HDF5File(MPI.comm_world, results_dir + "/HDF5/u2D.h5", "w")
    qhfile_1D = HDF5File(MPI.comm_world, results_dir + "/HDF5/q1D.h5", "w")
    whfile_1D = HDF5File(MPI.comm_world, results_dir + "/HDF5/w1D.h5", "w")
    phfile_1D = HDF5File(MPI.comm_world, results_dir + "/HDF5/p1D.h5", "w")
    phfile_2D = HDF5File(MPI.comm_world, results_dir + "/HDF5/p2D.h5", "w")


    import time as pytime
    tic = pytime.time()
    mem_usage = []
    
    for i in range(nb_cycles):
        cycle = i + 1
        if MPI.rank(MPI.comm_world) == 0:
            print("-- Start cycle ", cycle, " -- [", time, ", ", cycle*cycle_duration, "]")
        while(time <= cycle*cycle_duration):
            if MPI.rank(MPI.comm_world) == 0:
                print("Solving for t = %g" % time)
                
                mem_usage.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

                # Update coefficients that depend on R_pv
                R_a_val = R_a(time)
                D_a.R_a = R_a_val
                A_pv.time = time
                w.time = time      

                A = assemble(a)
                b = assemble(L)

                # Solve system
                solve(A, qp0.vector(), b)
                (q0, p0) = qp0.split(deepcopy=True)

                # Update qp_
                qp_.assign(qp0)

                # Export results
                qfile_1D.write(q0,time)
                pfile_1D.write(p0,time)
                qhfile_1D.write(q0, "/function", time)
                phfile_1D.write(p0, "/function", time)
                whfile_1D.write(interpolate(w, p0.function_space()), "/function", time)
                
                # Reconstruct 2D solution
                if params["reconstruct_sol"]:
                    u_2D = utils.reconstruct_2d_flux(mesh_2D, q0, R_a, R_pv, time)
                    p_2D = utils.reconstruct_2d_pressure(mesh_2D, p0, R_a, R_pv, time)
                    
                    u_2D.rename("u", "velocity")
                    p_2D.rename("p", "pressure")

                    uhfile_2D.write(u_2D, "/function", time)
                    phfile_2D.write(p_2D, "/function", time)

                # Update time
                time = time + dt

    print('** Memory and time usage (1D) **')
    toc = pytime.time()
    
    max_mem_usage = np.max(np.asarray(mem_usage))/1000
    print('Memory usage: %f (mb)' % (max_mem_usage))

    num_dofs = qp0.function_space().dim()
    print('Num of dofs: %g'%num_dofs)
    
    time_per_loop = (toc-tic)/(time/dt)
    print('Time per loop: %g'%time_per_loop)


    # Close files
    files = [qfile_1D, qhfile_1D, phfile_1D, pfile_1D, 
             ufile_2D, pfile_2D, uhfile_2D, phfile_2D]
    for file in files: file.close()


    ### ----------------------------------------------------- ###


### -------------------- MAIN ------------------------- ###
if __name__ == "__main__":
    pvs_model()
### --------------------------------------------------- ###

