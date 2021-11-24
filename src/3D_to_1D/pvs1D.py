import os
import numpy as np
from dolfin import *
import pvs1D_utils as utils
import mesh_utils as mesh_utils
import numpy as np
import resource


microm = 1e-3 # Convert [µm] to [mm]
meter = 1e3 # Convert [m] to [mm]

def default_params():
    params = dict()
    params["c_vel"] = 1e3 #[mm/s]
    params["frequency"] = 10
    params["nb_cycles"] = 1
    params["p_static_gradient"] = 0
    params["dt"] = 0.001
    params["wall_movement"] = True
    params["traveling_wave"] = False
    params["coord_factor"] = 1
    params["rho"] = 1.0
    params["nu"] = 0.697
    params["origin"] = [0,0,0]
    params["case_prefix"] = ""
    params["results_dir"] = ""

    return params

def pvs_model(params = default_params()):

    # Collect params from dictionnary
    c_vel = params["c_vel"]
    frequency = params["frequency"]
    nb_cycles = params["nb_cycles"]
    p_static_gradient = params["p_static_gradient"]
    dt = params["dt"]
    use_wall_mvt = params["wall_movement"]
    traveling_wave = params["traveling_wave"]
    coord_factor  = params["coord_factor"]
    x0,y0,z0 = params["origin"][0], params["origin"][1], params["origin"][2]

    prefix = params["case_prefix"]
    results_dir = params["results_dir"] + "1D"
    
    # Density of (CSF) water : 1000 [kg/m^3] -> 1e-3 [g/mm^3]
    # Note : We use [g/mm^3] to make sur we obtain the pressure in [Pa]
    nu = Constant(0.697) # Mesh is in mm : 0.697e-6 [m^2/s] -> 0.697 [mm^2/s]
    # Density of (CSF) water : 1000 [kg/m^3] -> 1e-3 [g/mm^3]
    # Note : We use [g/mm^3] to make sure we obtain the pressure in [Pa]
    rho = Constant(1e-3)

    mu=nu*rho

    cycle_duration = 1/frequency # [s]

    ### ------------ Read mesh ------------------------------ ###
    mesh_file = prefix + "_centerline_mesh.xdmf"
    mesh = Mesh()
    with XDMFFile(MPI.comm_world, mesh_file) as xdmf:
        xdmf.read(mesh)

    mesh.coordinates()[:]/=coord_factor

    print("[1D] nb vertices = ", mesh.num_entities(0))
    print("[1D] mesh hmin = ", mesh.hmin())
    print("[1D] mesh hmax = ", mesh.hmax())

    ### ----------------------------------------------------- ###

    ### -------- Markers and Measures ----------------------- ###

    # Read markers file
    markers_file = prefix + "_centerline_markers.xdmf"
    mf = MeshFunction('size_t',mesh, mesh.topology().dim()-1)
    with XDMFFile(MPI.comm_world, markers_file) as xdmf:
        xdmf.read(mf)

    # Branches length
    if os.path.exists(params["case_prefix"] + "_centerline_branches_length.txt"):
        print("Branches length loaded from : ", params["case_prefix"] + "_centerline_branches_length.txt")
        with open(params["case_prefix"] + "_centerline_branches_length.txt", "r") as length_file:
            branches_length = np.loadtxt(length_file, dtype='float')
            branches_length = np.atleast_1d(branches_length)
            branches_length = branches_length/coord_factor
    else:
        print("Length file does not exist!")
        branches_length = params["p_oscillation_L"]
    
    inlet_markers = [10] # Considering only one inlet #FIXME
    outlet_markers = [20 + i for i in range(len(branches_length))]

    # List of mesh segments (parent, daughters, ...)
    mesh_segments = []
    bifurcation_p = []
    
    
    # Trying to find if there is any bifurcation
    try:
        bifurcation_mf = mesh_utils.bifurcation_point(mesh)
        # DEBUG - Check marking
        with XDMFFile(MPI.comm_world, prefix + "_boundaries_markers.xdmf") as xdmf:
            xdmf.write(mf)
        with XDMFFile(MPI.comm_world, prefix + "_bifurcation_markers.xdmf") as xdmf:
            xdmf.write(bifurcation_mf)

        if len(bifurcation_mf.where_equal(1)) > 1:
            raise NotImplementedError("Cases with more than one bifurcation are not supported (yet)")

        # Define Point corresponding to inlet from inlet marking (MeshFunction)
        bifurcation_vertex = Vertex(mesh, bifurcation_mf.where_equal(1)[0])
        bifurcation_p = [bifurcation_vertex.x(0), bifurcation_vertex.x(1), bifurcation_vertex.x(2)]
        
        # Marking branches from bifurcation point :
        # The line segments are marked with the tag or the corresponding inlet(10, 11, ...)/outlet(20, 21, ...)
        # In case of multiple bifurcations, the line segments between bifurcation points are tagged 30, 31, ...
        mf_branches = mesh_utils.branches_marker(mesh, mf, inlet_markers, outlet_markers)
        # Check marking
        with XDMFFile(MPI.comm_world, prefix + "_branches_marker.xdmf") as xdmf:
            xdmf.write(mf_branches)

        # Build segment meshes (MeshView)
        for im in inlet_markers:
            mesh_p = MeshView.create(mf_branches, im)
            mesh_segments.append(mesh_p)
        for om in outlet_markers:
            mesh_d = MeshView.create(mf_branches, om)
            mesh_segments.append(mesh_d)

        # Marking boundary of segment meshes
        subdomains = [mesh_utils.segment_mesh_mf(msh, mf, inlet_markers, outlet_markers) for msh in mesh_segments]
        # Check marking
        for i, subd in enumerate(subdomains):
            with XDMFFile(MPI.comm_world, prefix + "_marked_branch" + str(i) + ".xdmf") as xdmf:
                xdmf.write(subd)

    except ValueError:
        print("No bifurcation point found")
        # Only one mesh segment (whole mesh)
        mesh_segments.append(mesh)
        subdomains = [mf] # Marker on the parent mesh

    # Inlets : 10, 11, ...
    # Outlets : 20, 21, ...
    dx_ = [Measure("dx", domain = msh) for msh in mesh_segments]
    ds_ = [Measure('ds', domain=mesh_segments[i], subdomain_data=subd) for i, subd in enumerate(subdomains)]
    
    for i,s in enumerate(mesh_segments):
        print("Length segment [", i , "] = ", assemble(Constant(1)*dx_[i]))

    ### ----------------------------------------------------- ###

    ### ------- Read centerline data ---------------------- ###
    
    # Read centerline data files
    V = FunctionSpace(mesh, "CG", 1)
    radius_a = Function(V)
    radius_pv = Function(V)

    radius_hfile  = HDF5File(MPI.comm_world, prefix + "_HDF5/centerline_radius.h5", "r")
    dataset = "/function/vector_%d"%0
    radius_hfile.read(radius_a, dataset)
    
    radius_a.vector()[:]/=coord_factor
    
    radius_a.vector()[:] *= 0.5 ## TODO: radius_a is twice its size when we compare with the actual 3D domain??

    # pvs width is proportional to r_a (L_PVS \approx artery = 2*radius_a)
    # When building 3D mesh, pvs thickness is 0.95*Ra -> R_pv = 2.95*R_a
    radius_pv = project(2.95*radius_a, V)

    # Get tangent components from mesh vertices
    tang_comps = mesh_utils.get_tangent_vector(mesh)

    
    ### ----------------------------------------------------- ###

    ### -------- Mesh displacement ------------------------ ###

    # The change in R_a over time is provided by the runscript    
    if "deltaR" in params:
        fdata = params["deltaR"] 
    else: 
        # if no deltaR is provided we assume zero change in R_a
        def foo(t): return 0.0
        fdata = foo
    
    # Starts at time = dt (as in 3D model)
    time = dt
    
    def RelDeltaD(_t):
        
        # Data is given for one cardiac cycle , x \in [0,1]
        # Compute value at given time _t
        val = fdata((_t/cycle_duration)%1)
        # The data are given in percents
        val = val*1e-2

        return val
    ### ----------------------------------------------------- ###

    ### --------- Constant and coeffs ----------------------- ###

    L_PVS = 44e-3 #mm
    
    
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
    A_pvs = [utils.A_PV(R_a, R_pv, time, element=el, domain=msh) for msh in mesh_segments]
    
    # Cross-section diameter
    D_a = utils.D(R_a, time)
    
    # Arterial wall velocity
    w_s = [utils.W(R_a, time, delta_t=0.01*dt, element=el, domain=msh) for msh in mesh_segments]
    
    # Coefficient alpha (as used in the 1D variational formulation)
    alpha = utils.Alpha(R_a, R_pv, time, degree=2)
    
    
    ### ----------------------------------------------------- ###

    ### -------- Stokes formulation ----------------------- ###    
    QP = [FunctionSpace(msh, 'CG', 2) for msh in mesh_segments] # Flux spaces in each segment
    LM = FunctionSpace(mesh, 'R', 0) # Lagrange multiplier (to impose bifurcation conditions)
    X = FunctionSpace(mesh, 'CG', 1) # Pressure space (on whole mesh)
    if bifurcation_p:
        QP.append(LM)
    QP.append(X)
    QP = MixedFunctionSpace(*QP) # QP = Q*QD1*QD2*...*P
    # Velocity-pressure at time n on Omega_t_n
    qp_ = Function(QP)

    # Flux-pressure test functions
    # vp, vd1, vd2 = vphi[0], vphi[1], ....and phi = vphi[-1] (test function Lagrange multiplier if bifurcation : vphi[-2])
    vphi = TestFunctions(QP)
    # Flux-pressure trial functions
    # qp, qd1, qd2 = qp[0], qp[1], ....and phi = qp[-1] (trial function Lagrange multiplier if bifurcation : qp[-2])
    qp = TrialFunctions(QP)

    def dds(f):
        return tang_comps[0]*f.dx(0) + tang_comps[1]*f.dx(1) + tang_comps[2]*f.dx(2) # inner(grad(f), tangent)


    # Variational formulation
    a = ((rho/A_pv)*qp[0]*vphi[0]*dx_[0]
         + dt*(mu/A_pv)*dds(qp[0])*dds(vphi[0])*dx_[0]
         - dt*qp[-1]*dds(vphi[0])*dx_[0]
         + dds(qp[0])*vphi[-1]*dx_[0]
         + dt*(alpha/A_pv)*mu*qp[0]*vphi[0]*dx_[0])

    L = + D_a*w_s[0]*vphi[-1]*dx_[0] + (rho/A_pv)*qp_.sub(0)*vphi[0]*dx_[0]

    # Need to define explicitly diagonal block associated with Lagrange multiplier
    if bifurcation_p:
        a += Constant(0.)*qp[-2]*vphi[-2]*dx
    
    # Contribution from other mesh segments
    for i in range(1, len(mesh_segments)):

        a += ((rho/A_pv)*qp[i]*vphi[i]*dx_[i]
             + dt*(mu/A_pv)*dds(qp[i])*dds(vphi[i])*dx_[i] #TEST
             - dt*qp[-1]*dds(vphi[i])*dx_[i]
             + dds(qp[i])*vphi[-1]*dx_[i]
             + dt*(alpha/A_pv)*mu*qp[i]*vphi[i]*dx_[i])

        L += + D_a*w_s[i]*vphi[-1]*dx_[i] + (rho/A_pv)*qp_.sub(i)*vphi[i]*dx_[i]

    #Impose static gradient at inlet
    L_max = max(branches_length)
    p_inlet = L_max*p_static_gradient
    print("Impose p_inlet = ", p_inlet)
    #tag = 10
    for tag in inlet_markers:
        # Note : Only parent segments has inlet markers
        for i in range(len(mesh_segments)):
            L += dt*p_inlet*vphi[i]*ds_[i](tag)
    
    # Impose (L1-L2)*dp/dx at outlet
    # FIXME : Check that branches_length and the outlets tags (20, 21, ...)
    # are in the same order
    for idx, length in enumerate(branches_length):
        if length < L_max: # impose 0 otherwise
            p_outlet = p_static_gradient*(L_max - length)
            #tag = 20 + idx
            tag = outlet_markers[idx]
            # Note : Only parent segments has outlet markers
            for i in range(len(mesh_segments)):
                L += -dt*p_outlet*vphi[i]*ds_[i](tag)

    qp0 = Function(QP)
    p0 = qp0.sub(QP.num_sub_spaces() - 1)

    for i,s in enumerate(mesh_segments):
        qp0.sub(i).rename("q"+str(i), "flux"+str(i))
    p0.rename("p", "pressure")

    with XDMFFile(MPI.comm_world, results_dir + "/XDMF/mesh1D.xdmf") as xdmf:
        xdmf.write(mesh)
    # Export mesh segments
    for i,msh in enumerate(mesh_segments):
        with XDMFFile(MPI.comm_world, results_dir + "/XDMF/mesh1D_"+ str(i) + ".xdmf") as xdmf:
            xdmf.write(msh)
    qfile_1D = [XDMFFile(MPI.comm_world, results_dir + "/XDMF/q1D"+ str(i) + ".xdmf")
                for i,s in enumerate(mesh_segments)]
    pfile_1D = XDMFFile(MPI.comm_world, results_dir + "/XDMF/p1D.xdmf")
    wfile_1D = [XDMFFile(MPI.comm_world, results_dir + "/XDMF/w1D" + str(i) + ".xdmf")
                for i,s in enumerate(mesh_segments)]
    qhfile_1D = [HDF5File(MPI.comm_world, results_dir + "/HDF5/q1D"+ str(i) + ".h5", "w")
                 for i,s in enumerate(mesh_segments)]
    phfile_1D = HDF5File(MPI.comm_world, results_dir + "/HDF5/p1D.h5", "w")

    radius_hfile = HDF5File(MPI.comm_world, results_dir + "/HDF5/radius_a.h5", "w")
                 
    # Define Point corresponding to inlet from inlet marking (MeshFunction)
    bc = DirichletBC(V, Constant(0), mf, 10)
    d2v = dof_to_vertex_map(V)
    inlet_dof = list(bc.get_boundary_values().keys())[0]
    vertex = Vertex(mesh, d2v[inlet_dof])
    inlet_point = Point(vertex.x(0), vertex.x(1), vertex.x(2))

    qinfile = open(results_dir + "/q_inlet.txt",'w')
    pinfile = open(results_dir + "/p_inlet.txt",'w')
    winfile = open(results_dir + "/w_inlet.txt",'w')

    # Get dof index of the bifurcation point to apply bifurcation conditions
    if bifurcation_p:
        W_bp_ix = [utils.get_dof_ix_of_point(W, bifurcation_p) for W in QP.sub_spaces()[0:3]] # ix of bifurcation point for velocity spaces
        # DEBUG - Checking bifurcation point coordinates from each segment
        # for i,W in enumerate(QP.sub_spaces()):
        #     dofs_x = W.tabulate_dof_coordinates()
        #     print("Segment (", i, "), dof coords = ", dofs_x[W_bp_ix[i]])


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
                for idx,s in enumerate(mesh_segments):
                    A_pvs[idx].time = time
                    w_s[idx].time = time
                for w in w_s: w.time=time
                alpha.time = time

                # Assemble the system
                system = assemble_mixed_system(a == L, qp0)
                A_list = system[0]
                rhs_blocks = system[1]

                # Apply bifurcation conditions (manually update corresponding blocks)
                if bifurcation_p:
                    ns = len(mesh_segments) # number of segments
                    idx = 0
                    for tag in inlet_markers:
                        A_list[idx*QP.num_sub_spaces() + ns] = utils.add_to_matrix(A_list[idx*QP.num_sub_spaces() + ns], j=0, i=W_bp_ix[idx], value=1.0)
                        A_list[ns*QP.num_sub_spaces() + idx] = utils.add_to_matrix(A_list[ns*QP.num_sub_spaces() + idx], j=W_bp_ix[idx], i=0, value=1.0)
                        idx += 1
                    for tag in outlet_markers:
                        A_list[idx*QP.num_sub_spaces() + ns] = utils.add_to_matrix(A_list[idx*QP.num_sub_spaces() + ns], j=0, i=W_bp_ix[idx], value=-1.0)
                        A_list[ns*QP.num_sub_spaces() + idx] = utils.add_to_matrix(A_list[ns*QP.num_sub_spaces() + idx], j=W_bp_ix[idx], i=0, value=-1.0)
                        idx += 1

                # Solve the system
                A_ = PETScNestMatrix(A_list) # recombine blocks
                b_ = Vector()
                A_.init_vectors(b_, rhs_blocks)
                A_.convert_to_aij() # Convert MATNEST to AIJ for LU solver

                sol_ = Vector(mesh.mpi_comm(), sum([W.dim() for W in QP.sub_spaces()]))
                solver = PETScLUSolver()
                solver.solve(A_, sol_, b_)
                
                # Transform sol_ into qp0 and update qp_
                dim_shift = 0
                for s in range(QP.num_sub_spaces()):
                    qp0.sub(s).vector().set_local(sol_.get_local()[dim_shift:dim_shift + qp0.sub(s).function_space().dim()])
                    dim_shift += qp0.sub(s).function_space().dim()
                    qp0.sub(s).vector().apply("insert")
                    assign(qp_.sub(s), qp0.sub(s))

                # Export results
                for idx,s in enumerate(mesh_segments):
                    # flux
                    qp0.sub(idx).rename("q"+str(idx), "flux"+str(idx))
                    qfile_1D[idx].write(qp0.sub(idx),time)
                    qhfile_1D[idx].write(qp0.sub(idx), "/function", time)
                    # cross-section area
                    A_pvs_ = project(A_pvs[idx], V)
                    A_pvs_.rename("A_pv", "A_pv")
                    # wall velocity
                    w_ = project(w_s[idx], V)
                    w_.rename("w", "w")
                    wfile_1D[idx].write(w_, time)
                # pressure
                p0.rename("p", "pressure")
                pfile_1D.write(p0,time)
                phfile_1D.write(p0, "/function", time)
                radius_hfile.write(interpolate(R_a_val, p0.function_space()), "/function", time)
                # Alpha coeff (as CG1)
                alpha_ = project(alpha, V)
                alpha_.rename("alpha", "alpha")
                
                if MPI.rank(MPI.comm_world) == 0:
                    qinfile.write('%g %g\n'%(time, qp0.sub(0)(inlet_point)))
                    pinfile.write('%g %g\n'%(time, p0(inlet_point)))
                    # Export abs(w(inlet)) - to be compared with w norm on 3D inner wall
                    winfile.write('%g %g\n'%(time, abs(w(inlet_point))))
                # Update time
                time = time + dt

                
    print('** Memory and time usage (1D) **')
    toc = pytime.time()
    
    max_mem_usage = np.max(np.asarray(mem_usage))/1000
    print('Memory usage: %f (mb)' % (max_mem_usage))

    num_dofs = 0
    for fs in QP.sub_spaces():
        num_dofs += fs.dim()
    print('Num of dofs: %g'%num_dofs)
    
    time_per_loop = (toc-tic)/(time/dt)
    print('Time per loop: %g'%time_per_loop)

    pfile_1D.close()
    phfile_1D.close()
    radius_hfile.close()
    qinfile.close()
    pinfile.close()
    winfile.close()
    
    for f in qfile_1D:
        f.close()
    for f in qhfile_1D:
        f.close()
    for f in wfile_1D:
        f.close()
        

### -------------------- MAIN ------------------------- ###
if __name__ == "__main__":
    pvs_model()
### --------------------------------------------------- ###
