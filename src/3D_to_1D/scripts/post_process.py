from dolfin import *
import pylab as plt
import os.path
import sys
import math
import numpy as np

model1D_path = '../3D_to_1D'
sys.path.insert(1, model1D_path)
import pvs1D_utils as utils


def generate_inflow_data(dir_3D, dir_1D):
    # 3D
    t,v = plt.loadtxt(dir_3D + '/inflow.txt').transpose()
    header_v_avg = "T \t V"
    data_v_inlet = plt.array([t,v]).transpose()
    plt.savetxt(dir_3D + '/vin.dat', data_v_inlet, delimiter="\t", header=header_v_avg)

    # 1D
    t,q1D = plt.loadtxt(dir_1D + "/q_inlet.txt").transpose()
    header_q_avg = "T \t Q"
    data_q_inlet = plt.array([t,q1D]).transpose()
    plt.savetxt(dir_1D + '/qin.dat', data_q_inlet, delimiter="\t", header=header_q_avg)

def generate_wall_velocity_data(dir_3D, dir_1D):
    #1D
    t,w = plt.loadtxt(dir_3D + '/winlet.txt').transpose()
    header_w = "T \t W"
    data_w_inlet = plt.array([t,w]).transpose()
    plt.savetxt(dir_3D + '/win.dat', data_w_inlet, delimiter="\t", header=header_w)
    # 1D
    t,w1D = plt.loadtxt(dir_1D + "/w_inlet.txt").transpose()
    data_w1D_inlet = plt.array([t,w1D]).transpose()
    plt.savetxt(dir_1D + '/win.dat', data_w1D_inlet, delimiter="\t", header=header_w)

def read_1D_mesh(case_path, case_prefix, result_dir, coord_factor):
    mesh1D = Mesh()
    mesh1D_file = case_path + case_prefix + "_centerline_mesh.xdmf"
    with XDMFFile(MPI.comm_world, mesh1D_file) as xdmf:
        xdmf.read(mesh1D)
    mesh1D.coordinates()[:]/=coord_factor
    with XDMFFile(MPI.comm_world, result_dir + "1D/XDMF/mesh1D.xdmf") as xdmf:
        xdmf.write(mesh1D)
    return mesh1D

def build_1D_coarse_meshes(case_path, case_prefix, results_path, mesh1D):

    filename = case_path + case_prefix  + "_branches_marker.xdmf"
    # Branches markers only exist for bifurcating geometries
    if os.path.exists(filename):
        marker1D = MeshFunction('size_t', mesh1D, mesh1D.topology().dim(), 0)
        # Loading branches markers
        with XDMFFile(MPI.comm_world, case_path + case_prefix  + "_branches_marker.xdmf") as xdmf:
            xdmf.read(marker1D)

        # Build segment meshes (MeshView)
        mesh_segments  = []
        cell_maps = []
        for tag in np.unique(marker1D.array()):
            mesh_b = MeshView.create(marker1D, tag)
            mesh_segments.append(mesh_b)
            cell_maps.append(mesh_b.topology().mapping()[mesh1D.id()].cell_map())
    else:
        mesh_segments = [mesh1D]
        cell_maps = []

    mesh1D_c_list = [] # Coarses submeshes with mesh1D as parent
    mesh1D_cs_list = [] # Coarse submeshes with mesh_segments as parents
    marker = MeshFunction('size_t', mesh1D, mesh1D.topology().dim(), 50)
    for i,m in enumerate(mesh_segments):
        marker_segment = MeshFunction('size_t', m, m.topology().dim(), 50)

        start = 2
        end = m.num_entities(0) - start

        cells_connectivity = np.asarray(m.cells())
        vertex = start
        connected_cells = np.where(cells_connectivity == vertex)
        c = connected_cells[0][-1]
        if cell_maps:
            marker[cell_maps[i][c]] = i
        else:
            marker[c] = i
        marker_segment[c] = i
        # Find other vertex
        cell = Cell(m, c)
        vertices = cell.entities(0)
        vertex = vertices[np.where(vertices != vertex)[0]]
        vertex = vertex[0]
    
        while vertex != end:
            connected_cells = np.where(cells_connectivity == vertex)
            connected_cells = connected_cells[0]
            for c in (c for c in connected_cells if ((cell_maps and marker[cell_maps[i][c]] != i) or (not cell_maps and marker[c] != i))):
                if cell_maps:
                    marker[cell_maps[i][c]] = i
                else:
                    marker[c] = i
                marker_segment[c] = i
                # Find other vertex
                cell = Cell(m, c)
                vertices = cell.entities(0)
                vertex = vertices[np.where(vertices != vertex)[0]]
                vertex = vertex[0]

        mesh_ = MeshView.create(marker, i) # MeshViews with mesh1D as parent
        mesh_segment = MeshView.create(marker_segment, i) # MeshViews with mesh_segments as parents
        mesh1D_c_list.append(mesh_)
        mesh1D_cs_list.append(mesh_segment)

        filename = results_path + "/avg/XDMF/mesh1D_c_"+ str(i) + ".xdmf"
        with XDMFFile(MPI.comm_world, filename) as xdmf:
            xdmf.write(mesh_)

    return mesh_segments, mesh1D_c_list, mesh1D_cs_list 
    
def read_3D_mesh(case_path, case_prefix, coord_factor):
    mesh3D = Mesh()
    mesh3D_file = case_path + case_prefix + "_PVS.xdmf"
    with XDMFFile(MPI.comm_world, mesh3D_file) as xdmf:
        xdmf.read(mesh3D)    
    mesh3D.coordinates()[:]/=coord_factor
    return mesh3D

def read_radius_data(case_path, case_prefix, mesh1D, coord_factor):
    radius_hfile  = HDF5File(MPI.comm_world, case_path + case_prefix + "_HDF5/centerline_radius.h5", "r")
    attr = radius_hfile.attributes("/function")
    nsteps = attr['count']
    radius_a = Function(FunctionSpace(mesh1D, 'CG', 1))
    for i in range(nsteps):
        dataset = "/function/vector_%d"%i
        radius_hfile.read(radius_a, dataset)

    radius_a.vector()[:]/=coord_factor
    radius_a.vector()[:]*= 0.5 ## TODO: Radius is 2 times too big, why?
    radius_pv = project(2.95*radius_a, FunctionSpace(mesh1D, 'CG', 1))

    return (radius_a, radius_pv)

def read_1D_solutions(results_path, mesh1D, mesh_segments):
    p1Ds = []
    fname_p = results_path + '1D/HDF5/p1D.h5'
    fp = HDF5File(MPI.comm_world, fname_p, 'r')
    attr = fp.attributes("/function")
    nsteps = attr['count']
    p1D = Function(FunctionSpace(mesh1D, "CG", 1))
    for i in range(0, nsteps):
        name = "/function/vector_%d"%(i)
        fp.read(p1D, name)
        p1D_copy = p1D.copy(deepcopy=True)
        p1Ds.append(p1D_copy)

    # Reading all the mesh segments
    q1Ds = []
    for i,s in enumerate(mesh_segments):
        qs = []
        fname_q = results_path + "1D/HDF5/q1D" + str(i) + ".h5"
        fq = HDF5File(MPI.comm_world, fname_q, 'r')
        attr = fq.attributes("/function")
        nsteps = attr['count']

        q1D = Function(FunctionSpace(s, "CG", 2))
        for j in range(0, nsteps):
            name = "/function/vector_%d"%(j)
            fq.read(q1D, name)
            q1D_copy = q1D.copy(deepcopy=True)
            qs.append(q1D_copy)
        q1Ds.append(qs)

    return (p1Ds, q1Ds)
    
def read_3D_solutions(results_path, mesh3D):
    us, ps = [], []
    fname_u = results_path + '3D/HDF5/u.h5'
    fname_p = results_path + '3D/HDF5/p.h5'
    fu = HDF5File(MPI.comm_world, fname_u,'r')
    fp = HDF5File(MPI.comm_world, fname_p,'r')
    attr = fu.attributes("/function")
    nsteps = attr['count']
    
    u = Function(VectorFunctionSpace(mesh3D, "CG", 2))
    p = Function(FunctionSpace(mesh3D, "CG", 1))
    for i in range(0, nsteps):
        name = "/function/vector_%d"%i
        fu.read(u, name)
        u_copy = u.copy(deepcopy=True)
        us.append(u_copy)
        fp.read(p, name)
        p_copy = p.copy(deepcopy=True)
        ps.append(p_copy)
    return (ps, us)

def read_tangent(case_path, mesh1D, mesh_segments):
    # Read in tangent components
    DG = FunctionSpace(mesh1D, 'DG', 0)
    tang_comps = []
    for c in range(3):
        f = Function(DG)
        fFile = HDF5File(MPI.comm_world, case_path + "tangent%i.h5"%c,"r")
        fFile.read(f,"/f")
        fFile.close()
        tang_comps.append(f)

    return tang_comps

def compute_correction_factor(tang_comps, radius_a, radius_pv, mesh1D_c, mesh3D):
    ## The averaging function uses the tangent components to form a Frenet frame    
    ## The pvs isn't perfectly radially symmetric, so our calculation of the 
    # area as a_pv = pi*r2^2-pi*r1^2 isn't quite correct
    # But we can calculate a correction factor
    ones = Function(FunctionSpace(mesh3D, 'CG', 1))
    ones.vector()[:] = 1.0
    V1_c = FunctionSpace(mesh1D_c, 'CG', 1)
    ones_avg = utils.average(V1_c, ones, tang_comps, radius_a, radius_pv, is_vector=False, debug=False)
    
    correction_factor = ones_avg.vector().get_local()
    return correction_factor

def compute_1D_avg_from_3D_solution(case_path, case_prefix, results_path, coord_factor, dt):
    
    mesh1D = read_1D_mesh(case_path, case_prefix, results_path, coord_factor)
    mesh3D = read_3D_mesh(case_path, case_prefix, coord_factor)
    
    radius_a, radius_pv = read_radius_data(case_path, case_prefix, mesh1D, coord_factor)
    A_pv = project(math.pi*(radius_pv**2.0-radius_a**2.0), FunctionSpace(mesh1D, 'CG', 1))
    
    mesh_segments, mesh1D_c_list, mesh1D_cs_list = build_1D_coarse_meshes(case_path, case_prefix, results_path, mesh1D)

    # Now with all the time steps
    p1Ds, q1Ds = read_1D_solutions(results_path, mesh1D, mesh_segments)
    ps, us = read_3D_solutions(results_path, mesh3D)

    tang_comps = read_tangent(case_path, mesh1D, mesh_segments)

    p_avg_s = []
    q_avg_s = []
    p1D_c_s = []
    q1D_c_s = []

    # Pressure
    V1 = FunctionSpace(mesh1D, 'CG', 1)
    correction_factor = compute_correction_factor(tang_comps,
                                                  radius_a,
                                                  radius_pv, mesh1D,
                                                  mesh3D)
    ## Compute average of 3D pressure
    p_avg_ts = []
    p_avg_file = XDMFFile(mesh1D.mpi_comm(), results_path + "/avg/XDMF/p_avg.xdmf")
    ph_avg_file = HDF5File(mesh1D.mpi_comm(), results_path + "/avg/HDF5/p_avg.h5", "w")
    
    for t,p in enumerate(ps):
        #p_avg = utils.average(V1, p, tang_comps, radius_a, radius_pv, is_vector=False, debug="avg_points.csv")
        p_avg = utils.average(V1, p, tang_comps, radius_a, radius_pv, is_vector=False, debug=False)
        p_avg.vector()[:] = p_avg.vector().get_local()/correction_factor
        p_avg.rename('p_avg', '0.0')

        #p_avg_file.write(p_avg, (t+1)*dt)
        p_avg_file.write(p_avg, t*dt)
        ph_avg_file.write(p_avg, "/function", t*dt)
        p_avg_ts.append(p_avg)
                    
    p_avg_s.append(p_avg_ts) # p_avg_s[branch][time]

    ## Project 1D pressure to coarse mesh
    p1D_c_ts = []
    p1D_file = XDMFFile(mesh1D.mpi_comm(), results_path + "/1D/XDMF/p1D_c.xdmf")
    for t,p1D in enumerate(p1Ds):
        p1D_c = interpolate(p1D, V1)
        p1D_c.rename('p1D_c', '0.0')
        p1D_c_ts.append(p1D_c)
        p1D_file.write(p1D_c, t*dt)
    p1D_c_s.append(p1D_c_ts)

    p_avg_file.close()
    p1D_file.close()

    print(len(mesh1D_c_list))
    for i,m in enumerate(mesh1D_c_list):

        correction_factor = compute_correction_factor(tang_comps,
                                                      radius_a,
                                                      radius_pv, m,
                                                      mesh3D)

        V1_c = FunctionSpace(m, 'CG', 1)
        V1_cs = FunctionSpace(mesh1D_cs_list[i], 'CG', 1)

        q_avg_file = XDMFFile(mesh1D.mpi_comm(), results_path + "/avg/XDMF/q_avg" + str(i) + ".xdmf")
        qh_avg_file = HDF5File(m.mpi_comm(), results_path + "/avg/HDF5/q_avg" + str(i) + ".h5", "w")
        q1D_file = XDMFFile(mesh1D.mpi_comm(), results_path + "/1D/XDMF/q1D_c" + str(i) + ".xdmf")
        
        ## Compute average of 3D flux
        q_avg_ts = []
        for t,u in enumerate(us):
            q_avg = utils.average(V1_c, u, tang_comps, radius_a, radius_pv, is_vector=True, debug=False)
            print(t, norm(q_avg))
            q_avg.vector()[:] = q_avg.vector().get_local()/correction_factor
            q_avg = project(q_avg*A_pv, V1_c) # cross-section flux is not an average, so we need to multiply back the cross-section area
            q_avg.rename('q_avg', '0.0')
            q_avg_ts.append(q_avg)
            q_avg_file.write(q_avg, t*dt)
            qh_avg_file.write(q_avg, "/function", t*dt)
        q_avg_s.append(q_avg_ts)

        ## Project 1D flux to coarse mesh
        ## (with mesh_segments as parents since q1D_s[i] is defined on the corresponding mesh segments)
        q1D_c_ts = []
        for t, q1D in enumerate(q1Ds[i]):
            q1D_c = interpolate(q1D, V1_cs)
            q1D_c.rename('q1D_c', '0.0')
            q1D_c_ts.append(q1D_c)
            q1D_file.write(q1D_c, t*dt)
            
        q1D_c_s.append(q1D_c_ts)

        q_avg_file.close()
        q1D_file.close()
        qh_avg_file.close()

        # Export mesh
        with XDMFFile(MPI.comm_world, results_path + "/avg/XDMF/mesh1D_c_"+ str(i) + ".xdmf") as xdmf:
            xdmf.write(m)

    # Export A_pv
    # XDMF
    with XDMFFile(mesh1D.mpi_comm(), results_path + "/1D/XDMF/A_pv.xdmf") as xdmf:
        xdmf.write(A_pv)
    # HDF5
    with HDF5File(mesh1D.mpi_comm(), results_path + "/1D/HDF5/A_pv.h5", "w") as h5file:
        h5file.write(A_pv, "/function")


    #return (p_avg_s, q_avg_s)
    return (p_avg_s, q_avg_s, p1D_c_s, q1D_c_s, tang_comps)
