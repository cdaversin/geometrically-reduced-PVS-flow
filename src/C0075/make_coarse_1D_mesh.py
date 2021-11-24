
import os
import meshio
from fenics import *
import numpy as np

def build_mesh():
    # Read 1D mesh
    mesh1D = Mesh()
    case_prefix = 'C0075_clip2_mesh1_0.95_ratio'
    mesh1D_file = case_prefix + "_centerline_mesh.xdmf"
    with XDMFFile(MPI.comm_world, mesh1D_file) as xdmf:
        xdmf.read(mesh1D)

    L_PVS = 44e-3 # [mm]
    coord_factor = 1.2/L_PVS #C0075
    
    mesh1D.coordinates()[:]/=coord_factor

    # We iterate through the mesh, adding specific points to the daughter and parent segments
    d1_ixs = [5 + 10*i for i in list(range(0, 6))]
    d2_ixs = [155 + 10*i for i in list(range(0, 8))]
    p_ixs = [85 + 10*i for i in list(range(1, 6))]

    V = FunctionSpace(mesh1D, 'CG', 1)
    coords = mesh1D.coordinates()

    mesh_and_ixs = {'red_p.xdmf':p_ixs, 'red_d1.xdmf':d1_ixs, 'red_d2.xdmf':d2_ixs}

    for key in mesh_and_ixs:

        filename = key

        centerline_points = np.empty((0,3), int)
        cells_array = np.empty((0,2), int)

        cell = 0

        for ix, i in enumerate(mesh_and_ixs[key]):
            centerline_points = np.append(centerline_points, [coords[i, :]], axis=0)
        
            if ix: cells_array = np.append(cells_array, [([ix-1, ix])], axis=0)

        centerline_cells = {"line": cells_array}

        if not os.path.exists(os.path.splitext(filename)[0] + ".h5"):
            f = open(os.path.splitext(filename)[0] + ".h5", "w+")
            f.close()

        meshio.write_points_cells(
            filename,
            centerline_points,
            centerline_cells,)
    return mesh_and_ixs.keys()
