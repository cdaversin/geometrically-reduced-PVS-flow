import meshio
import numpy as np
from math import *

def build_pvs_mesh(length, r1, r2, npts, filename):
    # Points
    r = (r1 + r2)/2
    centerline_points = np.empty((0,3), int)
    for i in range(0, npts+1):
        centerline_points = np.append(centerline_points, [[i*(length/npts),r,0]], axis=0) # along x
        # TEST - centerline x=y
        # d = i*(length/npts)*(0.5*(1/sqrt(2))*(r2-r1))
        # centerline_points = np.append(centerline_points, [[d, d, 0]], axis=0) # x=y

    # Cells    
    cells_array = np.empty((0,2), int)
    for i in range(1, npts+1):
        cells_array = np.append(cells_array, [[i-1, i]], axis=0)
        
    centerline_cells = {"line": cells_array}
    meshio.write_points_cells(
        filename,
        centerline_points,
        centerline_cells,
    )


### -------------------- MAIN ------------------------- ###
if __name__ == "__main__":
    build_pvs_mesh(1, 20, 60, 10, "pvs1D.xdmf")
### --------------------------------------------------- ###
