
from dolfin import *
import numpy as np


# Returns mesh functions marking bifurcation point with tag 1
def bifurcation_point(mesh):

    # Finding index of dof present in 3 cells
    cells_connectivity = np.asarray(mesh.cells())
    cells_connectivity = np.concatenate(cells_connectivity)
    count_occurences = np.bincount(cells_connectivity)
    bifurcation_index = np.where(count_occurences == 3)

    if len(bifurcation_index[0]) == 0:
        raise ValueError("No bifurcation point found")

    # Create corresponding marker
    bifurcation_mf = MeshFunction('size_t',mesh, mesh.topology().dim()-1)
    for i in bifurcation_index[0]:
        bifurcation_mf[i] = 1

    return bifurcation_mf

# Returns a mesh function with branch marking
# The line segments are marked with the tag or the corresponding inlet(10, 11, ...)/outlet(20, 21, ...)
# In case of multiple bifurcations, the line segments between bifurcation points are tagged 30, 31, ...
def branches_marker(mesh, mf, inlet_markers, outlet_markers):
    # Inlets : 10, 11, ...
    # Outlets : 20, 21, ...
    marker_bf = bifurcation_point(mesh) # tag : 1
    branches_marker = MeshFunction('size_t',mesh, mesh.topology().dim())

    inlet_vertices = [mf.where_equal(i) for i in inlet_markers]
    outlet_vertices = [mf.where_equal(o) for o in outlet_markers]
    cells_connectivity = np.asarray(mesh.cells())

    # Parent branch(es) : From inlet(s) to bifurcation point
    for i, iv in enumerate(inlet_vertices) :
        vertex = iv[0]
        # While we don't reach a bifurcation point
        while marker_bf[vertex] != 1:
            connected_cells = np.where(cells_connectivity == vertex)
            connected_cells = connected_cells[0]
            for c in (c for c in connected_cells if branches_marker[c] != inlet_markers[i]):
                # Mark cell
                branches_marker[c] = inlet_markers[i]
                # Find other vertex
                cell = Cell(mesh, c)
                vertices = cell.entities(0)
                vertex = vertices[np.where(vertices != vertex)[0]]

    # Daughter branch(es) : From outlet to bifurcation point
    for o, ov in enumerate(outlet_vertices) :
        vertex = ov[0]
        # While we don't reach a bifurcation point
        while marker_bf[vertex] != 1:
            connected_cells = np.where(cells_connectivity == vertex)
            connected_cells = connected_cells[0]
            for c in (c for c in connected_cells if branches_marker[c] != outlet_markers[o]):
                # Mark cell
                branches_marker[c] = outlet_markers[o]
                # Find other vertex
                cell = Cell(mesh, c)
                vertices = cell.entities(0)
                vertex = vertices[np.where(vertices != vertex)[0]]

    # TODO/FIXME : From one bifurcation point to another (More than one bifurcation)
    bifurcation_vertices = marker_bf.where_equal(1)
    for b, bv in enumerate(bifurcation_vertices) :
        # Find first vertex of the branch which is not the bifurcation point
        connected_cells = np.where(cells_connectivity == bv)
        connected_cells = connected_cells[0]
        # Bifurcation point is shared by three cells : find the ones that haven't been marked yet
        tag = 30
        for c in (c for c in connected_cells if branches_marker[c] == 0):
            # Mark cell
            branches_marker[c] =  tag
            # Find other vertex
            cell = Cell(mesh, c)
            vertices = cell.entities(0)
            vertex = vertices[np.where(vertices != bv)[0]]
            # Mark rest of this line segment
            # While we don't reach another bifurcation point
            while marker_bf[vertex] != 1:
                connected_cells = np.where(cells_connectivity == vertex)
                connected_cells = connected_cells[0]
                for c2 in (c2 for c2 in connected_cells if branches_marker[c2] != tag):
                    # Mark cell
                    branches_marker[c2] = tag
                    # Find other vertex
                    cell = Cell(mesh, c2)
                    vertices = cell.entities(0)
                    vertex = vertices[np.where(vertices != vertex)[0]]
            tag += 1

    return branches_marker

def segment_mesh_mf(mesh_segment, mf, inlet_markers, outlet_markers):

    segment_mf = MeshFunction('size_t', mesh_segment, mesh_segment.topology().dim() - 1)

    vmap = mesh_segment.topology().mapping()[mf.mesh().id()].vertex_map()

    markers = inlet_markers + outlet_markers
    marker_bf = bifurcation_point(mf.mesh()) # tag : 1
    for v in range(mesh_segment.num_vertices()):
        for m in markers:
            if mf[vmap[v]] == m: # Inlets/Outlets
                segment_mf[v] = m
        if marker_bf[vmap[v]] == 1: # Bifurcation
            segment_mf[v] = 1

    return segment_mf


def get_tangent_vector(mesh):
    ### Construct tangent vector explicitly from mesh vertices
    DG = FunctionSpace(mesh, 'DG', 0)
    DGdmap = DG.dofmap()
    tang_comps = [Function(DG), Function(DG), Function(DG)]
    tau_ = np.asarray([0,0,0])
    for cell in cells(mesh):
        # Compute tangent vector from vertices coordinates
        vx = cell.get_vertex_coordinates()
        vx0 = np.asarray(vx[:mesh.geometry().dim()])
        vx1 = np.asarray(vx[mesh.geometry().dim():])
        tau = vx0 - vx1
        tau /= np.linalg.norm(tau) # normalize to 1

        # Orientation : if not facing same direction as previous
        # tangent vector (i.e. dot product < 0), change sign
        if np.dot(tau, tau_) < 0:
            tau = -tau
        tau_ = tau

        # Assign tangent to DG0 dofs
        cell_dofs = DGdmap.cell_dofs(cell.index())
        for j in range(mesh.geometry().dim()):
            tang_comps[j].vector()[cell_dofs] = tau[j]

    for i,f in enumerate(tang_comps):
        fFile = HDF5File(MPI.comm_world,"tangent%i.h5"%i,"w")
        fFile.write(f,"/f")
        fFile.close()
        
    return tang_comps

