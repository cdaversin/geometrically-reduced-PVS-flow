from xii.linalg.matrix_utils import petsc_serial_matrix, is_number
from xii.assembler.average_form import average_cell, average_space

from numpy.polynomial.legendre import leggauss
from dolfin import PETScMatrix, cells, Point, Cell, Function
from petsc4py import PETSc
import numpy as np


def memoize_average(average_mat):
    '''Cached average'''
    cache = {}
    def cached_average_mat(V, TV, reduced_mesh, data):
        key = ((V.ufl_element(), V.mesh().id()),
               (TV.ufl_element(), TV.mesh().id()),
               data['shape'])

        if key not in cache:
            cache[key] = average_mat(V, TV, reduced_mesh, data)
        return cache[key]
    
    return cached_average_mat


@memoize_average
def avg_mat(V, TV, reduced_mesh, data):
    '''
    A mapping for computing the surface averages of function in V in the 
    space TV. Surface averaging is defined as 

    (Pi u)(x) = |C_R(x)|int_{C_R(x)} u(y) dy with C_R(x) the circle of 
    radius R(x) centered at x with a normal parallel with the edge tangent.
    '''
    assert TV.mesh().id() == reduced_mesh.id()
    
    # It is most natural to represent Pi u in a DG space
    assert TV.ufl_element().family() == 'Discontinuous Lagrange'
    
    # Compatibility of spaces
    assert V.ufl_element().value_shape() == TV.ufl_element().value_shape()
    assert average_cell(V) == TV.mesh().ufl_cell()
    assert V.mesh().geometry().dim() == TV.mesh().geometry().dim()

    shape = data['shape']
    # 3d-1d trace
    if shape is None:
        return PETScMatrix(trace_3d1d_matrix(V, TV, reduced_mesh))

    # Surface averages
    Rmat = average_matrix(V, TV, shape)
        
    return PETScMatrix(Rmat)
                

def average_matrix(V, TV, shape):
    '''
    Averaging matrix for reduction of g in V to TV by integration over shape.
    '''
    # We build a matrix representation of u in V -> Pi(u) in TV where
    #
    # Pi(u)(s) = |L(s)|^-1*\int_{L(s)}u(t) dx(s)
    #
    # Here L is the shape over which u is integrated for reduction.
    # Its measure is |L(s)|.
    
    mesh_x = TV.mesh().coordinates()
    # The idea for point evaluation/computing dofs of TV is to minimize
    # the number of evaluation. I mean a vector dof if done naively would
    # have to evaluate at same x number of component times.
    value_size = TV.ufl_element().value_size()

    mesh = V.mesh()
    # Eval at points will require serch
    tree = mesh.bounding_box_tree()
    limit = mesh.num_cells()

    TV_coordinates = TV.tabulate_dof_coordinates().reshape((TV.dim(), -1))
    line_mesh = TV.mesh()

    points_file = open('averaging_points.csv', 'w')
    points_file.write('x, y, z \n')
    TV_dm = TV.dofmap()
    V_dm = V.dofmap()
    # For non scalar we plan to make compoenents by shift
    if value_size > 1:
        TV_dm = TV.sub(0).dofmap()

    Vel = V.element()               
    basis_values = np.zeros(V.element().space_dimension()*value_size)
    with petsc_serial_matrix(TV, V) as mat:

        for line_cell in cells(line_mesh):
            # Get the tangent (normal of the plane which cuts the virtual
            # surface to yield the bdry curve
            v0, v1 = mesh_x[line_cell.entities(0)]
            n = v0 - v1

            # The idea is now to minimize the point evaluation
            scalar_dofs = TV_dm.cell_dofs(line_cell.index())
            scalar_dofs_x = TV_coordinates[scalar_dofs]
            for scalar_row, avg_point in zip(scalar_dofs, scalar_dofs_x):
                # Avg point here has the role of 'height' coordinate
                quadrature = shape.quadrature(avg_point, n)
                integration_points = quadrature.points
                wq = quadrature.weights

                curve_measure = sum(wq)

                data = {}
                for index, ip in enumerate(integration_points):
                    points_file.write(str(ip[0]) + ', ' + str(ip[1]) + ', ' + str(ip[2]) + '\n')
                    c = tree.compute_first_entity_collision(Point(*ip))
                    if c >= limit: continue

                    Vcell = Cell(mesh, c)
                    vertex_coordinates = Vcell.get_vertex_coordinates()
                    cell_orientation = Vcell.orientation()
                    basis_values[:] = Vel.evaluate_basis_all(ip, vertex_coordinates, cell_orientation)

                    cols_ip = V_dm.cell_dofs(c)
                    values_ip = basis_values*wq[index]
                    # Add
                    for col, value in zip(cols_ip, values_ip.reshape((-1, value_size))):
                        if col in data:
                            data[col] += value/curve_measure
                        else:
                            data[col] = value/curve_measure
                    

                # The thing now that with data we can assign to several
                # rows of the matrix
                column_indices = np.array(list(data.keys()), dtype='int32')
                for shift in range(value_size):
                    row = scalar_row + shift
                    column_values = np.array([data[col][shift] for col in column_indices])
                    mat.setValues([row], column_indices, column_values, PETSc.InsertMode.INSERT_VALUES)
            # On to next avg point
        # On to next cell
    points_file.close()
    return mat



# --------------------------------------------------------------------

if __name__ == '__main__':
    print('Hello')
    ## Todo: write tests