## Functions and definitions used for implementing the ##
## reduced 1d pvs model                                ##

from dolfin import *
import numpy as np

import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc


### ---------------- Velocity profile  ------------------ ###
# The 1d model uses the assumption u=[u_s, 0, 0] with
#    u_s = u_hat(s,t) u_vp(r)
# where u_vp(r) is some given velocity profile

# Take u_vp as annular Poiseuille flow
def u_pois_ann(r, R_a, R_pv): ## todo: is this normalized correctly?
    return (1.0 - r**2.0/R_a**2.0 + (R_pv**2.0 - R_a**2.0)*np.log(r/R_a)/(R_a**2.0*np.log(R_pv/R_a)))

def u_pois_ann_diff(r, R_a, R_pv):
    return -2.0*r/R_a**2.0 + (R_pv**2.0 - R_a**2.0)/(r*(R_a**2.0*np.log(R_pv/R_a)))

# Normalize it
def u_vp(r, R_a, R_pv):
    R_m = (R_a + R_pv)/2.0
    norm_factor = 1.0/u_pois_ann(R_m, R_a, R_pv)
    return u_pois_ann(r, R_a, R_pv)*norm_factor

### ----------------------------------------------------- ###

### -------------- Reduced model parameters ------------- ###

# The velocity profile later enters as into the coefficient alpha
# in the 1d variational formulation

# Derivative of velocity profile, i.e. d/dr u_vp (computed numerically)
def u_vp_diff(r, R_a, R_pv):
    R_m = (R_a + R_pv)/2.0
    norm_factor = 1.0/u_pois_ann(R_m, R_a, R_pv)
    return u_pois_ann_diff(r, R_a, R_pv)*norm_factor

# Average of velocity profile (computed analytically)
def u_vp_avg(R_a, R_pv):
    A_pv = np.pi*(R_pv**2.0-R_a**2.0)
    return (2.0/3.0)*(np.pi/A_pv)*(R_pv**2.0-R_a**2.0)

# Average of derivative of velocity profile (computed analytically)
def u_vp_diff_avg(R_a, R_pv):
    A_pv = np.pi*(R_pv**2.0-R_a**2.0)
    return (-4.0*np.pi/A_pv)*( (1.0/3.0)*(R_pv**3.0-R_a**3.0) + R_a**2.0*R_pv-R_pv**2.0*R_a)/(R_a-R_pv)**2.0

### ----------------------------------------------------- ###

### ---------- Cross-section area and diameter ---------- ###
class A_PV(UserExpression):
    def __init__(self, R_a, R_pv, time, **kwargs):
        self.time = time
        self.R_a = R_a
        self.R_pv = R_pv
        super().__init__(**kwargs)

    def eval(self, values, x):
        R_a_val = self.R_a(self.time)(x)
        R_pv_val = self.R_pv(self.time)(x)
        values[0] = np.pi*(R_pv_val**2.0 - R_a_val**2.0)

    def value_shape(self):
        return ()

class D(UserExpression):
    def __init__(self, R, time, **kwargs):
        self.time = time
        self.R = R
        super().__init__(**kwargs)

    def eval(self, values, x):
        R_val = self.R(self.time)(x)
        values[0] = 2.0*np.pi*R_val

    def value_shape(self):
        return ()

### ----------------------------------------------------- ###

### ------------ Arterial wall velocity ----------------- ###

class W(UserExpression):
    def __init__(self, R_a, time, delta_t, **kwargs):
        self.time = time
        self.R_a = R_a
        self.delta_t = delta_t
        super().__init__(**kwargs)

    def eval(self, values, x):
        R_a_val_m = self.R_a(self.time - self.delta_t)(x)
        R_a_val_p = self.R_a(self.time + self.delta_t)(x)
        R_a_val = self.R_a(self.time)(x)
        #values[0] = (R_a_val_p - R_a_val_m)/self.delta_t (??)

        # Wall velocity : (deformation(t) - deformation(t-dt))/dt
        # = ((Ra(t) - Ra(0)) - (Ra(t-dt) - Ra(0)))/dt
        # = (Ra(t) - Ra(t-dt))/dt
        values[0] = (R_a_val - R_a_val_m)/self.delta_t

    def value_shape(self):
        return ()

### ----------------------------------------------------- ###

### ---------- Alpha coefficient (1D form) -------------- ###

class Alpha(UserExpression):
    def __init__(self, R_a, R_pv, time, **kwargs):
        self.time = time
        self.R_a = R_a
        self.R_pv = R_pv
        super().__init__(**kwargs)

    def eval(self, values, x):
        R_a_val = self.R_a(self.time)(x)
        R_pv_val = self.R_pv(self.time)(x)
        
        D_a = 2.0*np.pi*R_a_val
        D_pv = 2.0*np.pi*R_pv_val

        A_pv = np.pi*(R_pv_val**2.0-R_a_val**2.0)

        numerator = (D_a*u_vp_diff(R_a_val, R_a_val, R_pv_val)
               - D_pv*u_vp_diff(R_pv_val, R_a_val, R_pv_val))
        values[0] = numerator/(A_pv*u_vp_avg(R_a_val, R_pv_val))

    def value_shape(self):
        return ()

### ----------------------------------------------------- ###

### ----------------------------------------------------- ###

# ----- Compute averaged 3D variable ----- # 

def average(V, func, tang_comps, radius_a, radius_pv, is_vector, debug=""):

    # If debug is on we write the averaging points to file
    if debug: 
        points_file = open(debug, 'w')
        points_file.write('x, y, z \n')

        # We also keep track of how many times the point we try to evaluate is 
        # inside or outside the domain
        eval_succ, eval_fails = 0, 0

    mesh1D_c = V.mesh()
    V1_c = FunctionSpace(mesh1D_c, 'CG', 1)

    V = V1_c
    p_avg_f = Function(V)
    for ix, x0 in enumerate(V.tabulate_dof_coordinates()):
        
        r1 = radius_a(x0)
        r2 = radius_pv(x0)

        A_pv = np.pi*(r2**2.0-r1**2.0)

        # Get tangent vector at x0
        tangent = np.array([tang_comps[0](x0), tang_comps[1](x0), tang_comps[2](x0)])
        tangent /= np.linalg.norm(tangent)
        
        # Compute normal vector at x0
        normal = np.array([1, 0, -tangent[0]/tangent[2]])
        normal /= np.linalg.norm(normal)

        # Compute binormal vector at x0
        binormal = np.cross(tangent, normal)
        binormal /= np.linalg.norm(binormal)

        # tangent, normal and binormal are now perpendicular

        # A maps from Frenet frame into the (x,y,z) coordinates
        A = np.empty((3,3))
        A[:,0] = normal
        A[:,1] = binormal
        A[:,2] = tangent
        
        #rs = np.linspace(0.5*r1, 1.4*r2, 30)
        rs = np.linspace(1.1*r1, 0.9*r2, 50)

        dr = rs[1] - rs[0]
        thetas = np.linspace(0, 2.0*np.pi, 10)
        dtheta = thetas[1] - thetas[0]

        p_avg = 0

        for theta in thetas:
            for r in rs:
                
                # Make point in circle with respect to Frenet frame 
                x = r*cos(theta)
                y = r*sin(theta)
                z = 0

                x_tilde = np.asarray([x, y, z])

                # Map from Frenet frame into (x,y,z)
                xx = A.dot(x_tilde).flatten() + x0
                
                try:
                    if is_vector:
                        pval = np.dot(func(xx), tangent)
                    else:
                        pval = func(xx)
                    
                    p_avg += r*dr*dtheta*pval
                    
                    if debug: 
                        points_file.write(f'{xx[0]}, {xx[1]}, {xx[2]}, {ix} \n')
                        eval_succ += 1
                except:
                    if debug:
                        eval_fails += 1
                    
        p_avg *= 1.0/A_pv
        p_avg_f.vector()[ix] = p_avg
                    
    if debug:
        print('Point eval success:', eval_succ, 'fails:', eval_fails)
        p_avg_f.rename('p', '0.0')
        File('p_avg.pvd') << p_avg_f

        points_file.close()

    return p_avg_f

            
def get_dof_ix_of_point(func_space, point):
    dof_coords = func_space.tabulate_dof_coordinates()
    return np.where(np.linalg.norm(dof_coords-point, axis=1)<DOLFIN_EPS)[0][0]

def add_to_matrix(A, i, j, value):
    # To add to existing values:
    ADD_MODE = PETSc.InsertMode.ADD

    # To replace values instead:
    #ADD_MODE = PETSc.InsertMode.INSERT
    Am = as_backend_type(A).mat()

    # If the value you want to modify is not allocated as a nonzero, you need to
    # set this option (with some cost to performance).  Ideally, you would
    # set up a matrix with the necessary space allocated, assemble into that one,
    # and then edit values without this.
    Am.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

    # In parallel, you want to make sure a given value is added only once, ideally
    # on the process that owns that row of the matrix.  (Can skip the check for
    # ownership range in serial.)
    Istart, Iend = Am.getOwnershipRange()
    if(i<Iend and i>=Istart):
        Am.setValue(i,j,value,addv=ADD_MODE)
    Am.assemble()
    return A

