from abc import ABCMeta, abstractmethod
from collections import namedtuple
from itertools import product
from math import pi, sqrt
import numpy as np

import quadpy

from numpy.polynomial.legendre import leggauss
import dolfin as df

from xii.linalg.matrix_utils import is_number
from xii.assembler.average_form import average_space
from xii.meshing.make_mesh_cpp import make_mesh

Quadrature = namedtuple('quadrature', ('points', 'weights'))


class BoundingSurface(metaclass=ABCMeta):
    '''Shape used for reducing a 3d function to 1d by carrying out integration'''
    
    @abstractmethod
    def quadrature(self, x0, n):
        '''Quadrature weights and points for reduction'''
        pass

class AnnularDisk(BoundingSurface):
    '''Annular disk in plane(x0, n) with inner radius given by radius1(x0)
    and outer radius given by radius2(x0)'''    
    def __init__(self, radius1, radius2, ratio, degree):
        self.radius1 = radius1
        self.radius2 = radius2

        # Will use quadrature from quadpy over unit disk in z=0 plane
        # and center (0, 0, 0)
        quad = quadpy.disk.Lether(degree)
        xq, wq = quad.points, quad.weights
        
        xq = []
        wq = []
        for theta in np.linspace(0, 2.0*3.14159, degree):
            for r in np.linspace(ratio, 1.0, degree):
                xq.append([r*np.sin(theta), r*np.cos(theta)] )
                wq.append(1)
            
        self.xq = xq
        lenwq = len(wq)
        wq = [wq[i]/lenwq for i in range(0, len(wq))]
        self.wq = np.asarray(wq)

    @staticmethod
    def map_from_reference(x0, n, R):
        '''
        Map unit disk in z = 0 to plane to disk of radius R with center at x0.
        '''
        n = n / np.linalg.norm(n)
        def transform(x, x0=x0, n=n, R=R):
            norm = np.dot(x, x)
            # Check assumptions
            assert norm < 1 + 1E-13 and abs(x[2]) < 1E-13
            
            y = x - n*np.dot(x, n)
            y = y / np.sqrt(norm - np.dot(x, n)**2)
            return x0 + R*np.sqrt(norm)*y

        return transform

    def quadrature(self, x0, n):
        '''Quadrature for disk(center x0, normal n, radius x0)'''
        xq, wq = self.xq, self.wq
        
        xq = np.c_[xq, np.zeros_like(wq)]
        
        R1 = self.radius1(x0)
        R2 = self.radius2(x0)
        
        # Circle viewed from reference
        Txq = list(map(AnnularDisk.map_from_reference(x0, n, R2), xq))
        # Scaled weights (R is jac of T, pi is from theta=pi*(-1, 1)
        wq = wq*R2**2
        
        return Quadrature(Txq, wq)

