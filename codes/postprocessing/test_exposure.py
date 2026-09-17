import unittest
import numpy as np
from exposure import geometric_exposure

class ExposureTests(unittest.TestCase):
    def test_single(self):
        np.testing.assert_array_equal(geometric_exposure([[0,0,0]],[1]),[1])
    def test_dimer_analytic(self):
        # Two tangent equal spheres: occluded spherical cap has half angle 30 degrees.
        expected=(1+np.sqrt(3)/2)/2
        got=geometric_exposure([[0,0,0],[2,0,0]],[1,1],4096)
        np.testing.assert_allclose(got,expected,atol=.002)
    def test_scale_and_particle_order(self):
        xyz=np.array([[0.,0,0],[2,0,0],[0,2,0]])
        a=geometric_exposure(xyz,np.ones(3))
        b=geometric_exposure((xyz+10)*1e-6,np.ones(3)*1e-6)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[::-1],geometric_exposure(xyz[::-1],np.ones(3)))

if __name__=='__main__':unittest.main()
