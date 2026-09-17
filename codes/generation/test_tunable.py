"""Verify mass-radius identity, contacts, repeatability and N=500 hierarchy."""
import unittest
import numpy as np
from tunable import generate,rg2
from generate import inspect

class TunableTests(unittest.TestCase):
    def test_design_corners(self):
        for df,kf in ((1.8,1.3),(2.2,1.1),(2.6,.8)):
            for n in (25,50,100,200,500):
                p=generate(n,df,kf,np.random.default_rng(100+n))
                metrics,edges=inspect(p)
                self.assertEqual(len(p),n)
                self.assertTrue(metrics['connected'])
                self.assertLess(metrics['max_overlap_over_radius'],1e-9)
                self.assertAlmostEqual(np.sqrt(rg2(p))/(n/kf)**(1/df),1.,places=10)
                self.assertGreaterEqual(len(edges),n-1)
    def test_repeatability(self):
        a=generate(50,2.2,1.1,np.random.default_rng(7))
        b=generate(50,2.2,1.1,np.random.default_rng(7))
        np.testing.assert_array_equal(a,b)
    def test_invalid_count(self):
        with self.assertRaises(ValueError):generate(51,2.2,1.1,np.random.default_rng(7))

if __name__=='__main__':unittest.main()
