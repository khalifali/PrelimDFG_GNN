"""Geometric invariants and analytically known collision cases."""
import unittest
import numpy as np
from generate import bpca, bcca, local_filling, inspect, first_contact, rotation

class GeometryTests(unittest.TestCase):
    def test_first_contact(self):
        axis = np.array([1., 0, 0])
        zero = np.zeros((1, 3))
        np.testing.assert_allclose(first_contact(zero, zero, axis, np.zeros(3)), [[2, 0, 0]])
        self.assertIsNone(first_contact(zero, zero, axis, np.array([0., 3, 0])))
        moved = first_contact(np.array([[0.,0,0], [4.,0,0]]), zero, axis, np.zeros(3))
        np.testing.assert_allclose(moved, [[6,0,0]])

    def test_reject_bad_geometry(self):
        for points in [[[0,0,0],[1,0,0]], [[0,0,0],[3,0,0]]]:
            with self.assertRaises(ValueError):
                inspect(np.array(points, dtype=float))

    def test_methods_and_seeds(self):
        for method in [bpca, bcca, local_filling]:
            for n in [2, 17, 50]:
                a = method(n, np.random.default_rng(42))
                b = method(n, np.random.default_rng(42))
                np.testing.assert_array_equal(a, b)
                self.assertEqual(len(a), n)
                self.assertTrue(inspect(a)[0]['connected'])
                np.testing.assert_allclose(a.mean(axis=0), 0, atol=1e-12)

    def test_rotation_invariants(self):
        rng = np.random.default_rng(5)
        q = rotation(rng)
        np.testing.assert_allclose(q @ q.T, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(q), 1)
        points = bcca(31, rng)
        a = inspect(points)[0]
        b = inspect(points @ q.T + 10)[0]
        self.assertEqual(a['contacts'], b['contacts'])
        self.assertAlmostEqual(a['rg_over_radius'], b['rg_over_radius'])

if __name__ == '__main__':
    unittest.main()
