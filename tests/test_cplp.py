import unittest
import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cplp_solver import CPLPSolver

class TestCPLPSolver(unittest.TestCase):
    def test_basic_cplp(self):
        fac_caps = [500, 700, 600]
        fixed_c = [10000, 15000, 12000]
        demands = [200, 300, 150, 400]
        ship_costs = [
            [12, 15, 20, 25],
            [18, 10, 14, 15],
            [15, 14, 12, 18]
        ]
        solver = CPLPSolver(fac_caps, fixed_c, demands, ship_costs)
        res = solver.solve()
        self.assertTrue(res["success"])
        self.assertGreater(res["total_cost"], 0)
        # Check demand satisfaction
        fulfilled_demands = np.sum(res["shipping_matrix"], axis=0)
        np.testing.assert_allclose(fulfilled_demands, demands, rtol=1e-4)

    def test_single_facility_sufficient(self):
        fac_caps = [1000, 100]
        fixed_c = [1000, 50000]
        demands = [100, 200]
        ship_costs = [[5, 5], [10, 10]]
        solver = CPLPSolver(fac_caps, fixed_c, demands, ship_costs)
        res = solver.solve()
        self.assertTrue(res["success"])
        self.assertEqual(res["open_facilities"], [1, 0])

if __name__ == "__main__":
    unittest.main()
