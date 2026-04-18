"""
Capacitated Plant Location Problem (CPLP) Solver
Optimization of Manufacturing Facility Locations and Production Cost Minimization

Formulation:
  Minimizes Total Cost = Fixed Facility Opening Costs + Variable Production & Shipping Costs
  Subject to:
    1. Demand Satisfaction: Every customer's demand must be fully satisfied.
    2. Capacity Constraints: Total production/shipping from facility i cannot exceed its capacity.
    3. Facility Linkage: Shipping from facility i to customer j is only allowed if facility i is open.
    4. Integrality: Facility opening variables y_i in {0, 1}.
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

class CPLPSolver:
    def __init__(self, facility_capacities, fixed_costs, customer_demands, shipping_costs):
        """
        Parameters:
        - facility_capacities: list/array of shape (I,) with max capacity for each facility
        - fixed_costs: list/array of shape (I,) with fixed cost to open facility
        - customer_demands: list/array of shape (J,) with demand for each customer location
        - shipping_costs: array of shape (I, J) with unit shipping cost from facility i to customer j
        """
        self.I = len(facility_capacities)  # Number of potential facilities
        self.J = len(customer_demands)      # Number of customer locations
        self.capacities = np.array(facility_capacities, dtype=float)
        self.fixed_costs = np.array(fixed_costs, dtype=float)
        self.demands = np.array(customer_demands, dtype=float)
        self.shipping_costs = np.array(shipping_costs, dtype=float)

    def solve(self):
        """
        Solves CPLP using scipy.optimize.milp
        Decision variables:
          - y_i in {0, 1}: Binary indicator if facility i is opened (I variables)
          - x_{i,j} >= 0: Flow fraction/quantity shipped from facility i to customer j (I * J variables)
        
        Total variables N = I + I * J
        Index mapping:
          y_i -> index i
          x_{i,j} -> index I + i * J + j
        """
        num_vars = self.I + self.I * self.J
        
        # Objective vector c:
        # Cost = sum_i (fixed_costs[i] * y_i) + sum_{i,j} (shipping_costs[i,j] * x_{i,j})
        c = np.zeros(num_vars)
        c[:self.I] = self.fixed_costs
        for i in range(self.I):
            for j in range(self.J):
                c[self.I + i * self.J + j] = self.shipping_costs[i, j]

        # Integrality: y_i are binary (1), x_{i,j} are continuous (0)
        integrality = np.zeros(num_vars)
        integrality[:self.I] = 1  # Integer/binary constraint for y_i

        # Variable Bounds:
        # y_i in [0, 1]
        # x_{i,j} in [0, inf]
        lb = np.zeros(num_vars)
        ub = np.full(num_vars, np.inf)
        ub[:self.I] = 1.0
        bounds = Bounds(lb, ub)

        constraints = []

        # Constraint 1: Demand satisfaction for each customer j
        # sum_i x_{i,j} == demands[j]  (J constraints)
        A_demand = np.zeros((self.J, num_vars))
        for j in range(self.J):
            for i in range(self.I):
                A_demand[j, self.I + i * self.J + j] = 1.0
        lc_demand = LinearConstraint(A_demand, self.demands, self.demands)
        constraints.append(lc_demand)

        # Constraint 2: Capacity limit for each facility i
        # sum_j x_{i,j} <= capacities[i] * y_i  =>  sum_j x_{i,j} - capacities[i] * y_i <= 0
        A_cap = np.zeros((self.I, num_vars))
        for i in range(self.I):
            A_cap[i, i] = -self.capacities[i]
            for j in range(self.J):
                A_cap[i, self.I + i * self.J + j] = 1.0
        lc_cap = LinearConstraint(A_cap, -np.inf, np.zeros(self.I))
        constraints.append(lc_cap)

        # Solve MILP
        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)

        if not res.success:
            return {"success": False, "status": res.status, "message": res.message}

        sol = res.x
        open_facilities = sol[:self.I]
        shipping_matrix = sol[self.I:].reshape((self.I, self.J))

        # Round open_facilities binary indicators
        open_facilities_binary = (open_facilities > 0.5).astype(int)
        total_cost = res.fun
        fixed_cost_total = np.sum(open_facilities_binary * self.fixed_costs)
        variable_cost_total = total_cost - fixed_cost_total

        return {
            "success": True,
            "total_cost": total_cost,
            "fixed_cost": fixed_cost_total,
            "variable_cost": variable_cost_total,
            "open_facilities": open_facilities_binary.tolist(),
            "shipping_matrix": shipping_matrix,
            "capacity_utilization": np.sum(shipping_matrix, axis=1) / (self.capacities + 1e-9)
        }

if __name__ == "__main__":
    # Example problem instance: 3 potential plants, 4 demand markets
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
    print("=== CPLP Optimization Results ===")
    print("Success:", res["success"])
    print(f"Total Cost: ${res['total_cost']:,.2f}")
    print(f"Fixed Facility Cost: ${res['fixed_cost']:,.2f}")
    print(f"Variable Shipping Cost: ${res['variable_cost']:,.2f}")
    print("Open Facilities (0/1):", res["open_facilities"])
    print("Shipping Matrix:\n", res["shipping_matrix"])
