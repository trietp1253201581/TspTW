from common import read_console, TimeWindowConstraint, MustFollowConstraint, PermuSolution, Solver
import ls
from common import HNNInitOperator, RandomInitOperator, HybridInitOperator

problem = read_console()

problem.add_constraint(TimeWindowConstraint())
# problem.add_constraint(MustFollowConstraint())

init_hnn = HNNInitOperator(prob=1, problem=problem)
init_random = RandomInitOperator(prob=1, problem=problem)
hybrid = HybridInitOperator(1, problem, 0.12)

random_swap_opr = ls.RandomSwapOperator(0.3, problem)
best_swap_opr = ls.FirstBestSwapOperator(0.2, problem)
first_replace_opr = ls.FirstRelocateOperator(0.1, problem)
delay_opr = ls.DelayMovingOperator(0.4, problem)

solver = ls.LSSolver(problem, init_opr=hybrid)
solver.add_moving_opr(random_swap_opr)
solver.add_moving_opr(best_swap_opr)
solver.add_moving_opr(first_replace_opr)
solver.add_moving_opr(delay_opr)

import random
random.seed(42)
status = solver.solve(debug=True, max_iter=700)

print(status)

print(solver.best_solution)
print(solver.best_violations)
print(solver.best_penalty)
print(solver.best_cost)
print(solver.solve_time)