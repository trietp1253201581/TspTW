from common import read_console, TimeWindowConstraint, MustFollowConstraint, PermuSolution, Solver, read_input_file
import ls
from common import HNNInitOperator, HybridInitOperator

for i in range(1, 12): # Iter for full test
    problem = read_input_file(f'tests/test{i}/input.in')
    print(f'Test {i}')
    problem.add_constraint(TimeWindowConstraint())
    # problem.add_constraint(MustFollowConstraint())

    init_hnn = HybridInitOperator(prob=1, problem=problem, hn_ratio=0.12)
    
    random_swap = ls.RandomSwapOperator(0.3, problem)
    best_swap = ls.FirstBestSwapOperator(0.2, problem)
    best_relocate = ls.FirstRelocateOperator(0.15, problem)
    delay_insert = ls.DelayMovingOperator(0.35, problem)

    solver = ls.SimulatedAnnealingSolver(problem, T0=1000, alpha=0.98)
    solver.clear_opr()
    solver.add_init_opr(init_hnn)

    solver.add_moving_opr(random_swap)
    solver.add_moving_opr(best_swap)
    solver.add_moving_opr(best_relocate)
    solver.add_moving_opr(delay_insert)
    
    import random
    random.seed(42)
    status = solver.solve(debug=False, max_iter=500, max_solve_time=100)
    
    results = []

    results.append(f'Status: {status}')
    results.append(f'Best sol: {str(solver.best_solution)}')
    results.append(f'Violations: {str(solver.best_violations)}')
    results.append(f'Cost: {solver.best_cost:.2f}')
    results.append(f'Num Iter: {solver.this_iter}')
    results.append(f'Solve Time: {solver.solve_time:.2f} secs')

    for result in results:
        print(result)
        
    with open(f'tests/test{i}/sol_sa.out', 'w') as f:
        for result in results:
            f.write(result)
            f.write('\n')
            
    


