from common import read_console, TimeWindowConstraint, MustFollowConstraint, PermuSolution, Solver, read_input_file
import alns
from common import HNNInitOperator, HybridInitOperator


for i in range(1, 12): # Iter for full test
    problem = read_input_file(f'tests/test{5}/input.in')

    problem.add_constraint(TimeWindowConstraint())
    # problem.add_constraint(MustFollowConstraint())

    init_hnn = HybridInitOperator(prob=1, problem=problem, hn_ratio=0.12)

    remove_random = alns.RandomRemoveOperator(prob=0.3, problem=problem)
    remove_worst = alns.WorstRemoveOperator(prob=0.1, problem=problem)
    remove_shaw = alns.ShawRemoveOperator(prob=0.6, problem=problem)

    insert_greedy = alns.GreedyInsertOperator(prob=0.3, problem=problem)
    insert_regret = alns.RegretInsertOperator(prob=0.7, problem=problem)

    solver = alns.ALNSSolver(problem, lr=0.1)



    solver.add_init_opr(init_hnn)
    solver.add_remove_opr(remove_random)
    solver.add_remove_opr(remove_worst)
    solver.add_remove_opr(remove_shaw)
    solver.add_insert_opr(insert_greedy)
    solver.add_insert_opr(insert_regret)
    
    import random
    random.seed(42)
    status = solver.solve(debug=True, num_iters=100, max_solve_time=500, remove_fraction=0.2,
                        insert_idx_selected=30, update_weight_freq=0.2)
    
    results = []

    results.append(f'Status: {status}')
    results.append(f'Best sol: {str(solver.best_solution)}')
    results.append(f'Violations: {str(solver.best_violations)}')
    results.append(f'Cost: {solver.best_cost:.2f}')
    results.append(f'Num Iter: {solver.this_iter}')
    results.append(f'Solve Time: {solver.solve_time:.2f} secs')

    for result in results:
        print(result)
        
    with open(f'tests/test{i}/sol_alns.out', 'w') as f:
        for result in results:
            f.write(result)
            f.write('\n')
            
    break


