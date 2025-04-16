import ga
from common import read_input_file, RandomInitOperator, Solver
import random
random.seed(42)

problem = read_input_file(f'tests/test1/input.in')

init_random = RandomInitOperator(1, problem)

crossover_opr = ga.CrossoverOperator(1, problem)
mutation_opr = ga.MutationOperator(1, problem)

solver = ga.GASolver(problem)
solver.set_init_opr(init_random)
solver.set_crossover_opr(crossover_opr)
solver.set_mutation_opr(mutation_opr)

status = solver.solve(debug=True, num_gen=500, pop_size=100, pc=0.6, pm=0.15, elite_ratio=0.1)


print(solver.best_violations)
print(solver.best_cost)
print(solver.best_solution)
