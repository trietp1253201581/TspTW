from common import PermuSolution, TSPTWProblem, InitOperator, Operator, Solver
from typing import List, Tuple
import random
import copy
import time

class Individual:
    def __init__(self):
        self.chromosome: PermuSolution = None
        self.fitness = -1e8
        
    def generate(self, init_func: InitOperator):
        self.chromosome = init_func()
        
    def decode(self):
        return self.chromosome
    
    def cal_fitness(self, problem: TSPTWProblem):
        sol = self.decode()
        p = problem.cal_penalty(sol)
        c = problem.cal_cost(sol)
        
        self.fitness = - (p * 10 + c)
        
class Population:
    def __init__(self, size: int, problem: TSPTWProblem):
        self.size = size
        self.problem = problem
        self.inds: List[Individual] = []
    
    def generate(self, init_func: InitOperator):
        self.inds.clear()
        for _ in range(self.size):
            new_ind = Individual()
            new_ind.generate(init_func)
            new_ind.cal_fitness(self.problem)
            self.inds.append(new_ind)
            
class CrossoverOperator(Operator):
    def __call__(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        n = p1.chromosome.size - 2
        pos1, pos2 = sorted(random.sample(range(n), 2))
        route1, route2 = p1.chromosome.get_main(), p2.chromosome.get_main()
        croute1 = [None] * n
        croute2 = [None] * n
        
        croute1[pos1:pos2] = route1[pos1:pos2]
        croute2[pos1:pos2] = route2[pos1:pos2]
        
        def fill_child(c, p):
            idx = pos2
            for client in p:
                if client not in c:
                    if idx >= n:
                        idx = 0
                    c[idx] = client
                    idx += 1
                    
        fill_child(croute1, route2)
        fill_child(croute2, route1)
        
        sol1, sol2 = PermuSolution(n+2), PermuSolution(n+2)
        sol1.route = [self.problem.start] + croute1 + [self.problem.start]
        sol2.route = [self.problem.start] + croute2 + [self.problem.start]
        
        c1, c2 = Individual(), Individual()
        c1.chromosome = sol1
        c2.chromosome = sol2

        return c1, c2
    
class MutationOperator(Operator):
    def __call__(self, p: Individual) -> Individual:
        c = Individual()
        c.chromosome = copy.deepcopy(p.chromosome)
        pos1, pos2 = sorted(random.sample(range(1, c.chromosome.size - 1), k=2))
        c.chromosome.route[pos1], c.chromosome.route[pos2] = c.chromosome.route[pos2], c.chromosome.route[pos1]
        return c

class GASolver(Solver):
    def __init__(self, problem, init_opr: InitOperator|None=None,
                 crossover_opr: CrossoverOperator|None=None,
                 mutation_opr: MutationOperator|None=None):
        super().__init__(problem)
        self.init_opr = init_opr
        self.crossover_opr = crossover_opr
        self.mutation_opr = mutation_opr
        
    def set_init_opr(self, init_opr: InitOperator):
        self.init_opr = init_opr
    
    def set_crossover_opr(self, crossover_opr: CrossoverOperator):
        self.crossover_opr = crossover_opr
        
    def set_mutation_opr(self, mutation_opr: MutationOperator):
        self.mutation_opr = mutation_opr
        
    def solve(self, debug = False, num_gen: int = 100,
              pop_size: int = 30,
              pc: float=0.8, pm: float=0.1, elite_ratio: float=0.1,
              max_solve_time: int = 3600):
        start = time.time()
        pop = Population(pop_size, self.problem)
        pop.generate(self.init_opr)
        elite_count = int(elite_ratio * pop_size)
        
        for gen in range(1, num_gen + 1):
            self.this_iter = gen
            if time.time() - start > max_solve_time:
                return self.finish(start)
            
            offs: List[Individual] = []
            while len(offs) < pop_size:
                p1, p2 = random.sample(pop.inds, 2)
                if random.random() < pc:
                    off1, off2 = self.crossover_opr(p1, p2)
                    if random.random() < pm:
                        off1 = self.mutation_opr(off1)
                    if random.random() < pm:
                        off2 = self.mutation_opr(off2)
                        
                    off1.cal_fitness(self.problem)
                    off2.cal_fitness(self.problem)
                    
                    offs.append(off1)
                    offs.append(off2)
                    
            olds = sorted(pop.inds, key=lambda x: x.fitness, reverse=True)
            old_best = olds[:elite_count]
            offs = offs[:pop_size]
            remains = olds[elite_count:] + offs
            
            selected = random.sample(remains, pop_size-elite_count)
            
            pop.inds = old_best + selected
            pop.inds.sort(reverse=True, key=lambda x: x.fitness)
            
            self._print_with_debug(f"Gen {gen}, best fitness: {pop.inds[0].fitness}", debug)
            
        self.update_best(pop.inds[0].decode())
        return self.finish(start)