from common import Solver, TSPTWProblem, PermuSolution, Client, Operator, InitOperator
from typing import Callable, List, Tuple, Dict, Literal
from abc import ABC, abstractmethod
import random
import copy
import time
import numpy as np
from sklearn.cluster import KMeans

class RemoveOperator(Operator):
        
    @abstractmethod
    def __call__(self, solution: PermuSolution, remove_cnt: int) -> Tuple[PermuSolution, PermuSolution]:
        pass

class InsertOperator(Operator):
    
    @abstractmethod
    def __call__(self, partial_solution: PermuSolution, removed_solution: PermuSolution,
               insert_idx_selected: int|Literal['all']) -> PermuSolution:
        pass
    
class RandomRemoveOperator(RemoveOperator):
    def __call__(self, solution, remove_cnt):
        if len(solution.route) <= 3:
            return solution, PermuSolution(size=0)
        removed = random.sample(solution.get_main(), k=min(remove_cnt, len(solution.get_main())))
        removed_set = set(removed)
        partial = [c for c in solution.route if c not in removed_set]
        
        partial_sol = PermuSolution(size=len(partial))
        partial_sol.route = partial
        
        removed_sol = PermuSolution(size=len(removed))
        removed_sol.route = removed
        
        return partial_sol, removed_sol
    
class WorstRemoveOperator(RemoveOperator):
    def __call__(self, solution, remove_cnt):
        route = solution.decode()
        curr_time = 0
        scores = []
        for i in range(solution.size - 1):
            cur, nxt = route[i], route[i + 1]
            travel_time = cur.travel_times[nxt.id]
            arrival_time = curr_time + travel_time
            penalty = 0.0
            if arrival_time < nxt.earliness:
                penalty = (nxt.earliness - arrival_time)
                arrival_time = nxt.earliness
            if arrival_time > nxt.tardiness:
                penalty = 3 * (arrival_time - nxt.tardiness)
            
            curr_time = arrival_time + nxt.service_time
            scores.append((nxt, penalty))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        removed = [e[0] for e in scores[:remove_cnt]]
        removed_set = set(removed)
        partial = [c for c in route if c not in removed_set]
        
        removed_sol = PermuSolution(len(removed))
        removed_sol.route = removed
        
        partial_sol = PermuSolution(len(partial))
        partial_sol.route = partial
        
        return partial_sol, removed_sol
    
class ShawRemoveOperator(RemoveOperator):
    def __call__(self, solution, remove_cnt):
        main_route = copy.deepcopy(solution.get_main())
        if len(main_route) <= 0:
            return solution, PermuSolution(size=0)
        
        seed = random.choice(main_route)
        similarities = []
        for client in main_route:
            if client == seed:
                continue
            travel_time = seed.travel_times[client.id] 
            time_diff = (abs(seed.earliness - client.earliness) + abs(seed.tardiness - client.tardiness)) / 2.0
            similarity = travel_time + time_diff
            similarities.append((client, similarity))
        
        similarities.sort(key=lambda x: x[1])
        removed = [e[0] for e in similarities[:remove_cnt]]
        removed_set = set(removed)
        partial = [c for c in solution.route if c not in removed_set] 
            
        removed_sol = PermuSolution(len(removed))
        removed_sol.route = removed
        
        partial_sol = PermuSolution(len(partial))
        partial_sol.route = partial
        
        return partial_sol, removed_sol
    
class TWClusterRemoveOperator(RemoveOperator):
    def __init__(self, prob, problem = None, score = 0, k=10):
        super().__init__(prob, problem, score)
        self.k = k
        
    def __call__(self, solution, remove_cnt):
        main_route = solution.get_main()
        if len(main_route) == 0:
            return solution, PermuSolution(size=0)
        
        # Extract feature
        features = np.array([[c.earliness, c.tardiness, c.tardiness - c.earliness, c.service_time] for c in main_route])
        kmeans = KMeans(n_clusters=self.k, n_init='auto', random_state=42).fit(features)
        labels = kmeans.labels_
        
        # Chọn 1 cluster
        target_cluster = random.choice(range(self.k))
        to_remove = [client for idx, client in enumerate(main_route) if labels[idx] == target_cluster]
        removed = random.sample(to_remove, k=min(remove_cnt, len(to_remove)))
        removed_set = set(removed)
        partial = [c for c in solution.route if c not in removed_set]
        
        removed_sol = PermuSolution(len(removed))
        removed_sol.route = removed
        partial_sol = PermuSolution(len(partial))
        partial_sol.route = partial
        
        return partial_sol, removed_sol
    
class GreedyInsertOperator(InsertOperator):
    def __call__(self, partial_solution, removed_solution, insert_idx_selected):
        route = copy.deepcopy(partial_solution.route)
        
        for unorder_client in removed_solution.route:
            best_pos = None
            best_eval = (float('inf'), float('inf'), float('inf'))
            
            insert_ids = [i for i in range(1, len(route))]
            if insert_idx_selected != 'all':
                insert_ids = random.sample(insert_ids, k=min(insert_idx_selected, len(route)-2))
            
            for i in insert_ids:
                new_route = route[:i] + [unorder_client] + route[i:]
                new_sol = PermuSolution(size=len(new_route))
                new_sol.route = new_route
                
                violation = self.problem.cal_violations(new_sol)
                penalty = self.problem.cal_penalty(new_sol)
                cost = self.problem.cal_cost(new_sol)
                
                new_eval = (violation, penalty, cost)
                
                if new_eval < best_eval:
                    best_eval = new_eval
                    best_pos = i
                    
            if best_pos is not None:
                route.insert(best_pos, unorder_client)
            else:
                random_pos = random.randint(1, len(route))
                route.insert(random_pos, unorder_client)
        
        new_sol = PermuSolution(len(route))
        new_sol.route = route
        
        return new_sol
                    
class RegretInsertOperator(InsertOperator):
    def __call__(self, partial_solution, removed_solution, insert_idx_selected):
        route = copy.deepcopy(partial_solution.route)
        remaining = copy.deepcopy(removed_solution.route)
        
        while remaining:
            regrets = []
            for client in remaining:
                options = []
                insert_ids = [i for i in range(1, len(route))]
                if insert_idx_selected != 'all':
                    insert_ids = random.sample(insert_ids, k=min(insert_idx_selected, len(route)-2))
                
                for i in insert_ids:
                    new_route = route[:i] + [client] + route[i:]
                    new_sol = PermuSolution(size=len(new_route))
                    new_sol.route = new_route
                    
                    violation = self.problem.cal_violations(new_sol)
                    penalty = self.problem.cal_penalty(new_sol)
                    cost = self.problem.cal_cost(new_sol)
                    
                    new_eval = (violation, penalty, cost)
                    
                    options.append((i, new_eval))
                
                if options:
                    options.sort(key=lambda x: x[1])
                    if len(options) == 1:
                        regret_value = 0
                    else:
                        regret_value = options[1][1][1] - options[0][1][1]
                    regrets.append((client, options[0][0], regret_value))
                else:
                    regrets.append((client, None, -1))
                    
            regrets.sort(key=lambda x: x[1], reverse=True)
            best_client, best_pos, _ = regrets[0]
            if best_pos is not None:
                route.insert(best_pos, best_client)
            else:
                route.append(best_client)
            remaining.remove(best_client)
        
        new_sol = PermuSolution(len(route))
        new_sol.route = route
        
        return new_sol
    
class RandomInsertOperator(InsertOperator):
    def __call__(self, partial_solution, removed_solution, insert_idx_selected):
        route = copy.deepcopy(partial_solution.route)
        remaining = copy.deepcopy(removed_solution.route)
        random.shuffle(remaining)
        for unorder_client in remaining:
            candidates = []
            
            insert_ids = [i for i in range(1, len(route))]
            if insert_idx_selected != 'all':
                insert_ids = random.sample(insert_ids, k=min(insert_idx_selected, len(route) - 2))
            
            for i in insert_ids:
                new_route = route[:i] + [unorder_client] + route[i:]
                new_sol = PermuSolution(size=len(new_route))
                new_sol.route = new_route
                
                violation = self.problem.cal_violations(new_sol)
                penalty = self.problem.cal_penalty(new_sol)
                cost = self.problem.cal_cost(new_sol)
                
                new_eval = (violation, penalty, cost)
                
                candidates.append((i, new_eval))
                    
            if candidates:
                candidates.sort(key=lambda x: x[1])
                top_candidates = candidates[:4]
                pos = random.choice(top_candidates)[0]
                route.insert(pos, unorder_client)
            else:
                pos = random.randint(1, len(route))
                route.insert(pos, unorder_client)
        
        new_sol = PermuSolution(len(route))
        new_sol.route = route
        
        return new_sol        
    
class TWClusterInsertOperator(InsertOperator):
        
    def __call__(self, partial_solution, removed_solution, insert_idx_selected):
        route = partial_solution.route.copy()
        to_insert = removed_solution.route.copy()
        
        if len(to_insert) == 0:
            return partial_solution
        
        features = np.array([[c.earliness, c.tardiness, c.tardiness - c.earliness, c.service_time] 
                             for c in route[1:-1]])
        centroid = np.mean(features, axis=0) 
        
        # Tính độ tương đồng
        scored_clients = []
        for client in to_insert:
            vec = np.array([client.earliness, client.tardiness,
                            client.earliness - client.tardiness, client.service_time])
            sim = np.linalg.norm(vec - centroid)
            scored_clients.append((sim, client))
            
        scored_clients.sort()  # Ưu tiên TW gần centroid (tức ít gây ảnh hưởng tiến độ)

        # Chèn từng client vào vị trí tốt nhất
        for _, client in scored_clients:
            best_eval = (float('inf'), float('inf'), float('inf'))
            best_pos = None
            insert_ids = list(range(1, len(route)))
            if insert_idx_selected != 'all':
                insert_ids = random.sample(insert_ids, k=min(insert_idx_selected, len(insert_ids)))
            for i in insert_ids:
                new_route = route[:i] + [client] + route[i:]
                new_sol = PermuSolution(len(new_route))
                new_sol.route = new_route
                ev = (self.problem.cal_violations(new_sol),
                      self.problem.cal_penalty(new_sol),
                      self.problem.cal_cost(new_sol))
                if ev < best_eval:
                    best_eval = ev
                    best_pos = i
            if best_pos is not None:
                route.insert(best_pos, client)
            else:
                route.insert(random.randint(1, len(route)-1), client)

        new_sol = PermuSolution(len(route))
        new_sol.route = route
        return new_sol
                
class ALNSSolver(Solver):
    def __init__(self, problem: TSPTWProblem, 
                 init_opr: InitOperator|None = None,
                 remove_oprs: List[RemoveOperator] = [],
                 insert_oprs: List[InsertOperator] = [],
                 lr: float = 0.1):
        super().__init__(problem)
        self.init_opr = init_opr
        self.remove_oprs = remove_oprs
        self.insert_oprs = insert_oprs
        self.lr = lr
        
        for opr in remove_oprs:
            opr.score = 0
        for opr in insert_oprs:
            opr.score = 0
            
    def add_init_opr(self, init_opr: InitOperator):
        self.init_opr = init_opr
        
    def add_remove_opr(self, remove_opr: RemoveOperator):
        self.remove_oprs.append(remove_opr)
    
    def add_insert_opr(self, insert_opr: InsertOperator):
        self.insert_oprs.append(insert_opr)
    
    def update_weight(self, oprs: List[Operator]):
        total_scores = sum(opr.score for opr in oprs)
        if total_scores == 0:
            total_scores = 1
        for opr in oprs:
            new_weight = (1 - self.lr) * opr.prob + self.lr * (opr.score / total_scores)
            opr.prob = new_weight
        
    def solve(self, debug: bool=False, num_iters: int = 1000, max_solve_time: int=3600, 
              remove_fraction: int|float = 0.2,
              insert_idx_selected: int|Literal['all'] = 'all',
              update_weight_freq: float = 0.1):
        start = time.time()
        init_sol = self.init_opr()
        self.update_best(init_sol)
        
        update_weight_cycle = int(max(1, num_iters * update_weight_freq))
        
        for iter in range(1, num_iters+1):
            self.this_iter = iter
            # Check time
            if time.time() - start > max_solve_time:
                return self.finish(start)
            
            # Remove phase
            remove_opr: RemoveOperator = self._choose_opr(self.remove_oprs)
            if isinstance(remove_fraction, float):
                remove_cnt = int(max(1, remove_fraction * len(self.problem.clients)))
            else:
                remove_cnt = remove_fraction
            partial_sol, remove_sol = remove_opr(self.best_solution, remove_cnt)
            
            if time.time() - start > max_solve_time:
                return self.finish(start)
            
            # Insert phase
            insert_opr: InsertOperator = self._choose_opr(self.insert_oprs)
            new_sol = insert_opr(partial_sol, remove_sol, insert_idx_selected)
            # Update cost & violation
            reward = self.update_best(new_sol)
            
            remove_opr.score += reward
            insert_opr.score += reward
            
            if iter % update_weight_cycle == 0:
                self.update_weight(self.remove_oprs)
                self.update_weight(self.insert_oprs)
            
            self._print_with_debug(f'Iter {iter}: Best vio: {self.best_violations}, Best penalty: {self.best_penalty}, Best cost: {self.best_cost}', debug)
        
        return self.finish(start)