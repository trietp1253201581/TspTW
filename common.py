from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
import random
import time

class Client:
    def __init__(self, id: int|str, time_window: Tuple[int, int, int]):
        self.id = int(id)
        self.earliness = time_window[0]
        self.tardiness = time_window[1]
        self.service_time = time_window[2]
        self.travel_times: Dict[int|str, int] = {} 
        
    def add_travel_time(self, other_id: int|str, travel_time: int):
        self.travel_times[other_id] = travel_time
        
    def __eq__(self, value):
        if not isinstance(value, Client):
            return False
        return self.id == value.id
    
    def __hash__(self):
        return hash(self.id)
    
    def __str__(self):
        travel_str = ", ".join(f'{c}: {t}' for c, t in self.travel_times.items())
        return f'Client({self.id}, e={self.earliness}, t={self.tardiness}, d={self.service_time}, times=[{travel_str}])'
    
class PermuSolution:
    def __init__(self, size: int):
        self.size = size
        self.route: List[Client] = []
        
    def decode(self):
        if len(self.route) != self.size:
            return None
        if self.route[0] != self.route[-1]:
            return None
        return self.route
    
    def get_main(self):
        return self.route[1:-1]
    
    def __str__(self):
        return f'[{", ".join(str(c.id) for c in self.route)}]'
        
class Constraint(ABC):
    @abstractmethod
    def get_violation(self, solution: PermuSolution) -> int:
        pass

    def get_penalty(self, solution: PermuSolution) -> float:
        return self.get_violation(solution)
    
    def check(self, solution: PermuSolution) -> bool:
        return self.get_violation(solution) == 0
    
class TimeWindowConstraint(Constraint):
    def get_violation(self, solution):
        route = solution.decode()
        curr_time = 0
        violations = 0
        for i in range(solution.size - 1):
            cur, nxt = route[i], route[i + 1]
            travel_time = cur.travel_times[nxt.id]
            arrival_time = curr_time + travel_time
            
            if arrival_time < nxt.earliness:
                arrival_time = nxt.earliness
            if arrival_time > nxt.tardiness:
                violations += 1
            
            curr_time = arrival_time + nxt.service_time
        return violations
            
    def get_penalty(self, solution):
        route = solution.decode()
        curr_time = 0
        penalty = 0.0
        for i in range(solution.size - 1):
            cur, nxt = route[i], route[i + 1]
            travel_time = cur.travel_times[nxt.id]
            arrival_time = curr_time + travel_time
            
            if arrival_time < nxt.earliness:
                penalty += (nxt.earliness - arrival_time)
                arrival_time = nxt.earliness
            if arrival_time > nxt.tardiness:
                penalty += 3 * (arrival_time - nxt.tardiness)
            
            curr_time = arrival_time + nxt.service_time
        
        return penalty

class MustFollowConstraint(Constraint):
    def get_violation(self, solution):
        violations = 0
        route = solution.decode()
        for i in range(1, solution.size - 1):
            for j in range(1, solution.size - 1):
                if i > j and route[i].tardiness + route[i].travel_times[route[j].id] < route[j].earliness:
                    violations += 1
                    
        return violations

class TSPTWProblem:
    def __init__(self, clients: List[Client], start_at: Client):
        self.start = start_at
        self.clients = clients
        self.constraints: List[Constraint] = []
        
    def add_constraint(self, constraint: Constraint):
        self.constraints.append(constraint)
        
    def cal_violations(self, solution: PermuSolution):
        total_violations = 0
        for constraint in self.constraints:
            total_violations += constraint.get_violation(solution)
            
        return total_violations
    
    def cal_penalty(self, solution: PermuSolution):
        total_penalties = 0
        for constraint in self.constraints:
            total_penalties += constraint.get_penalty(solution)
        return total_penalties

    def check(self, solution: PermuSolution):
        return self.cal_violations(solution) == 0
    
    def cal_cost(self, solution: PermuSolution):
        route = solution.decode()
        total_cost = 0.0
        for i in range(solution.size - 1):
            cur, nxt = route[i], route[i + 1]
            travel_time = cur.travel_times[nxt.id]
            total_cost += travel_time
            
        return total_cost
            
MAX_TARDINESS: int = 1e8            
            
def read_console() -> TSPTWProblem:
    n = int(input())
    clients: List[Client] = []
    clients.append(Client(0, time_window=(0, MAX_TARDINESS, 0)))
    
    for i in range(1, n + 1):
        parts = input().strip().split()
        e, l, d = map(int, parts[:3])
        clients.append(Client(id=i, time_window=(e,l,d)))
    
    for i in range(n + 1):
        row = list(map(int, input().strip().split()))
        for j in range(n + 1):
            if i != j:
                clients[i].add_travel_time(clients[j].id, row[j])
    
    problem = TSPTWProblem(clients=clients, start_at=clients[0])
    return problem

def read_input_file(input_file: str) -> TSPTWProblem:
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    n = int(lines[0].strip())
    clients: List[Client] = []
    clients.append(Client(0, time_window=(0, MAX_TARDINESS, 0)))
    
    for i in range(1, n + 1):
        parts = lines[i].strip().split()
        e, l, d = map(int, parts[:3])
        clients.append(Client(id=i, time_window=(e,l,d)))
    
    for i in range(n + 1):
        row = list(map(int, lines[n + 1 + i].strip().split()))
        for j in range(n + 1):
            if i != j:
                clients[i].add_travel_time(clients[j].id, row[j])
    
    problem = TSPTWProblem(clients=clients, start_at=clients[0])
    return problem
            
    
class Solver(ABC):
    class Status:
        FEASIBLE = "FEASIBLE"
        INFEASIBLE = "INFEASIBLE"
    def __init__(self, problem: TSPTWProblem):
        self.problem = problem
        self.best_solution: PermuSolution = None
        self.best_violations = 1e8
        self.best_cost = 1e8
        self.best_penalty = 1e8
        self.solve_time = 0
        self.this_iter = 0
    
    def _choose_opr(self, oprs: List['Operator']):
        weights = [opr.prob for opr in oprs]
        return random.choices(oprs, weights=weights, k=1)[0]
        
    def _print_with_debug(self, msg: str, debug: bool=False):
        if debug:
            print(msg)
            
    def update_best(self, new_sol: PermuSolution):
        new_violations = self.problem.cal_violations(new_sol)
        new_penalty = self.problem.cal_penalty(new_sol)
        new_cost = self.problem.cal_cost(new_sol)
        
        if new_violations > self.best_violations:
            return 0.0
        if new_violations == self.best_violations:
            if new_penalty > self.best_penalty:
                return 0.0
            if new_penalty == self.best_penalty:
                if new_cost >= self.best_cost:
                    return 0.0
                
        reward = 0.0
        
        reward += max(0, (1 - new_violations/self.best_violations) if self.best_violations > 0 else 0)
        reward += max(0, (1 - new_penalty/self.best_penalty) if self.best_penalty > 0 else 0)
        reward += max(0, (1 - new_cost/self.best_cost) if self.best_cost > 0 else 0)      
        
        self.best_solution = new_sol
        self.best_violations = new_violations
        self.best_penalty = new_penalty
        self.best_cost = new_cost
        
        return reward
    
    def update_sol_time(self, start: float):
        self.solve_time = time.time() - start
        
    def finish(self, start: int|float):
        self.update_sol_time(start)
        return Solver.Status.FEASIBLE if self.best_violations == 0 else Solver.Status.INFEASIBLE
        
        
        
    @abstractmethod
    def solve(self, debug: bool=False, **kwargs) -> Status:
        pass
    
class Operator(ABC):
    def __init__(self, prob: float, problem: TSPTWProblem|None = None, score: float=0.0):
        self.prob = prob
        self.problem = problem
        self.score = score
        
class InitOperator(Operator):
    
    @abstractmethod
    def init(self) -> PermuSolution:
        pass
    
class RandomInitOperator(InitOperator):
    def init(self):
        remains = [client for client in self.problem.clients if not client == self.problem.start]
        random.shuffle(remains)
        
        init_sol = PermuSolution(len(remains) + 2)
        init_sol.route = [self.problem.start] + remains + [self.problem.start]
        return init_sol
class HNNInitOperator(InitOperator):
    
    def init(self):
        route: List[Client] = []
        curr = self.problem.start
        route.append(curr)
        remains = [client for client in self.problem.clients if not client == self.problem.start]
        curr_time = curr.earliness
        
        while remains:
            feasible = []
            infeasible = []
            
            for nxt in remains:
                travel_time = curr.travel_times[nxt.id]
                arr_time = curr_time + travel_time
                
                penalty = 0
                if arr_time < nxt.earliness:
                    penalty = nxt.earliness - arr_time
                elif arr_time > nxt.tardiness:
                    penalty = 3 * (arr_time - nxt.tardiness)
                    
                if penalty == 0:
                    feasible.append((nxt, travel_time))
                else:
                    infeasible.append((nxt, penalty))
            
            if feasible:
                nxt = min(feasible, key=lambda x: x[1])[0]
            else:
                nxt = min(infeasible, key=lambda x: x[1])[0]
                
            route.append(nxt)
            
            travel_time = curr.travel_times[nxt.id]
            arr_time = curr_time + travel_time
            arr_time = max(arr_time, nxt.earliness)
            curr_time = arr_time + nxt.service_time
            curr = nxt
            remains.remove(curr)
            
        route.append(self.problem.start)
        sol = PermuSolution(size=len(route))
        sol.route = route
        return sol
    
class HybridInitOperator(InitOperator):
    def __init__(self, prob=1.0, problem=None, hn_ratio=0.2):
        super().__init__(prob, problem)
        self.hn_ratio = hn_ratio  # tỉ lệ áp dụng HNN
    
    def init(self):
        clients = [c for c in self.problem.clients if c != self.problem.start]
        random.shuffle(clients)
        num_hnn = int(len(clients) * self.hn_ratio)
        
        route = [self.problem.start]
        curr = self.problem.start
        curr_time = curr.earliness
        hnn_part = []
        remains = clients.copy()

        # Áp dụng HNN cho 20% đầu
        for _ in range(min(num_hnn, len(remains))):
            best = None
            best_eval = float('inf')
            for nxt in remains:
                t = curr.travel_times[nxt.id]
                arr_time = curr_time + t
                penalty = max(0, nxt.earliness - arr_time) + 3 * max(0, arr_time - nxt.tardiness)
                if penalty < best_eval:
                    best_eval = penalty
                    best = nxt
            if best:
                hnn_part.append(best)
                curr_time += curr.travel_times[best.id]
                curr_time = max(curr_time, best.earliness)
                curr_time += best.service_time
                curr = best
                remains.remove(best)
        
        # Random phần còn lại
        random.shuffle(remains)
        full_route = route + hnn_part + remains + [self.problem.start]
        sol = PermuSolution(size=len(full_route))
        sol.route = full_route
        return sol
