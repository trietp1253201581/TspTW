from common import PermuSolution, Solver, Operator, InitOperator
from abc import abstractmethod
import random
import copy
from typing import List, Tuple
import time
import math

class MovingOperator(Operator):
    @abstractmethod
    def __call__(self, solution: PermuSolution) -> PermuSolution:
        pass
    
class RandomSwapOperator(MovingOperator):
    def __call__(self, solution):
        route = solution.get_main()
        n = len(route)
        i, j = random.sample(range(0, n), 2)
        new_route = route.copy()
        new_route[i], new_route[j] = new_route[j], new_route[i]
        full = [self.problem.start] + new_route + [self.problem.start]
        new_sol = PermuSolution(len(full))
        new_sol.route = full
        return new_sol
    
class FirstBestSwapOperator(MovingOperator):
    def __call__(self, solution: PermuSolution) -> PermuSolution:
        best = solution
        best_eval = (self.problem.cal_violations(solution),
                     self.problem.cal_penalty(solution),
                     self.problem.cal_cost(solution))
        route = solution.get_main()
        n = len(route)
        # thử mọi cặp i<j
        for i in range(n):
            for j in range(i+1, n):
                new_route = route.copy()
                new_route[i], new_route[j] = new_route[j], new_route[i]
                full = [self.problem.start] + new_route + [self.problem.start]
                cand = PermuSolution(len(full))
                cand.route = full
                ev = (self.problem.cal_violations(cand),
                      self.problem.cal_penalty(cand),
                      self.problem.cal_cost(cand))
                if ev < best_eval:
                    return cand  # first improvement
        return best
    
class FirstRelocateOperator(MovingOperator):
    def __call__(self, solution: PermuSolution) -> PermuSolution:
        best = solution
        best_eval = (self.problem.cal_violations(solution),
                     self.problem.cal_penalty(solution),
                     self.problem.cal_cost(solution))
        route = solution.get_main()
        n = len(route)
        # thử di dời mỗi vị trí i đến vị trí j
        for i in range(n):
            client = route[i]
            without = route[:i] + route[i+1:]
            for j in range(n):
                new_route = without[:j] + [client] + without[j:]
                full = [self.problem.start] + new_route + [self.problem.start]
                cand = PermuSolution(len(full))
                cand.route = full
                ev = (self.problem.cal_violations(cand),
                      self.problem.cal_penalty(cand),
                      self.problem.cal_cost(cand))
                if ev < best_eval:
                    return cand
        return best
    
class DelayMovingOperator(MovingOperator):
    def __call__(self, solution):
        # tìm khách hàng vi phạm nhiều nhất (hoặc có penalty cao nhất)
        route = solution.get_main()
        penalties = []
        curr_time = self.problem.start.earliness
        full = [self.problem.start] + route + [self.problem.start]
        # tính arrival và penalty cục bộ
        times = [0]*(len(full))
        times[0] = curr_time
        for idx in range(len(full)-1):
            c, nxt = full[idx], full[idx+1]
            t = c.travel_times[nxt.id]
            arr = times[idx] + t
            e,l,d = nxt.earliness,nxt.tardiness,nxt.service_time
            pen = max(0, e-arr) + 3*max(0, arr-l)
            penalties.append((idx+1, pen))
            arr = max(arr, e)
            times[idx+1] = arr + d
        # chọn vị trí có penalty cao nhất
        idx, _ = max(penalties, key=lambda x: x[1])
        client = full[idx]
        # loại bỏ
        new_main = route.copy()
        new_main.remove(client)
        # chèn lại vào vị trí tốt nhất
        best = solution
        best_eval = (self.problem.cal_violations(solution),
                     self.problem.cal_penalty(solution),
                     self.problem.cal_cost(solution))
        for j in range(len(new_main)+1):
            cand_route = new_main[:j] + [client] + new_main[j:]
            full_c = [self.problem.start] + cand_route + [self.problem.start]
            cand = PermuSolution(len(full_c))
            cand.route = full_c
            ev = (self.problem.cal_violations(cand),
                  self.problem.cal_penalty(cand),
                  self.problem.cal_cost(cand))
            if ev < best_eval:
                return cand
        return best
    
class LSSolver(Solver):
    def __init__(self, problem,
                 init_opr: InitOperator|None=None,
                 moving_oprs: List[MovingOperator]=[]):
        super().__init__(problem)
        self.init_opr = init_opr
        self.moving_oprs = moving_oprs
        
    def add_moving_opr(self, moving_opr: MovingOperator):
        self.moving_oprs.append(moving_opr)
        
    def add_init_opr(self, init_opr: InitOperator):
        self.init_opr = init_opr
        
    def clear_opr(self):
        self.init_opr = None
        self.moving_oprs.clear()
        
    @abstractmethod
    def solve(self, debug = False, **kwargs):
        pass
    
class HillClimbingSolver(LSSolver):
    def __init__(self, problem,
                 init_opr: InitOperator|None=None,
                 moving_oprs: List[MovingOperator]=[]):
        super().__init__(problem, init_opr, moving_oprs)

    def solve(self, debug = False, max_iter: int = 100, max_solve_time: int = 3600):
        # Init
        start = time.time()
        sol = self.init_opr()
        self.update_best(sol)
    
        it = 0
        while it < max_iter:
            it += 1
            self.this_iter = it
            if time.time() - start > max_solve_time:
                return self.finish(start)
            moving_opr: MovingOperator = self._choose_opr(self.moving_oprs)
            cand = moving_opr(sol)
            self.update_best(cand)
            self._print_with_debug(f'Iter {it}: Best vio: {self.best_violations}, Best penalty: {self.best_penalty}, Best cost: {self.best_cost}, use: {moving_opr}', debug)
            
        return self.finish(start)
    
class SimulatedAnnealingSolver(LSSolver):
    def __init__(self,
                 problem,
                 init_opr: InitOperator|None=None,
                 moving_oprs: List[Operator]=[],
                 T0: float = 1000.0,
                 alpha: float = 0.995):
        super().__init__(problem, init_opr, moving_oprs)
        self.T0 = T0
        self.alpha = alpha

    def solve(self,
              debug: bool = False,
              max_iter: int = 10000,
              max_solve_time: int = 3600) -> Solver.Status:
        start = time.time()

        # 1) Khởi tạo
        curr = self.init_opr()
        # compute penalty, cost

        # đặt best = curr
        self.update_best(curr)
        p, c = self.best_penalty, self.best_cost
        curr_energy = p + c

        T = self.T0

        for it in range(1, max_iter+1):
            self.this_iter = it
            # 2) chọn ngẫu nhiên một move operator
            if time.time() - start > max_solve_time:
                return self.finish(start)
            moving_opr: MovingOperator = self._choose_opr(self.moving_oprs)
            cand = moving_opr(curr)
            # đánh giá
            p2, c2 = self.problem.cal_penalty(cand), self.problem.cal_cost(cand)
            cand_energy = p2 + c2
            delta = cand_energy - curr_energy

            # 3) chấp nhận theo Simulated Annealing
            if delta <= 0 or random.random() < math.exp(-delta / T):
                curr = cand
                curr_energy = cand_energy
                # nếu tốt hơn best, cập nhật
                self.update_best(cand)

            # 4) làm lạnh
            T *= self.alpha

            # 5) debug
            if debug and it % (max_iter//10 or 1) == 0:
                print(f"[SA] iter={it:5d} T={T:.3f}  "
                      f"best=(vio={self.best_violations}, pen={self.best_penalty:.1f}, cost={self.best_cost:.1f})")

        # lưu thời gian
        return self.finish(start)
    
