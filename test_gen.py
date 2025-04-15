import random
import os

def generate_distance_matrix(n, max_dist=100):
    """Sinh ma trận khoảng cách đối xứng kích thước (n+1)x(n+1)."""
    mat = [[0]*(n+1) for _ in range(n+1)]
    for i in range(n+1):
        for j in range(i+1, n+1):
            d = random.randint(1, max_dist)
            mat[i][j] = mat[j][i] = d
    return mat

def generate_service_times(n, max_service=10):
    """Sinh service time cho các điểm 1..n, điểm 0 (kho) có service time = 0."""
    return [0] + [random.randint(1, max_service) for _ in range(n)]

def compute_arrival_times(route, dist, service, t0=0):
    """Tính arrival time dọc theo route (bao gồm kho ở đầu/cuối)."""
    curr_time = t0
    arrivals = [curr_time]
    for i in range(len(route)-1):
        u, v = route[i], route[i+1]
        curr_time += dist[u][v]
        arrivals.append(curr_time)
        curr_time += service[v]
    return arrivals

def generate_time_windows(arrivals, wide=True, slack_wide=1000, slack_narrow=5):
    """
    Sinh time window quanh arrival times:
     - wide: [0, arrival+slack_wide]
     - narrow: [arrival−slack_narrow..arrival+slack_narrow]
    """
    tw = []
    for arr in arrivals:
        if wide:
            e, l = 0, arr + slack_wide
        else:
            slack = random.randint(0, slack_narrow)
            e = max(0, arr - slack)
            l = arr + slack
        tw.append((e, l, 0))
    return tw

def make_test(n, wide):
    """
    Sinh một test với:
     - n khách hàng
     - TW rộng/narrow theo tham số wide
    Trả về chuỗi input chuẩn:
      Line1: N
      tiếp theo N dòng: e(i) l(i) d(i)
      tiếp theo N+1 dòng: ma trận khoảng cách
    """
    # 1) tạo một base route: kho(0) → shuffle(1..n) → kho(0)
    customers = list(range(1, n+1))
    random.shuffle(customers)
    route = [0] + customers + [0]

    # 2) sinh dữ liệu
    dist    = generate_distance_matrix(n)
    service = generate_service_times(n)
    arrivals= compute_arrival_times(route, dist, service)

    # 3) sinh time windows
    tw = generate_time_windows(arrivals, wide=wide)

    # 4) build input text
    lines = [str(n)]
    for i in range(1, n+1):
        e, l, _ = tw[i]
        d = service[i]
        lines.append(f"{e} {l} {d}")
    for i in range(n+1):
        lines.append(" ".join(map(str, dist[i])))
    return "\n".join(lines), " ".join(str(x) for x in customers)

def main():
    random.seed(42)
    specs = [
        (20,  True),  # 1. N≤10, TW rộng
        (50,  True),  # 2. N≤50, TW rộng
        (10,  False), # 3. N≤10, TW hẹp
        (50,  False), # 4. N≤50, TW hẹp
        (200, True),  # 5. N≤200, TW rộng
        (300, False), # 6. N≤200, TW hẹp
        (5000,True),  # 7. N≤5000, TW rộng
        (2000,False), # 8. N≤5000, TW hẹp
    ]

    # Tạo thư mục chứa test nếu chưa có
    os.makedirs("tests", exist_ok=True)

    for idx, (n, wide) in enumerate(specs, 1):
        os.makedirs(f"tests/test{idx}", exist_ok=True)
        content = make_test(n, wide)
        in_name = f"tests/test{idx}/input.in"
        out_name = f"tests/test{idx}/output.out"
        with open(in_name, "w") as f:
            f.write(content[0])
        with open(out_name, "w") as f:
            f.write(content[1])  

def generate_cluster_trap(n1, n2, inter_dist=1000, slack_narrow=5):
    """
    Cluster-based trap:
     - Two clusters of sizes n1 and n2.
     - Intra-cluster distances small [1..10], inter-cluster distances large ~inter_dist.
     - Base route alternates between clusters: 1, n1+1, 2, n1+2, ...
     - TW narrow around true arrival times so only alternating route is feasible.
    """
    n = n1 + n2
    # Build distance matrix
    N = n + 1
    dist = [[0]*N for _ in range(N)]
    # intra-cluster
    for i in range(1, n1+1):
        for j in range(i+1, n1+1):
            d = random.randint(1, 10)
            dist[i][j] = dist[j][i] = d
    for i in range(n1+1, n+1):
        for j in range(i+1, n+1):
            d = random.randint(1, 10)
            dist[i][j] = dist[j][i] = d
    # inter-cluster
    for i in range(1, n1+1):
        for j in range(n1+1, n+1):
            d = inter_dist + random.randint(-5, 5)
            dist[i][j] = dist[j][i] = d
    # distances to/from depot 0: small to both clusters
    for i in range(1, n+1):
        dist[0][i] = dist[i][0] = random.randint(10, 20)
    # service times zero for simplicity
    service = [0]*(n+1)
    # base alternating route
    route = [0]
    for k in range(max(n1, n2)):
        if k < n1: route.append(k+1)
        if k < n2: route.append(n1 + k+1)
    route.append(0)
    # compute arrivals
    curr_time = 0
    arrivals = [0]
    for i in range(len(route)-1):
        u, v = route[i], route[i+1]
        curr_time += dist[u][v]
        arrivals.append(curr_time)
        curr_time += service[v]
    # narrow TW
    tw = []
    for arr in arrivals:
        slack = random.randint(0, slack_narrow)
        e = max(0, arr - slack)
        l = arr + slack
        tw.append((e, l, 0))
    # build input
    lines = [str(n)]
    # skip tw[0] for depot, use dummy
    for i in range(1, n+1):
        e, l, d = tw[i]
        lines.append(f"{e} {l} {d}")
    for i in range(n+1):
        lines.append(" ".join(map(str, dist[i])))
    return "\n".join(lines), " ".join(str(x) for x in route[1:-1])

def generate_greedy_trap(n, slack_narrow=5):
    """
    Greedy trap:
     - One customer X close to depot (dist small) but TW late,
       others farther but TW early.
     - Nearest-Neighbor picks X first and then violates others.
     - Optimum visits others first then X.
    """
    # choose X = 1
    # distances
    N = n + 1
    dist = [[0]*N for _ in range(N)]
    # to depot: X very close, others far
    dist[0][1] = dist[1][0] = 1
    for i in range(2, n+1):
        d = random.randint(20, 30)
        dist[0][i] = dist[i][0] = d
    # inter-customer distances random moderate
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            dist[i][j] = dist[j][i] = random.randint(5, 15)
    # service times zero
    service = [0]*(n+1)
    # optimum route: 0 → 2,3,...,n → 1 → 0
    route = [0] + list(range(2, n+1)) + [1, 0]
    # compute arrivals
    curr_time = 0
    arrivals = [0]
    for i in range(len(route)-1):
        u, v = route[i], route[i+1]
        curr_time += dist[u][v]
        arrivals.append(curr_time)
        curr_time += service[v]
    # TW: X=1 wide slack, others narrow
    tw = []
    for idx, arr in enumerate(arrivals):
        if idx == 1:  # arrival at first visit (2)
            # narrow for others
            slack = random.randint(0, slack_narrow)
            e = max(0, arr - slack)
            l = arr + slack
        elif idx == len(arrivals)-2:  # arrival at 1
            # wide for X
            e, l = 0, arr + 1000
        else:
            # if idx != last depot
            slack = random.randint(0, slack_narrow)
            e = max(0, arr - slack)
            l = arr + slack
        tw.append((e, l, 0))
    # build input
    lines = [str(n)]
    for i in range(1, n+1):
        e, l, d = tw[i]
        lines.append(f"{e} {l} {d}")
    for i in range(n+1):
        lines.append(" ".join(map(str, dist[i])))
    return "\n".join(lines), " ".join(str(x) for x in route[1:-1])

def generate_delay_trap(n, slack_narrow=2):
    """
    Delay trap:
     - Chain of n customers where TW are tightly chained.
     - If any is inserted incorrectly, suffix will violate heavily.
    """
    N = n + 1
    # linear chain distances = 10 between i→i+1
    dist = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if abs(i-j)==1:
                dist[i][j] = 10
            elif i!=j:
                dist[i][j] = 20
    # service times zero
    service = [0]*N
    # base route 0→1→2→...→n→0
    route = list(range(0, n+1)) + [0]
    # compute arrivals
    curr_time = 0
    arrivals = [0]
    for i in range(len(route)-1):
        u, v = route[i], route[i+1]
        curr_time += dist[u][v]
        arrivals.append(curr_time)
        curr_time += service[v]
    # narrow TW around arrivals
    tw = []
    for arr in arrivals:
        slack = random.randint(0, slack_narrow)
        e = max(0, arr - slack)
        l = arr + slack
        tw.append((e, l, 0))
    # build input
    lines = [str(n)]
    for i in range(1, n+1):
        e, l, d = tw[i]
        lines.append(f"{e} {l} {d}")
    for i in range(n+1):
        lines.append(" ".join(map(str, dist[i])))
    return "\n".join(lines), " ".join(str(x) for x in route[1:-1])

def gen_trap_test():
    # Example: write these to files
    os.makedirs("tests/trap", exist_ok=True)
    content = generate_cluster_trap(100, 100, inter_dist=600)
    with open("tests/test9/input.in", "w") as f:
        f.write(content[0])
    with open("tests/test9/output.out", "w") as f:
        f.write(content[1])
        
    content = generate_greedy_trap(150)
    with open("tests/test10/input.in", "w") as f:
        f.write(content[0])
    with open("tests/test10/output.out", "w") as f:
        f.write(content[1])
        
    content = generate_delay_trap(200)
    with open("tests/test11/input.in", "w") as f:
        f.write(content[0])
    with open("tests/test11/output.out", "w") as f:
        f.write(content[1])

    print("Generated cluster_trap.in, greedy_trap.in, delay_trap.in")
random.seed(42)
gen_trap_test()
