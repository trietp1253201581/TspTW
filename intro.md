# Hướng dẫn sử dụng
## Mô hình hóa
1. Mỗi đối tượng trên đường đi sẽ được miêu tả bằng lớp `Client`.
2. Một bài toán TSP Time Window sẽ được mô hình hóa bằng lớp `TSPTWProblem`.
3. Mỗi ràng buộc có thể được thêm vào problem để tính số vi phạm (violations) đều phải implement `Constraint`.
4. Một lời giải hoán vị được mô hình hóa bằng lớp `PermuSolution`.
5. Một thuật toán heuristic, meta-heuristic sẽ implement `Solver` và cài đặt phương thức `solve` (với tham số tùy chọn).
## Example
Xem trong [example.py](./example.py)