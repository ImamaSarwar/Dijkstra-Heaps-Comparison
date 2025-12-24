# Dijkstra Heaps Comparison

## Project Overview
This project presents an empirical analysis of three priority queue implementations, **Binary Heap**, **Fibonacci Heap**, and **Hollow Heap**, within the context of **Dijkstra's shortest path algorithm** on large-scale road networks. The study evaluates operational efficiency, memory consumption, structural characteristics, and practical performance on real-world datasets from major Asian metropolitan areas.

The goal is to compare these heaps in terms of **insert, extract-min, and decrease-key operations**, and assess their practical suitability for dynamic shortest-path computations.

## Heaps Overview

### 1. Binary Heap
- Complete binary tree with the min-heap property.
- **Operations:**
  - Insert: O(log n)
  - Extract-Min: O(log n)
  - Decrease-Key: O(log n)
- **Strengths:** Simple array-based implementation, excellent cache performance.
- **Limitations:** Cannot efficiently merge heaps; decrease-key requires position tracking.

### 2. Fibonacci Heap
- Collection of min-heap-ordered trees with lazy consolidation.
- **Operations (Amortized):**
  - Insert: O(1)
  - Extract-Min: O(log n)
  - Decrease-Key: O(1)
- **Strengths:** Excellent theoretical performance for decrease-key.
- **Limitations:** High memory overhead, complex implementation, pointer-heavy operations.

### 3. Hollow Heap
- Modern heap combining theoretical efficiency with practical simplicity.
- **Operations (Amortized):**
  - Insert: O(1)
  - Extract-Min: O(log n)
  - Decrease-Key: O(1)
- **Strengths:** No cascading cuts, simpler decrease-key, predictable memory behavior.
- **Limitations:** Slight overhead from hollow nodes during cleanup.

## Implementation Details

- **Binary Heap:** Array-based with position map for decrease-key.
- **Fibonacci Heap:** Circular doubly-linked root list, cascading cuts, and degree tracking.
- **Hollow Heap:** Nodes can be hollow; decrease-key creates new solid nodes, lazy cleanup.

**Instrumentation:**
- Operation timing (microsecond precision)
- Heap structural metrics (height, tree count, rank)
- Memory usage tracking
- CSV output for results

**Graph Representation:**
- Adjacency list: `vector<vector<pair<int, double>>>`
- Directed graphs with weighted edges (meters)

## Datasets

1. **Hong Kong Road Network:** 43,620 vertices, 91,542 edges
2. **Shanghai Road Network:** 390,171 vertices, 855,982 edges
3. **Chongqing Road Network:** 1,185,464 vertices, 2,428,866 edges

*All datasets are edge lists with distance weights.*

## Experiments

### Experiment A — All-Pairs Shortest Path
- Dijkstra’s algorithm run from multiple source vertices.
- Metrics recorded: operation time, heap height, tree count, memory usage.

### Experiment B — Operation Profiling
- 100,000 random operations (40% insert, 30% extract-min, 30% decrease-key)
- Tracked per-operation timing and heap structural changes.

## Key Findings

- **Binary Heap:** Fastest in practice due to cache-friendly array layout; low memory usage.
- **Fibonacci Heap:** Theoretical O(1) decrease-key; practical runtime dominated by extract-min and pointer overhead.
- **Hollow Heap:** Best balance of simplicity and performance; avoids cascading cuts and maintains good memory behavior.

**Performance Highlights (Hong Kong dataset):**

| Heap Type     | Insert (μs) | Extract-Min (μs) | Decrease-Key (μs) | Total Runtime (ms) |
|--------------|-------------|-----------------|-----------------|------------------|
| Binary Heap  | 0.161       | 0.179           | 0.170           | 1,379.466        |
| Fibonacci Heap | 3.176     | 20.217          | 3.420           | 62,681.793       |
| Hollow Heap  | 0.160       | 0.892           | 0.191           | 5,689.371        |

## Recommendations

- Use **Binary Heap** for small to medium graphs—simple and efficient.
- Use **Hollow Heap** for very large graphs—efficient decrease-key without cascading cuts.
- Avoid **Fibonacci Heap** in production; suitable mostly for theoretical analysis.


## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/ImamaSarwar/Dijkstra-Heaps-Comparison.git
   ```
2. Navigate to the project folder:
```bash
cd dijkstra-heaps-comparison
```
3. Compile the code (example for g++):
```bash
g++ -std=c++17 dijkstra_heaps.cpp -o dijkstra
```
4. Run the program
5. Follow prompts to select dataset and view results. Output tables are also saved to CSV files.
