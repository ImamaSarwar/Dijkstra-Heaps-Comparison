#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <tuple>
#include <sstream>
#include <iomanip>
#include <map>
#include <cmath>
#include <algorithm>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <limits>
#include <queue>
#include <cassert>
#include <random>
using namespace std;

// ==========================================================
//               HOLLOW HEAP IMPLEMENTATION
// ==========================================================
class HollowHeap {
public:
    struct Node {
        int key;
        double value;
        int rank;
        Node* child;
        Node* next;
        Node* ep;
        bool is_hollow;

        Node(int k, double v)
            : key(k), value(v), rank(0),
            child(nullptr), next(nullptr),
            ep(nullptr), is_hollow(false) {
        }
    };

private:
    Node* min_root;
    int size_;
    int num_trees;
    int max_rank_observed;
    int heap_height_;
    size_t maxMemoryObserved;

    // Metrics
    int max_heap_height_observed_;
    int max_trees_observed_;
    int max_heap_size_observed_;
    int insert_count_;
    int extract_min_count_;
    int decrease_key_count_;

    // Timing
    chrono::duration<double, micro> insertTime;
    chrono::duration<double, micro> extractMinTime;
    chrono::duration<double, micro> decreaseKeyTime;

    void updateMemoryMetrics() {
        size_t currentMemory = calculateMemoryUsage();
        if (currentMemory > maxMemoryObserved) {
            maxMemoryObserved = currentMemory;
        }
    }

    // Link two trees
    Node* link(Node* a, Node* b) {
        if (a->value > b->value) swap(a, b);

        // Link b under a
        b->next = a->child;
        a->child = b;
        b->ep = a;  // ep points to parent

        if (a->rank == b->rank) {
            a->rank++;
        }

        return a;
    }

    // Calculate height of a tree
    int calculateTreeHeight(Node* node) {
        if (!node) return 0;

        int maxChildHeight = 0;
        Node* child = node->child;
        while (child) {
            maxChildHeight = max(maxChildHeight, calculateTreeHeight(child));
            child = child->next;
        }

        // Only count solid nodes in the height If current node is hollow, height is just max child height If current node is solid, height is 1 + max child height
        return node->is_hollow ? maxChildHeight : 1 + maxChildHeight;
    }

    // Calculate overall heap height
    int calculateHeapHeight() {
        int maxHeight = 0;
        Node* cur = min_root;
        while (cur) {
            maxHeight = max(maxHeight, calculateTreeHeight(cur));
            cur = cur->next;
        }
        return maxHeight;
    }

    void updateMaxRank() {
        max_rank_observed = 0;
        for (Node* cur = min_root; cur; cur = cur->next) {
            if (!cur->is_hollow)
                max_rank_observed = max(max_rank_observed, cur->rank);
        }
    }

    void updateMetrics() {
        heap_height_ = calculateHeapHeight();
        max_heap_height_observed_ = max(max_heap_height_observed_, heap_height_);
        max_trees_observed_ = max(max_trees_observed_, num_trees);
        max_heap_size_observed_ = max(max_heap_size_observed_, size_);
        updateMaxRank();
    }

    // Remove hollow roots that are no longer referenced
    void removeHollowRoots() {
        Node* prev = nullptr;
        Node* cur = min_root;

        while (cur) {
            Node* next = cur->next;

            if (cur->is_hollow && cur->ep == nullptr) {

                if (prev) {
                    prev->next = next;
                }
                else {
                    min_root = next;
                }

                // Add children to root list
                Node* child = cur->child;
                while (child) {
                    Node* next_child = child->next;
                    if (child->ep == cur) {
                        child->ep = nullptr; // Remove reference to hollow parent
                    }
                    // Add child to root list
                    child->next = min_root;
                    min_root = child;
                    num_trees++;
                    child = next_child;
                }

                delete cur;
                num_trees--;

                updateMemoryMetrics();
            }
            else {
                prev = cur;
            }

            cur = next;
        }
    }

    // Consolidate using HashMap
    void consolidate() {
        if (!min_root) return;

        unordered_map<int, Node*> rank_map;
        vector<Node*> roots;

        // Collect all solid roots
        Node* cur = min_root;
        while (cur) {
            Node* next = cur->next;
            cur->next = nullptr;
            if (!cur->is_hollow) {
                roots.push_back(cur);
            }
            cur = next;
        }

        min_root = nullptr;
        num_trees = 0;

        // Consolidate solid roots by rank
        for (Node* root : roots) {
            while (rank_map.count(root->rank)) {
                Node* other = rank_map[root->rank];
                rank_map.erase(root->rank);
                root = link(root, other);
            }
            rank_map[root->rank] = root;
        }

        // Rebuild root list
        for (auto& pair : rank_map) {
            Node* root = pair.second;
            root->next = min_root;
            min_root = root;
            num_trees++;
        }

        updateMemoryMetrics();  //Update after consolidation
    }

    // Find actual minimum solid node
    Node* findActualMin() const {
        Node* actual_min = nullptr;
        for (Node* cur = min_root; cur; cur = cur->next) {
            if (!cur->is_hollow) {
                if (!actual_min || cur->value < actual_min->value)
                    actual_min = cur;
            }
        }
        return actual_min;
    }

    void removeAllHollowRoots() {
        bool removed_any;
        do {
            removed_any = false;
            Node* prev = nullptr;
            Node* cur = min_root;

            while (cur) {
                Node* next = cur->next;

                if (cur->is_hollow) {
                    // Remove this hollow root
                    if (prev) {
                        prev->next = next;
                    }
                    else {
                        min_root = next;
                    }

                    // Add all children to root list
                    Node* child = cur->child;
                    while (child) {
                        Node* next_child = child->next;
                        child->next = min_root;
                        min_root = child;
                        if (child->ep == cur) {
                            child->ep = nullptr;
                        }
                        num_trees++;
                        child = next_child;
                    }

                    delete cur;
                    num_trees--;
                    removed_any = true;
                    updateMemoryMetrics();

                    break;
                }
                else {
                    prev = cur;
                }

                cur = next;
            }
        } while (removed_any); // Keep going until no more hollow roots are found
    }
public:
    HollowHeap()
        : min_root(nullptr), size_(0), num_trees(0),
        max_rank_observed(0), heap_height_(0),
        max_heap_height_observed_(0), max_trees_observed_(0),
        max_heap_size_observed_(0),
        insert_count_(0), extract_min_count_(0), decrease_key_count_(0),
        maxMemoryObserved(0) {
        insertTime = chrono::duration<double, micro>{ 0 };
        extractMinTime = chrono::duration<double, micro>{ 0 };
        decreaseKeyTime = chrono::duration<double, micro>{ 0 };
    }

    ~HollowHeap() {
        // Clean up all nodes
        vector<Node*> nodes;
        for (Node* cur = min_root; cur; cur = cur->next) nodes.push_back(cur);
        for (size_t i = 0; i < nodes.size(); i++) {
            Node* node = nodes[i];
            for (Node* child = node->child; child; child = child->next)
                nodes.push_back(child);
        }
        for (Node* node : nodes) delete node;
    }

    size_t calculateMemoryUsage() const {
        size_t memory = sizeof(*this);
        vector<Node*> nodes;

        // count nodes
        int nodeCount = 0;
        for (Node* cur = min_root; cur; cur = cur->next) {
            nodes.push_back(cur);
            nodeCount++;
        }

        for (size_t i = 0; i < nodes.size(); i++) {
            Node* node = nodes[i];
            for (Node* child = node->child; child; child = child->next) {
                nodes.push_back(child);
                nodeCount++;
            }
        }

        memory += nodes.size() * sizeof(Node);

        return memory;
    }

    bool empty() const {
        if (!min_root) return true;
        Node* cur = min_root;
        while (cur) {
            if (!cur->is_hollow) return false;
            cur = cur->next;
        }
        return true;
    }

    pair<int, double> findMin() const {
        Node* min_node = findActualMin();
        if (!min_node) return { -1, -1 };
        return { min_node->key, min_node->value };
    }

    Node* insert(int key, double value) {
        auto start = chrono::high_resolution_clock::now();

        Node* new_node = new Node(key, value);
        insert_count_++;

        // Insert as root
        new_node->next = min_root;
        min_root = new_node;

        num_trees++;
        size_++;
        updateMetrics();

        auto end = chrono::high_resolution_clock::now();
        insertTime += end - start;

        updateMemoryMetrics();
        return new_node;
    }

    pair<int, double> extractMin() {
        auto start = chrono::high_resolution_clock::now();

        if (empty()) return { -1, -1 };

        Node* old_min = findActualMin();
        if (!old_min) return { -1, -1 };

        pair<int, double> result = { old_min->key, old_min->value };
        extract_min_count_++;

        // Remove old_min from root list
        Node* prev = nullptr;
        Node* cur = min_root;
        while (cur && cur != old_min) { prev = cur; cur = cur->next; }
        if (prev) prev->next = old_min->next;
        else min_root = old_min->next;

        // Add children to root list and update their ep pointers
        Node* child = old_min->child;
        while (child) {
            Node* next_child = child->next;
            child->next = min_root;
            min_root = child;
            if (child->ep == old_min) {
                child->ep = nullptr; //remove reference to the deleted parent
            }
            num_trees++; //each child becomes a new root
            child = next_child;
        }

        delete old_min;
        size_--;

        removeAllHollowRoots();

        //consolidate the heap
        consolidate();
        updateMetrics();

        auto end = chrono::high_resolution_clock::now();
        extractMinTime += end - start;

        updateMemoryMetrics();
        return result;
    }

    Node* decreaseKey(Node* node, double new_value) {
        auto start = chrono::high_resolution_clock::now();

        if (new_value >= node->value) return node;
        decrease_key_count_++;

        // Create new solid node
        Node* new_node = new Node(node->key, new_value);

        // Hollow the old node
        node->is_hollow = true;

        // Rank transfer
        new_node->rank = max(0, node->rank - 2);

        // Add new node to root list
        new_node->next = min_root;
        min_root = new_node;

        num_trees++;
        size_++;
        insert_count_++;

        updateMetrics();

        auto end = chrono::high_resolution_clock::now();
        decreaseKeyTime += end - start;

        updateMemoryMetrics();
        return new_node;
    }

    // Timing getters and metrics
    double getInsertTimeMicro() const { return insertTime.count(); }
    double getExtractMinTimeMicro() const { return extractMinTime.count(); }
    double getDecreaseKeyTimeMicro() const { return decreaseKeyTime.count(); }
    int getSize() const { return size_; }
    int getNumTrees() const { return num_trees; }
    int getHeapHeight() const { return heap_height_; }
    int getMaxRankObserved() const { return max_rank_observed; }
    int getMaxHeapHeightObserved() const { return max_heap_height_observed_; }
    int getMaxTreesObserved() const { return max_trees_observed_; }
    int getMaxHeapSizeObserved() const { return max_heap_size_observed_; }
    int getInsertCount() const { return insert_count_; }
    int getExtractMinCount() const { return extract_min_count_; }
    int getDecreaseKeyCount() const { return decrease_key_count_; }
    int getTotalOperations() const {
        return insert_count_ + extract_min_count_ + decrease_key_count_;
    }
    size_t getMaxMemoryObserved() const { return maxMemoryObserved; }
};

// ==========================================================
//                FIBONACCI HEAP IMPLEMENTATION
// ==========================================================

class FibonacciHeap {
public:
    struct Node {
        int vertex;
        double dist;
        Node* parent;
        Node* child;
        Node* left;
        Node* right;
        int degree;
        bool marked;
        bool inHeap;

        Node(int v, double d)
            : vertex(v), dist(d), parent(nullptr), child(nullptr),
            left(this), right(this), degree(0), marked(false), inHeap(true) {
        }
    };

private:
    Node* minNode;
    int nodeCount;
    int maxNodeCountObserved = 0;
    unordered_map<int, Node*> nodeMap;
    size_t maxMemoryObserved = 0;

    // Metrics tracking
    int insertCount;
    int extractMinCount;
    int decreaseKeyCount;
    int totalCascadingCuts;
    int maxTreesObserved;
    int maxDegreeObserved;
    double sampledHeightsSum;
    int sampledHeightsCount;
    int maxHeapHeightObserved;

    chrono::duration<double, micro> insertTime;
    chrono::duration<double, micro> extractMinTime;
    chrono::duration<double, micro> decreaseKeyTime;

    void updateMemoryMetrics() {
        size_t currentMemory = calculateMemoryUsage();
        if (currentMemory > maxMemoryObserved) {
            maxMemoryObserved = currentMemory;
        }
    }

    void addToRootList(Node* node) {
        if (minNode == nullptr) {
            minNode = node;
            node->left = node;
            node->right = node;
        }
        else {
            node->right = minNode;
            node->left = minNode->left;
            minNode->left->right = node;
            minNode->left = node;
            if (node->dist < minNode->dist) {
                minNode = node;
            }
        }
    }

    void removeFromRootList(Node* node) {
        if (node->right == node) {
            minNode = nullptr;
        }
        else {
            node->left->right = node->right;
            node->right->left = node->left;
            if (node == minNode) {
                minNode = node->right;
            }
        }
        node->left = node;
        node->right = node;
    }

    void link(Node* child, Node* parent) {
        removeFromRootList(child);
        child->parent = parent;

        if (parent->child == nullptr) {
            parent->child = child;
            child->left = child;
            child->right = child;
        }
        else {
            child->right = parent->child;
            child->left = parent->child->left;
            parent->child->left->right = child;
            parent->child->left = child;
        }
        parent->degree++;
        child->marked = false;

        updateMemoryMetrics();
    }

    void consolidate() {
        if (minNode == nullptr) return;

        vector<Node*> degreeTable(64, nullptr);
        vector<Node*> rootNodes;

        Node* current = minNode;
        do {
            rootNodes.push_back(current);
            current = current->right;
        } while (current != minNode);

        for (Node* node : rootNodes) {
            Node* x = node;
            int d = x->degree;

            while (d >= (int)degreeTable.size()) {
                degreeTable.resize(degreeTable.size() * 2, nullptr);
            }

            while (degreeTable[d] != nullptr) {
                Node* y = degreeTable[d];
                if (x->dist > y->dist) swap(x, y);
                link(y, x);
                degreeTable[d] = nullptr;
                d++;

                while (d >= (int)degreeTable.size()) {
                    degreeTable.resize(degreeTable.size() * 2, nullptr);
                }
            }
            degreeTable[d] = x;
        }

        minNode = nullptr;
        for (Node* node : degreeTable) {
            if (node != nullptr) {
                addToRootList(node);
                maxDegreeObserved = max(maxDegreeObserved, node->degree);
            }
        }

        sampleStructureMetrics();
        updateMemoryMetrics();
    }

    void cut(Node* child, Node* parent) {
        // Remove child from parent's child list
        if (child->right == child) {
            parent->child = nullptr;
        }
        else {
            child->left->right = child->right;
            child->right->left = child->left;
            if (parent->child == child) {
                parent->child = child->right;
            }
        }
        parent->degree--;

        // Add child to root list
        addToRootList(child);
        child->parent = nullptr;
        child->marked = false;

        sampleStructureMetrics();
        updateMemoryMetrics();
    }

    void cascadingCut(Node* node) {
        Node* parent = node->parent;
        if (parent != nullptr) {
            if (!node->marked) {
                node->marked = true;
            }
            else {
                cut(node, parent);
                totalCascadingCuts++;
                cascadingCut(parent);
            }
        }
    }

    int countRootTrees() const {
        if (minNode == nullptr) return 0;

        int count = 0;
        Node* current = minNode;
        do {
            count++;
            current = current->right;
        } while (current != minNode);

        return count;
    }

    // Height of an individual tree
    int calculateTreeHeight(Node* node) const {
        if (node == nullptr) return 0;
        int maxChildHeight = 0;
        if (node->child != nullptr) {
            Node* child = node->child;
            do {
                maxChildHeight = max(maxChildHeight, calculateTreeHeight(child));
                child = child->right;
            } while (child != node->child);
        }
        return 1 + maxChildHeight;
    }

    // Sample current heap height and number of trees for metrics
    void sampleStructureMetrics() {
        if (minNode == nullptr) return;
        int h = 0;
        Node* current = minNode;
        do {
            h = max(h, calculateTreeHeight(current));
            current = current->right;
        } while (current != minNode);

        sampledHeightsSum += h;
        sampledHeightsCount += 1;
        maxHeapHeightObserved = max(maxHeapHeightObserved, h);

        int trees = countRootTrees();
        maxTreesObserved = max(maxTreesObserved, trees);
    }

public:
    FibonacciHeap()
        : minNode(nullptr), nodeCount(0),
        insertCount(0), extractMinCount(0), decreaseKeyCount(0),
        totalCascadingCuts(0), maxTreesObserved(0), maxDegreeObserved(0),
        sampledHeightsSum(0.0), sampledHeightsCount(0), maxHeapHeightObserved(0),
        maxMemoryObserved(0) {
        insertTime = chrono::duration<double, micro>{ 0 };
        extractMinTime = chrono::duration<double, micro>{ 0 };
        decreaseKeyTime = chrono::duration<double, micro>{ 0 };
    }

    ~FibonacciHeap() {
        if (minNode != nullptr) {
            vector<Node*> nodes;
            Node* current = minNode;
            do {
                nodes.push_back(current);
                current = current->right;
            } while (current != minNode);

            for (Node* node : nodes) {
                deleteSubtree(node);
            }
        }
    }

    size_t calculateMemoryUsage() const {
        size_t memory = sizeof(*this);

        if (minNode == nullptr) return memory;

        unordered_set<Node*> visited;
        queue<Node*> q;

        // Start with root nodes
        Node* current = minNode;
        do {
            if (visited.find(current) == visited.end()) {
                q.push(current);
                visited.insert(current);
            }
            current = current->right;
        } while (current != minNode);

        // Process all nodes
        while (!q.empty()) {
            Node* node = q.front();
            q.pop();

            memory += sizeof(Node);

            // Process children
            if (node->child != nullptr) {
                Node* child = node->child;
                do {
                    if (visited.find(child) == visited.end()) {
                        visited.insert(child);
                        q.push(child);
                    }
                    child = child->right;
                } while (child != node->child);
            }
        }

        memory += nodeMap.size() * (sizeof(int) + sizeof(Node*));

        return memory;
    }

    void deleteSubtree(Node* node) {
        if (node == nullptr) return;
        Node* child = node->child;
        if (child != nullptr) {
            vector<Node*> children;
            Node* current = child;
            do {
                children.push_back(current);
                current = current->right;
            } while (current != child);

            for (Node* c : children) {
                deleteSubtree(c);
            }
        }
        delete node;
    }

    bool empty() const { return minNode == nullptr; }
    int size() const { return nodeCount; }

    bool contains(int v) const {
        auto it = nodeMap.find(v);
        return it != nodeMap.end() && it->second->inHeap;
    }

    void insert(int v, double d) {
        auto start = chrono::high_resolution_clock::now();

        Node* newNode = new Node(v, d);
        nodeMap[v] = newNode;
        addToRootList(newNode);
        nodeCount++;
        maxNodeCountObserved = max(maxNodeCountObserved, nodeCount);

        sampleStructureMetrics();
        updateMemoryMetrics();

        auto end = chrono::high_resolution_clock::now();
        insertTime += end - start;
        insertCount++;
    }

    pair<int, double> extractMin() {
        auto start = chrono::high_resolution_clock::now();

        if (minNode == nullptr) throw runtime_error("Heap is empty");

        Node* oldMin = minNode;
        pair<int, double> result = { oldMin->vertex, oldMin->dist };

        oldMin->inHeap = false;

        // Move children to root list
        if (oldMin->child != nullptr) {
            vector<Node*> children;
            Node* cur = oldMin->child;
            do {
                children.push_back(cur);
                cur = cur->right;
            } while (cur != oldMin->child);

            for (Node* c : children) {
                c->left = c;
                c->right = c;
                c->parent = nullptr;
                addToRootList(c);
            }
            oldMin->child = nullptr;
        }

        Node* nextRoot = oldMin->right;
        removeFromRootList(oldMin);

        if (nextRoot == oldMin) {
            minNode = nullptr;
        }
        else {
            minNode = nextRoot;
            consolidate();
        }

        nodeMap.erase(oldMin->vertex);
        delete oldMin;
        nodeCount--;

        if (minNode != nullptr) sampleStructureMetrics();
        updateMemoryMetrics();

        auto end = chrono::high_resolution_clock::now();
        extractMinTime += end - start;
        extractMinCount++;

        return result;
    }

    void decreaseKey(int v, double newDist) {
        auto start = chrono::high_resolution_clock::now();

        auto it = nodeMap.find(v);
        if (it == nodeMap.end() || !it->second->inHeap) {
            return;
        }

        Node* node = it->second;
        if (!node->inHeap) {
            return;
        }

        if (newDist > node->dist) {
            throw runtime_error("New key is greater than current key");
        }

        node->dist = newDist;
        Node* parent = node->parent;

        if (parent != nullptr && node->dist < parent->dist) {
            cut(node, parent);
            cascadingCut(parent);
        }

        if (minNode == nullptr || node->dist < minNode->dist) {
            minNode = node;
        }

        sampleStructureMetrics();
        updateMemoryMetrics();

        auto end = chrono::high_resolution_clock::now();
        decreaseKeyTime += end - start;
        decreaseKeyCount++;
    }

    size_t getMaxMemoryObserved() const { return maxMemoryObserved; }

    // Metrics getters
    int getHeapHeight() const { return maxHeapHeightObserved; }
    int getNumTrees() const { return maxTreesObserved; }
    double getAvgSampledHeight() const { return (sampledHeightsCount > 0) ? (sampledHeightsSum / sampledHeightsCount) : 0.0; }
    int getMaxHeapSize() const { return maxNodeCountObserved; }
    int getMaxTreesObserved() const { return maxTreesObserved; }
    int getCascadingCuts() const { return totalCascadingCuts; }
    int getMaxDegreeObserved() const { return maxDegreeObserved; }
    int getInsertCount() const { return insertCount; }
    int getExtractMinCount() const { return extractMinCount; }
    int getDecreaseKeyCount() const { return decreaseKeyCount; }
    double getInsertTimeMicro() const { return insertTime.count(); }
    double getExtractMinTimeMicro() const { return extractMinTime.count(); }
    double getDecreaseKeyTimeMicro() const { return decreaseKeyTime.count(); }
};

// ==========================================================
//                BINARY MIN HEAP WITH DECREASE-KEY
// ==========================================================

class BinaryHeap {
public:
    struct Node { int vertex; double dist; };

    vector<Node> heap;
    vector<int> pos;

    int insertCount;
    int extractMinCount;
    int decreaseKeyCount;
    int maxHeapSize;
    int maxActualHeight;
    chrono::duration<double, micro> insertTime;
    chrono::duration<double, micro> extractMinTime;
    chrono::duration<double, micro> decreaseKeyTime;

    BinaryHeap(int n) {
        heap.reserve(n);
        pos.assign(n, -1);
        insertCount = 0;
        extractMinCount = 0;
        decreaseKeyCount = 0;
        maxHeapSize = 0;
        maxActualHeight = 0;
        insertTime = chrono::duration<double, micro>{ 0 };
        extractMinTime = chrono::duration<double, micro>{ 0 };
        decreaseKeyTime = chrono::duration<double, micro>{ 0 };
    }

    bool empty() const { return heap.empty(); }
    int size() const { return (int)heap.size(); }

    void swapNodes(int i, int j) {
        swap(heap[i], heap[j]);
        pos[heap[i].vertex] = i;
        pos[heap[j].vertex] = j;
    }

    int getNodeHeight(int i) const {
        int height = 1;
        while (i > 0) {
            i = (i - 1) / 2;
            height++;
        }
        return height;
    }

    int getActualHeight() const {
        if (heap.empty()) return 0;
        int maxHeight = 0;
        for (int i = 0; i < (int)heap.size(); i++) {
            maxHeight = max(maxHeight, getNodeHeight(i));
        }
        return maxHeight;
    }

    void bubbleUp(int i) {
        while (i > 0) {
            int parent = (i - 1) / 2;
            if (heap[parent].dist <= heap[i].dist) break;
            swapNodes(parent, i);
            i = parent;
        }
        int currentHeight = getActualHeight();
        if (currentHeight > maxActualHeight) {
            maxActualHeight = currentHeight;
        }
    }

    void bubbleDown(int i) {
        int n = heap.size();
        while (true) {
            int left = 2 * i + 1;
            int right = 2 * i + 2;
            int smallest = i;
            if (left < n && heap[left].dist < heap[smallest].dist) smallest = left;
            if (right < n && heap[right].dist < heap[smallest].dist) smallest = right;
            if (smallest == i) break;
            swapNodes(i, smallest);
            i = smallest;
        }
        int currentHeight = getActualHeight();
        if (currentHeight > maxActualHeight) {
            maxActualHeight = currentHeight;
        }
    }

    void insert(int v, double d) {
        auto start = chrono::high_resolution_clock::now();
        Node n{ v, d };
        heap.push_back(n);
        int idx = (int)heap.size() - 1;
        pos[v] = idx;
        bubbleUp(idx);
        if ((int)heap.size() > maxHeapSize) {
            maxHeapSize = (int)heap.size();
        }
        auto end = chrono::high_resolution_clock::now();
        insertTime += end - start;
        insertCount++;
    }

    void decreaseKey(int v, double newDist) {
        auto start = chrono::high_resolution_clock::now();
        int i = pos[v];
        heap[i].dist = newDist;
        bubbleUp(i);
        auto end = chrono::high_resolution_clock::now();
        decreaseKeyTime += end - start;
        decreaseKeyCount++;
    }
    // In BinaryHeap class
    size_t calculateMemoryUsage() const {
        size_t memory = sizeof(*this);
        memory += heap.capacity() * sizeof(Node);
        memory += pos.capacity() * sizeof(int);
        return memory;
    }

    Node extractMin() {
        auto start = chrono::high_resolution_clock::now();
        Node root = heap[0];
        if (heap.size() > 1) {
            Node last = heap.back();
            heap[0] = last;
            pos[last.vertex] = 0;
            heap.pop_back();
            pos[root.vertex] = -1;
            bubbleDown(0);
        }
        else {
            heap.pop_back();
            pos[root.vertex] = -1;
        }
        auto end = chrono::high_resolution_clock::now();
        extractMinTime += end - start;
        extractMinCount++;
        return root;
    }

    int getMaxActualHeight() const { return maxActualHeight; }
    double getTheoreticalHeight() const { return (maxHeapSize == 0) ? 0 : log2(maxHeapSize) + 1; }
    void resetMetrics() {
        insertCount = extractMinCount = decreaseKeyCount = 0;
        maxHeapSize = 0;
        maxActualHeight = 0;
        insertTime = extractMinTime = decreaseKeyTime = chrono::duration<double, micro>{ 0 };
    }
    int getInsertCount() const { return insertCount; }
    int getExtractMinCount() const { return extractMinCount; }
    int getDecreaseKeyCount() const { return decreaseKeyCount; }
    double getInsertTimeMicro() const { return insertTime.count(); }
    double getExtractMinTimeMicro() const { return extractMinTime.count(); }
    double getDecreaseKeyTimeMicro() const { return decreaseKeyTime.count(); }
    int getMaxHeapSize() const { return maxHeapSize; }
};

// ==========================================================
//                   LOAD GRAPH FROM Dataset
// ==========================================================

vector<vector<pair<int, double>>> loadGraph(const string& filename, int& maxNode, int& edgeCount) {
    // Directed graphs
    ifstream file(filename.c_str());
    if (!file.is_open()) {
        cout << "ERROR: Unable to open file: " << filename << "\n";
        exit(1);
    }

    maxNode = 0;
    edgeCount = 0;
    vector<tuple<int, int, double>> edges;
    edges.reserve(3000000);

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        stringstream ss(line);
        int u, v;
        double w;

        if (!(ss >> u >> v >> w)) {
            continue;
        }

        edges.emplace_back(u, v, w);
        maxNode = max(maxNode, max(u, v));
        edgeCount++;
    }

    int n = maxNode + 1;
    vector<vector<pair<int, double>>> graph(n);

    for (size_t i = 0; i < edges.size(); i++) {
        int u = get<0>(edges[i]);
        int v = get<1>(edges[i]);
        double w = get<2>(edges[i]);
        graph[u].push_back(make_pair(v, w));
    }

    return graph;
}

// ==========================================================
//                      DIJKSTRA USING BINARY HEAP
// ==========================================================

pair<vector<double>, map<string, double>> dijkstra_binaryHeap(const vector<vector<pair<int, double>>>& graph, int src) {

    int n = graph.size();
    vector<double> dist(n, 1e18);
    map<string, double> metrics;

    BinaryHeap pq(n);
    dist[src] = 0;
    pq.insert(src, 0);

    vector<bool> visited(n, false);
    int visitedCount = 0;

    auto dijkstraStart = chrono::high_resolution_clock::now();

    while (!pq.empty()) {
        auto node = pq.extractMin();
        int u = node.vertex;
        double d = node.dist;

        if (visited[u]) continue;
        visited[u] = true;
        visitedCount++;

        for (size_t i = 0; i < graph[u].size(); i++) {
            int v = graph[u][i].first;
            double w = graph[u][i].second;

            if (!visited[v] && d + w < dist[v]) {
                dist[v] = d + w;
                if (pq.pos[v] == -1) {
                    pq.insert(v, dist[v]);
                }
                else {
                    pq.decreaseKey(v, dist[v]);
                }
            }
        }
    }

    auto dijkstraEnd = chrono::high_resolution_clock::now();
    auto totalTime = chrono::duration<double, milli>(dijkstraEnd - dijkstraStart);

    metrics["total_time_ms"] = totalTime.count();
    metrics["insert_avg_us"] = pq.getInsertCount() > 0 ? pq.getInsertTimeMicro() / pq.getInsertCount() : 0;
    metrics["extract_min_avg_us"] = pq.getExtractMinCount() > 0 ? pq.getExtractMinTimeMicro() / pq.getExtractMinCount() : 0;
    metrics["decrease_key_avg_us"] = pq.getDecreaseKeyCount() > 0 ? pq.getDecreaseKeyTimeMicro() / pq.getDecreaseKeyCount() : 0;
    metrics["nodes_visited"] = visitedCount;
    metrics["heap_height"] = pq.getMaxActualHeight();
    metrics["theoretical_heap_height"] = pq.getTheoreticalHeight();
    metrics["num_trees"] = 1;
    metrics["final_heap_size"] = pq.size();
    metrics["max_heap_size"] = pq.getMaxHeapSize();
    metrics["total_operations"] = pq.getInsertCount() + pq.getExtractMinCount() + pq.getDecreaseKeyCount();
    metrics["insert_count"] = pq.getInsertCount();
    metrics["extract_min_count"] = pq.getExtractMinCount();
    metrics["decrease_key_count"] = pq.getDecreaseKeyCount();
    metrics["cascading_cuts"] = 0;
    metrics["max_trees_observed"] = 1;
    metrics["max_degree_observed"] = 0;
    metrics["heap_memory_bytes"] = pq.calculateMemoryUsage();
    metrics["heap_memory_mb"] = pq.calculateMemoryUsage() / (1024.0 * 1024.0);

    return make_pair(dist, metrics);
}

// ==========================================================
//                    DIJKSTRA USING FIBONACCI HEAP
// ==========================================================

pair<vector<double>, map<string, double>> dijkstra_fibonacciHeap(const vector<vector<pair<int, double>>>& graph, int src) {

    int n = graph.size();
    vector<double> dist(n, 1e18);
    map<string, double> metrics;
    vector<bool> visited(n, false);

    FibonacciHeap pq;
    dist[src] = 0;
    pq.insert(src, 0);

    int visitedCount = 0;

    auto dijkstraStart = chrono::high_resolution_clock::now();

    size_t peakMemory = pq.calculateMemoryUsage();

    while (!pq.empty()) {
        auto result = pq.extractMin();
        int u = result.first;
        double d = result.second;

        if (visited[u]) continue;
        visited[u] = true;
        visitedCount++;

        size_t currentMemory = pq.calculateMemoryUsage();
        if (currentMemory > peakMemory) {
            peakMemory = currentMemory;
        }

        for (size_t i = 0; i < graph[u].size(); i++) {
            int v = graph[u][i].first;
            double w = graph[u][i].second;

            if (!visited[v]) {
                double newDist = d + w;

                if (newDist < dist[v]) {
                    dist[v] = newDist;

                    if (pq.contains(v)) {
                        pq.decreaseKey(v, newDist);
                    }
                    else {
                        pq.insert(v, newDist);
                    }

                    currentMemory = pq.calculateMemoryUsage();
                    if (currentMemory > peakMemory) {
                        peakMemory = currentMemory;
                    }
                }
            }
        }
    }

    auto dijkstraEnd = chrono::high_resolution_clock::now();
    auto totalTime = chrono::duration<double, milli>(dijkstraEnd - dijkstraStart);

    // Store metrics
    metrics["total_time_ms"] = totalTime.count();
    metrics["insert_avg_us"] = pq.getInsertCount() > 0 ? pq.getInsertTimeMicro() / pq.getInsertCount() : 0;
    metrics["extract_min_avg_us"] = pq.getExtractMinCount() > 0 ? pq.getExtractMinTimeMicro() / pq.getExtractMinCount() : 0;
    metrics["decrease_key_avg_us"] = pq.getDecreaseKeyCount() > 0 ? pq.getDecreaseKeyTimeMicro() / pq.getDecreaseKeyCount() : 0;
    metrics["nodes_visited"] = visitedCount;
    metrics["heap_height"] = pq.getHeapHeight();
    metrics["avg_heap_height"] = pq.getAvgSampledHeight();
    metrics["num_trees"] = pq.getNumTrees();
    metrics["final_heap_size"] = pq.size();
    metrics["max_heap_size"] = pq.getMaxHeapSize();
    metrics["total_operations"] = pq.getInsertCount() + pq.getExtractMinCount() + pq.getDecreaseKeyCount();
    metrics["insert_count"] = pq.getInsertCount();
    metrics["extract_min_count"] = pq.getExtractMinCount();
    metrics["decrease_key_count"] = pq.getDecreaseKeyCount();
    metrics["cascading_cuts"] = pq.getCascadingCuts();
    metrics["max_trees_observed"] = pq.getMaxTreesObserved();
    metrics["max_degree_observed"] = pq.getMaxDegreeObserved();
    metrics["heap_memory_bytes"] = peakMemory;
    metrics["heap_memory_mb"] = peakMemory / (1024.0 * 1024.0);

    return make_pair(dist, metrics);
}

// ==========================================================
//                    DIJKSTRA USING HOLLOW HEAP
// ==========================================================

pair<vector<double>, unordered_map<string, double>> dijkstra_hollowHeap(const vector<vector<pair<int, double>>>& graph, int src) {

    int n = graph.size();
    vector<double> dist(n, numeric_limits<double>::max());
    unordered_map<string, double> metrics;

    HollowHeap heap;
    vector<HollowHeap::Node*> node_ptrs(n, nullptr);

    auto start = chrono::high_resolution_clock::now();

    dist[src] = 0;
    node_ptrs[src] = heap.insert(src, 0.0);

    int operations = 0;
    int duplicate_extractions = 0;

    size_t peakMemory = heap.calculateMemoryUsage();
    while (!heap.empty()) {
        auto min_elem = heap.extractMin();
        int u = min_elem.first;
        double d = min_elem.second;
        operations++;

        size_t currentMemory = heap.calculateMemoryUsage();
        if (currentMemory > peakMemory) {
            peakMemory = currentMemory;
        }

        if (d > dist[u] + 1e-9) {
            duplicate_extractions++;
            continue;
        }

        for (const auto& edge : graph[u]) {
            int v = edge.first;
            double weight = edge.second;

            double new_dist = dist[u] + weight;
            if (new_dist < dist[v] - 1e-9) {
                dist[v] = new_dist;

                if (node_ptrs[v] == nullptr) {
                    node_ptrs[v] = heap.insert(v, dist[v]);
                }
                else {
                    node_ptrs[v] = heap.decreaseKey(node_ptrs[v], dist[v]);
                }

                currentMemory = heap.calculateMemoryUsage();
                if (currentMemory > peakMemory) {
                    peakMemory = currentMemory;
                }
            }
        }

        // Mark current node as processed
        node_ptrs[u] = nullptr;
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration<double, milli>(end - start);

    metrics["total_time_ms"] = duration.count();
    metrics["insert_avg_us"] = heap.getInsertCount() > 0 ? heap.getInsertTimeMicro() / heap.getInsertCount() : 0;
    metrics["extract_min_avg_us"] = heap.getExtractMinCount() > 0 ? heap.getExtractMinTimeMicro() / heap.getExtractMinCount() : 0;
    metrics["decrease_key_avg_us"] = heap.getDecreaseKeyCount() > 0 ? heap.getDecreaseKeyTimeMicro() / heap.getDecreaseKeyCount() : 0;
    metrics["heap_height"] = heap.getMaxHeapHeightObserved();
    metrics["max_heap_height_observed"] = heap.getMaxHeapHeightObserved();
    metrics["num_trees"] = heap.getNumTrees();
    metrics["max_trees_observed"] = heap.getMaxTreesObserved();
    metrics["max_rank_observed"] = heap.getMaxRankObserved();
    metrics["total_operations"] = heap.getTotalOperations();
    metrics["insert_operations"] = heap.getInsertCount();
    metrics["extract_operations"] = heap.getExtractMinCount();
    metrics["decrease_operations"] = heap.getDecreaseKeyCount();
    metrics["max_heap_size_observed"] = heap.getMaxHeapSizeObserved();
    metrics["final_heap_size"] = heap.getSize();
    metrics["heap_memory_bytes"] = peakMemory;
    metrics["heap_memory_mb"] = peakMemory / (1024.0 * 1024.0);

    return { dist, metrics };
}

// ==========================================================
//               SHORTEST PATH FUNCTIONS
// ==========================================================

map<string, double> aggregateMetrics(const vector<map<string, double>>& allMetrics) {
    map<string, double> aggregated;
    if (allMetrics.empty()) return aggregated;

    for (const auto& metric : allMetrics[0]) {
        aggregated[metric.first] = 0.0;
    }

    // Sum all metrics
    for (const auto& metrics : allMetrics) {
        for (const auto& metric : metrics) {
            if (aggregated.find(metric.first) != aggregated.end()) {
                aggregated[metric.first] += metric.second;
            }
            else {
                aggregated[metric.first] = metric.second; // For new metrics
            }
        }
    }

    // Calculating averages
    int numSources = (int)allMetrics.size();
    for (auto& metric : aggregated) {
        metric.second /= numSources;
    }

    return aggregated;
}

void printTableHeader() { // For output
    cout << "\n" << string(120, '=') << endl;
    cout << setw(20) << left << "Heap Type"
        << setw(12) << "Ins Time(mic s)"
        << setw(12) << "Ext Time(mic s)"
        << setw(12) << "Dec Time(mic s)"
        << setw(12) << "Total(ms)"
        << setw(10) << "Height"
        << setw(10) << "Trees"
        << setw(12) << "Casc Cuts"
        << setw(12) << "Memory(KB)" << endl;
    cout << string(120, '=') << endl;
}

void printTableRow(const string& heapType, const map<string, double>& metrics) {
    cout << setw(20) << left << heapType;

    double insert_time = 0.0, extract_time = 0.0, decrease_time = 0.0;
    double total_time = 0.0;
    int height = 0, trees = 0, casc_cuts = 0;
    double memory_mb = 0.0;

    auto it = metrics.find("insert_avg_us");
    if (it != metrics.end()) insert_time = it->second;

    it = metrics.find("extract_min_avg_us");
    if (it != metrics.end()) extract_time = it->second;

    it = metrics.find("decrease_key_avg_us");
    if (it != metrics.end()) decrease_time = it->second;

    it = metrics.find("total_apsp_time_ms");
    if (it != metrics.end()) total_time = it->second;

    it = metrics.find("max_heap_height_observed");
    if (it != metrics.end()) height = (int)it->second;

    it = metrics.find("max_trees_observed");
    if (it != metrics.end()) trees = (int)it->second;

    it = metrics.find("total_cascading_cuts");
    if (it != metrics.end()) casc_cuts = (int)it->second;

    it = metrics.find("heap_memory_mb");
    if (it != metrics.end()) memory_mb = it->second;
    else {
        // Fallback to old calculation if new metric not available
        it = metrics.find("memory_mb");
        if (it != metrics.end()) memory_mb = it->second;
    }

    cout << setw(12) << fixed << setprecision(3) << insert_time
        << setw(12) << extract_time
        << setw(12) << decrease_time
        << setw(12) << total_time
        << setw(10) << height
        << setw(10) << trees;

    // Handle cascading cuts (only for Fibonacci heap)
    if (heapType == "Fibonacci_Heap") {
        cout << setw(12) << casc_cuts;
    }
    else {
        cout << setw(12) << "N/A";
    }

    cout << setw(12) << fixed << setprecision(1) << (memory_mb * 1024.0) << endl; // Show as KB
}

void runAllPairsDijkstra(const vector<vector<pair<int, double>>>& graph, const string& filename, int totalNodes, int totalEdges, const string& heapType = "Binary_Heap") {

    int n = graph.size();
    int testSources = min(20, n);

    vector<map<string, double>> allMetrics;
    auto startAll = chrono::high_resolution_clock::now();

    // Initializing all tracking variables
    double maxHeapHeightObserved = 0;
    double maxTheoreticalHeightObserved = 0;
    double maxHeapSizeObserved = 0;
    double maxTimePerSourceObserved = 0;
    double maxInsertTimeObserved = 0;
    double maxExtractMinTimeObserved = 0;
    double maxDecreaseKeyTimeObserved = 0;
    double maxTreesObserved = 0;
    double maxCascadingCutsObserved = 0;
    double maxDegreeObserved = 0;
    double maxRankObserved = 0;

    long long totalInsertCount = 0;
    long long totalExtractMinCount = 0;
    long long totalDecreaseKeyCount = 0;
    long long totalCascadingCuts = 0;
    long long totalOperations = 0;

    for (int src = 0; src < testSources; src++) {
        map<string, double> resultMetrics;

        if (heapType == "Binary_Heap") {
            auto result = dijkstra_binaryHeap(graph, src);
            resultMetrics = result.second;
        }
        else if (heapType == "Fibonacci_Heap") {
            auto result = dijkstra_fibonacciHeap(graph, src);
            resultMetrics = result.second;
        }
        else if (heapType == "Hollow_Heap") {
            auto result = dijkstra_hollowHeap(graph, src);
            // Convert unordered_map to map for consistency
            for (const auto& metric : result.second) {
                resultMetrics[metric.first] = metric.second;
            }
        }

        allMetrics.push_back(resultMetrics);

        // Tracking max operation times for ALL heap types
        if (resultMetrics.find("insert_avg_us") != resultMetrics.end() &&
            resultMetrics["insert_avg_us"] > maxInsertTimeObserved) {
            maxInsertTimeObserved = resultMetrics["insert_avg_us"];
        }
        if (resultMetrics.find("extract_min_avg_us") != resultMetrics.end() &&
            resultMetrics["extract_min_avg_us"] > maxExtractMinTimeObserved) {
            maxExtractMinTimeObserved = resultMetrics["extract_min_avg_us"];
        }
        if (resultMetrics.find("decrease_key_avg_us") != resultMetrics.end() &&
            resultMetrics["decrease_key_avg_us"] > maxDecreaseKeyTimeObserved) {
            maxDecreaseKeyTimeObserved = resultMetrics["decrease_key_avg_us"];
        }

        // For Hollow Heap, using different metric names
        if (heapType == "Hollow_Heap") {
            if (resultMetrics.find("insert_operations") != resultMetrics.end()) {
                totalInsertCount += (long long)resultMetrics["insert_operations"];
            }
            if (resultMetrics.find("extract_operations") != resultMetrics.end()) {
                totalExtractMinCount += (long long)resultMetrics["extract_operations"];
            }
            if (resultMetrics.find("decrease_operations") != resultMetrics.end()) {
                totalDecreaseKeyCount += (long long)resultMetrics["decrease_operations"];
            }
            if (resultMetrics.find("total_operations") != resultMetrics.end()) {
                totalOperations += (long long)resultMetrics["total_operations"];
            }

            if (resultMetrics.find("heap_height") != resultMetrics.end() &&
                resultMetrics["heap_height"] > maxHeapHeightObserved) {
                maxHeapHeightObserved = resultMetrics["heap_height"];
            }
            if (resultMetrics.find("max_rank_observed") != resultMetrics.end() &&
                resultMetrics["max_rank_observed"] > maxRankObserved) {
                maxRankObserved = resultMetrics["max_rank_observed"];
            }
            if (resultMetrics.find("num_trees") != resultMetrics.end() &&
                resultMetrics["num_trees"] > maxTreesObserved) {
                maxTreesObserved = resultMetrics["num_trees"];
            }
        }
        else {
            // For Binary and Fibonacci heaps
            if (resultMetrics.find("insert_count") != resultMetrics.end()) {
                totalInsertCount += (long long)resultMetrics["insert_count"];
            }
            if (resultMetrics.find("extract_min_count") != resultMetrics.end()) {
                totalExtractMinCount += (long long)resultMetrics["extract_min_count"];
            }
            if (resultMetrics.find("decrease_key_count") != resultMetrics.end()) {
                totalDecreaseKeyCount += (long long)resultMetrics["decrease_key_count"];
            }
            totalOperations = totalInsertCount + totalExtractMinCount + totalDecreaseKeyCount;

            if (resultMetrics.find("heap_height") != resultMetrics.end() &&
                resultMetrics["heap_height"] > maxHeapHeightObserved) {
                maxHeapHeightObserved = resultMetrics["heap_height"];
            }
            if (heapType == "Binary_Heap" && resultMetrics.find("theoretical_heap_height") != resultMetrics.end() &&
                resultMetrics["theoretical_heap_height"] > maxTheoreticalHeightObserved) {
                maxTheoreticalHeightObserved = resultMetrics["theoretical_heap_height"];
            }
            if (resultMetrics.find("max_heap_size") != resultMetrics.end() &&
                resultMetrics["max_heap_size"] > maxHeapSizeObserved) {
                maxHeapSizeObserved = resultMetrics["max_heap_size"];
            }
            if (resultMetrics.find("max_trees_observed") != resultMetrics.end() &&
                resultMetrics["max_trees_observed"] > maxTreesObserved) {
                maxTreesObserved = resultMetrics["max_trees_observed"];
            }
            if (resultMetrics.find("cascading_cuts") != resultMetrics.end() &&
                resultMetrics["cascading_cuts"] > maxCascadingCutsObserved) {
                maxCascadingCutsObserved = resultMetrics["cascading_cuts"];
            }
            if (resultMetrics.find("max_degree_observed") != resultMetrics.end() &&
                resultMetrics["max_degree_observed"] > maxDegreeObserved) {
                maxDegreeObserved = resultMetrics["max_degree_observed"];
            }
            if (resultMetrics.find("cascading_cuts") != resultMetrics.end()) {
                totalCascadingCuts += (long long)resultMetrics["cascading_cuts"];
            }
        }

        if (resultMetrics.find("total_time_ms") != resultMetrics.end() &&
            resultMetrics["total_time_ms"] > maxTimePerSourceObserved) {
            maxTimePerSourceObserved = resultMetrics["total_time_ms"];
        }
    }

    auto endAll = chrono::high_resolution_clock::now();
    auto totalTime = chrono::duration<double, milli>(endAll - startAll);

    map<string, double> avgMetrics = aggregateMetrics(allMetrics);
    avgMetrics["total_apsp_time_ms"] = totalTime.count();
    avgMetrics["sources_tested"] = testSources;
    avgMetrics["time_per_source_ms"] = totalTime.count() / testSources;

    // Common metrics
    avgMetrics["max_time_per_source_observed"] = maxTimePerSourceObserved;
    avgMetrics["total_insert_count"] = (double)totalInsertCount;
    avgMetrics["total_extract_min_count"] = (double)totalExtractMinCount;
    avgMetrics["total_decrease_key_count"] = (double)totalDecreaseKeyCount;
    avgMetrics["total_operations"] = (double)totalOperations;
    avgMetrics["max_insert_time_observed"] = maxInsertTimeObserved;
    avgMetrics["max_extract_min_time_observed"] = maxExtractMinTimeObserved;
    avgMetrics["max_decrease_key_time_observed"] = maxDecreaseKeyTimeObserved;

    // Heap-specific metrics
    if (heapType == "Hollow_Heap") {
        avgMetrics["max_heap_height_observed"] = maxHeapHeightObserved;
        avgMetrics["max_rank_observed"] = maxRankObserved;
        avgMetrics["max_trees_observed"] = maxTreesObserved;
        avgMetrics["max_cascading_cuts_observed"] = 0; // Hollow heaps don't have cascading cuts
        avgMetrics["total_cascading_cuts"] = 0; // Hollow heaps don't have cascading cuts
    }
    else {
        avgMetrics["max_heap_height_observed"] = maxHeapHeightObserved;
        avgMetrics["max_theoretical_height_observed"] = maxTheoreticalHeightObserved;
        avgMetrics["max_heap_size_observed"] = maxHeapSizeObserved;
        avgMetrics["max_trees_observed"] = maxTreesObserved;
        avgMetrics["max_cascading_cuts_observed"] = maxCascadingCutsObserved;
        avgMetrics["max_degree_observed"] = maxDegreeObserved;
        avgMetrics["total_cascading_cuts"] = (double)totalCascadingCuts;
    }

    // Calculate memory usage
    size_t graphMemory = sizeof(graph) + (graph.capacity() * sizeof(vector<pair<int, double>>));
    for (const auto& adjList : graph) {
        graphMemory += adjList.capacity() * sizeof(pair<int, double>);
    }
    avgMetrics["memory_mb"] = graphMemory / (1024.0 * 1024.0);

    // Print table row
    printTableRow(heapType, avgMetrics);

    // Save to CSV file
    ofstream file(filename.c_str(), ios::app);
    if (file.is_open()) {
        file.seekp(0, ios::end);
        if (file.tellp() == 0) {
            file << "Heap_Type,Nodes,Edges,Sources_Tested,"
                << "Total_Insert_Count,Total_Extract_Min_Count,Total_Decrease_Key_Count,Total_Cascading_Cuts,"
                << "Avg_Insert_Time_μs,Avg_Extract_Min_Time_μs,Avg_Decrease_Key_Time_μs,"
                << "Max_Insert_Time_μs,Max_Extract_Min_Time_μs,Max_Decrease_Key_Time_μs,"
                << "Total_APSP_Time_ms,Avg_Time_Per_Source_ms,Max_Time_Per_Source_ms,"
                << "Avg_Actual_Height,Max_Actual_Height,Avg_Theoretical_Height,Max_Theoretical_Height,"
                << "Avg_Max_Heap_Size,Max_Heap_Size,Num_Trees,Max_Trees_Observed,Max_Degree_Observed,Max_Rank_Observed,Memory_MB,Platform\n";
        }

        file << heapType << ","
            << totalNodes << "," << totalEdges << ","
            << testSources << ","
            << (long long)avgMetrics["total_insert_count"] << ","
            << (long long)avgMetrics["total_extract_min_count"] << ","
            << (long long)avgMetrics["total_decrease_key_count"] << ","
            << (long long)avgMetrics["total_cascading_cuts"] << ",";

        if (heapType == "Hollow_Heap") {
            file << "0,0,0,0,0,0,"; // Placeholder for operation times
        }
        else {
            file << avgMetrics["insert_avg_us"] << ","
                << avgMetrics["extract_min_avg_us"] << ","
                << avgMetrics["decrease_key_avg_us"] << ","
                << avgMetrics["max_insert_time_observed"] << ","
                << avgMetrics["max_extract_min_time_observed"] << ","
                << avgMetrics["max_decrease_key_time_observed"] << ",";
        }

        file << avgMetrics["total_apsp_time_ms"] << ","
            << avgMetrics["time_per_source_ms"] << ","
            << avgMetrics["max_time_per_source_observed"] << ","
            << avgMetrics["heap_height"] << ","
            << avgMetrics["max_heap_height_observed"] << ",";

        if (heapType == "Binary_Heap") {
            file << avgMetrics["theoretical_heap_height"] << ","
                << avgMetrics["max_theoretical_height_observed"] << ",";
        }
        else {
            file << "0,0,"; // Placeholder for theoretical height
        }

        if (heapType == "Hollow_Heap") {
            file << "0," // Avg_Max_Heap_Size
                << "0," // Max_Heap_Size
                << avgMetrics["num_trees"] << ","
                << avgMetrics["max_trees_observed"] << ","
                << "0," // Max_Degree_Observed
                << avgMetrics["max_rank_observed"] << ",";
        }
        else {
            file << avgMetrics["max_heap_size"] << ","
                << avgMetrics["max_heap_size_observed"] << ","
                << avgMetrics["num_trees"] << ","
                << avgMetrics["max_trees_observed"] << ","
                << avgMetrics["max_degree_observed"] << ","
                << "0,"; // Placeholder for max_rank_observed
        }

        file << avgMetrics["memory_mb"] << ","
            << "Local\n";

        file.close();
    }
}

// ==========================================================
//               EXPERIMENT B - OPERATION PROFILING
// ==========================================================
enum OperationType { INSERT, EXTRACT_MIN, DECREASE_KEY };

vector<OperationType> generateRandomOperations(int count, double insertProb = 0.4, double extractProb = 0.3, double decreaseProb = 0.3) {
    vector<OperationType> operations;
    operations.reserve(count);

    for (int i = 0; i < count; i++) {
        double randVal = (double)rand() / RAND_MAX;
        if (randVal < insertProb) {
            operations.push_back(INSERT);
        }
        else if (randVal < insertProb + extractProb) {
            operations.push_back(EXTRACT_MIN);
        }
        else {
            operations.push_back(DECREASE_KEY);
        }
    }
    return operations;
}

void runOperationProfiling(const string& heapType) {
    cout << "\n=== EXPERIMENT B: Operation Profiling for " << heapType << " ===\n";

    const int NUM_OPERATIONS = 100000;
    auto operations = generateRandomOperations(NUM_OPERATIONS);


    vector<int> nodeHandles;
    unordered_map<int, double> currentValues;
    unordered_map<int, void*> nodePtrs;

    if (heapType == "Binary_Heap") {
        BinaryHeap heap(1000000);

        auto start = chrono::high_resolution_clock::now();

        for (int i = 0; i < NUM_OPERATIONS; i++) {
            switch (operations[i]) {
            case INSERT: {
                int key = rand() % 1000000;
                double value = (double)rand() / RAND_MAX * 1000.0;
                heap.insert(key, value);
                nodeHandles.push_back(key);
                currentValues[key] = value;
                break;
            }
            case EXTRACT_MIN: {
                if (!heap.empty()) {
                    auto minNode = heap.extractMin();

                    auto it = find(nodeHandles.begin(), nodeHandles.end(), minNode.vertex);
                    if (it != nodeHandles.end()) {
                        nodeHandles.erase(it);
                    }
                    currentValues.erase(minNode.vertex);
                }
                break;
            }
            case DECREASE_KEY: {
                if (!nodeHandles.empty()) {
                    int randomIdx = rand() % nodeHandles.size();
                    int key = nodeHandles[randomIdx];
                    if (heap.pos[key] != -1) {

                        double currentVal = currentValues[key];
                        double newValue = currentVal * (0.1 + 0.8 * ((double)rand() / RAND_MAX));
                        heap.decreaseKey(key, newValue);
                        currentValues[key] = newValue;
                    }
                }
                break;
            }
            }
        }

        auto end = chrono::high_resolution_clock::now();
        auto totalTime = chrono::duration<double, milli>(end - start);

        cout << "Total time: " << totalTime.count() / 1000.0 << " ms\n";
        cout << "Time per operation: " << totalTime.count() / NUM_OPERATIONS << " ms\n";
        cout << "Insert operations: " << heap.getInsertCount() << "\n";
        cout << "Extract-min operations: " << heap.getExtractMinCount() << "\n";
        cout << "Decrease-key operations: " << heap.getDecreaseKeyCount() << "\n";

    }
    else if (heapType == "Fibonacci_Heap") {
        FibonacciHeap heap;

        auto start = chrono::high_resolution_clock::now();

        for (int i = 0; i < NUM_OPERATIONS; i++) {
            switch (operations[i]) {
            case INSERT: {
                int key = rand() % 1000000;
                double value = (double)rand() / RAND_MAX * 1000.0;
                heap.insert(key, value);
                nodeHandles.push_back(key);
                currentValues[key] = value;
                break;
            }
            case EXTRACT_MIN: {
                if (!heap.empty()) {
                    auto minPair = heap.extractMin();
                    // Remove from tracking
                    auto it = find(nodeHandles.begin(), nodeHandles.end(), minPair.first);
                    if (it != nodeHandles.end()) {
                        nodeHandles.erase(it);
                    }
                    currentValues.erase(minPair.first);
                }
                break;
            }
            case DECREASE_KEY: {
                if (!nodeHandles.empty()) {
                    int randomIdx = rand() % nodeHandles.size();
                    int key = nodeHandles[randomIdx];
                    if (heap.contains(key)) {

                        double currentVal = currentValues[key];
                        double newValue = currentVal * (0.1 + 0.8 * ((double)rand() / RAND_MAX));
                        heap.decreaseKey(key, newValue);
                        currentValues[key] = newValue;
                    }
                }
                break;
            }
            }
        }

        auto end = chrono::high_resolution_clock::now();
        auto totalTime = chrono::duration<double, milli>(end - start);

        cout << "Total time: " << totalTime.count() / 1000.0 << " ms\n";
        cout << "Time per operation: " << totalTime.count() / NUM_OPERATIONS << " ms\n";
        cout << "Insert operations: " << heap.getInsertCount() << "\n";
        cout << "Extract-min operations: " << heap.getExtractMinCount() << "\n";
        cout << "Decrease-key operations: " << heap.getDecreaseKeyCount() << "\n";
        cout << "Cascading cuts: " << heap.getCascadingCuts() << "\n";

    }
    else if (heapType == "Hollow_Heap") {
        HollowHeap heap;

        auto start = chrono::high_resolution_clock::now();

        for (int i = 0; i < NUM_OPERATIONS; i++) {
            switch (operations[i]) {
            case INSERT: {
                int key = rand() % 1000000;
                double value = (double)rand() / RAND_MAX * 1000.0;
                auto nodePtr = heap.insert(key, value);
                nodePtrs[key] = nodePtr;
                nodeHandles.push_back(key);
                currentValues[key] = value;
                break;
            }
            case EXTRACT_MIN: {
                if (!heap.empty()) {
                    auto minPair = heap.extractMin();

                    auto it = find(nodeHandles.begin(), nodeHandles.end(), minPair.first);
                    if (it != nodeHandles.end()) {
                        nodeHandles.erase(it);
                    }
                    nodePtrs.erase(minPair.first);
                    currentValues.erase(minPair.first);
                }
                break;
            }
            case DECREASE_KEY: {
                if (!nodeHandles.empty()) {
                    int randomIdx = rand() % nodeHandles.size();
                    int key = nodeHandles[randomIdx];
                    if (nodePtrs.count(key)) {

                        double currentVal = currentValues[key];
                        double newValue = currentVal * (0.1 + 0.8 * ((double)rand() / RAND_MAX));
                        auto oldPtr = static_cast<HollowHeap::Node*>(nodePtrs[key]);
                        auto newPtr = heap.decreaseKey(oldPtr, newValue);
                        nodePtrs[key] = newPtr;
                        currentValues[key] = newValue;
                    }
                }
                break;
            }
            }
        }

        auto end = chrono::high_resolution_clock::now();
        auto totalTime = chrono::duration<double, milli>(end - start);

        cout << "Total time: " << totalTime.count() / 1000.0 << " ms\n";
        cout << "Time per operation: " << totalTime.count() / NUM_OPERATIONS << " ms\n";
        cout << "Insert operations: " << heap.getInsertCount() << "\n";
        cout << "Extract-min operations: " << heap.getExtractMinCount() << "\n";
        cout << "Decrease-key operations: " << heap.getDecreaseKeyCount() << "\n";
    }
}

void runExperimentB() {
    cout << "\n" << string(60, '=') << endl;
    cout << "EXPERIMENT B: OPERATION PROFILING (100,000 operations)" << endl;
    cout << string(60, '=') << endl;

    runOperationProfiling("Binary_Heap");
    runOperationProfiling("Fibonacci_Heap");
    runOperationProfiling("Hollow_Heap");
}

// ==========================================================
//                  TEST SUITE IMPLEMENTATION
// ==========================================================

class HeapTestSuite {
private:
    random_device rd;
    mt19937 gen;
    uniform_real_distribution<double> value_dist;

public:
    HeapTestSuite() : gen(rd()), value_dist(0.0, 1000.0) {}

    void testHollowHeapBasic() {
        cout << "Testing Hollow Heap Basic Operations..." << endl;

        HollowHeap heap;

        // Test 1: Basic insert and extract
        auto node1 = heap.insert(1, 5.0);
        auto min1 = heap.extractMin();
        assert(min1.first == 1);
        assert(min1.second == 5.0);
        assert(heap.empty());
        cout << "Single insert/extract" << endl;

        // Test 2: Multiple elements in correct order
        heap.insert(3, 15.0);
        heap.insert(2, 5.0);
        heap.insert(4, 10.0);

        auto min2 = heap.extractMin();
        assert(min2.first == 2 && min2.second == 5.0);
        auto min3 = heap.extractMin();
        assert(min3.first == 4 && min3.second == 10.0);
        auto min4 = heap.extractMin();
        assert(min4.first == 3 && min4.second == 15.0);
        cout << "Multiple elements in order" << endl;

        // Test 3: Decrease key - CORRECTED EXPECTATIONS
        auto node5 = heap.insert(5, 20.0);
        auto node6 = heap.insert(6, 25.0);
        heap.decreaseKey(node5, 3.0);

        assert(!heap.empty());

        auto min5 = heap.extractMin();
        assert(min5.first == 5 && min5.second == 3.0);

        assert(!heap.empty());

        auto min6 = heap.extractMin();
        assert(min6.first == 6 && min6.second == 25.0);

        // the heap should be empty
        assert(heap.empty());

        cout << "Decrease key works correctly" << endl;

        // Test 4: Final verification with sentinel values
        auto finalResult = heap.extractMin();
        assert(finalResult.first == -1 && finalResult.second == -1);
        cout << "Empty heap returns proper sentinel values" << endl;
    }

    void testFibonacciHeapBasic() {
        cout << "\nTesting Fibonacci Heap Basic Operations..." << endl;

        FibonacciHeap heap;

        // Test 1: Basic insert and extract
        heap.insert(1, 5.0);
        auto min1 = heap.extractMin();
        assert(min1.first == 1 && min1.second == 5.0);
        assert(heap.empty());
        cout << "Single insert/extract" << endl;

        // Test 2: Multiple elements
        heap.insert(3, 15.0);
        heap.insert(2, 5.0);
        heap.insert(4, 10.0);

        auto min2 = heap.extractMin();
        assert(min2.first == 2 && min2.second == 5.0);
        auto min3 = heap.extractMin();
        assert(min3.first == 4 && min3.second == 10.0);
        auto min4 = heap.extractMin();
        assert(min4.first == 3 && min4.second == 15.0);
        cout << "Multiple elements in order" << endl;

        // Test 3: Decrease key
        heap.insert(5, 20.0);
        heap.insert(6, 25.0);
        heap.decreaseKey(5, 3.0);

        auto min5 = heap.extractMin();
        assert(min5.first == 5 && min5.second == 3.0);
        cout << "Decrease key works" << endl;
    }

    void testBinaryHeapBasic() {
        cout << "\n\nTesting Binary Heap Basic Operations..." << endl;

        BinaryHeap heap(100);

        // Test 1: Basic insert and extract
        heap.insert(1, 5.0);
        auto min1 = heap.extractMin();
        assert(min1.vertex == 1 && min1.dist == 5.0);
        assert(heap.empty());
        cout << "Single insert/extract" << endl;

        // Test 2: Multiple elements
        heap.insert(3, 15.0);
        heap.insert(2, 5.0);
        heap.insert(4, 10.0);

        auto min2 = heap.extractMin();
        assert(min2.vertex == 2 && min2.dist == 5.0);
        auto min3 = heap.extractMin();
        assert(min3.vertex == 4 && min3.dist == 10.0);
        auto min4 = heap.extractMin();
        assert(min4.vertex == 3 && min4.dist == 15.0);
        cout << "Multiple elements in order" << endl;

        // Test 3: Decrease key
        heap.insert(5, 20.0);
        heap.insert(6, 25.0);
        heap.decreaseKey(5, 3.0);

        auto min5 = heap.extractMin();
        assert(min5.vertex == 5 && min5.dist == 3.0);
        cout << "Decrease key works" << endl;
    }

    void testAllHeapsComplex() {
        cout << "Testing Complex Operations on All Heaps..." << endl;

        // Test Hollow Heap with complex operations
        {
            HollowHeap heap;
            vector<HollowHeap::Node*> nodes;

            // Insert 10 elements
            for (int i = 0; i < 10; i++) {
                nodes.push_back(heap.insert(i, 10 - i));
            }

            // Decrease some keys
            heap.decreaseKey(nodes[5], 0.5);
            heap.decreaseKey(nodes[2], 0.2);

            // Extract and verify order
            auto min1 = heap.extractMin();
            assert(min1.first == 2 && min1.second == 0.2);

            auto min2 = heap.extractMin();
            assert(min2.first == 5 && min2.second == 0.5);

            cout << "Hollow Heap complex operations" << endl;
        }

        // Test Fibonacci Heap with complex operations
        {
            FibonacciHeap heap;

            for (int i = 0; i < 10; i++) {
                heap.insert(i, 10 - i);
            }

            heap.decreaseKey(5, 0.5);
            heap.decreaseKey(2, 0.2);

            auto min1 = heap.extractMin();
            assert(min1.first == 2 && min1.second == 0.2);

            auto min2 = heap.extractMin();
            assert(min2.first == 5 && min2.second == 0.5);

            cout << "Fibonacci Heap complex operations" << endl;
        }

        // Test Binary Heap with complex operations
        {
            BinaryHeap heap(100);

            for (int i = 0; i < 10; i++) {
                heap.insert(i, 10 - i);
            }

            heap.decreaseKey(5, 0.5);
            heap.decreaseKey(2, 0.2);

            auto min1 = heap.extractMin();
            assert(min1.vertex == 2 && min1.dist == 0.2);

            auto min2 = heap.extractMin();
            assert(min2.vertex == 5 && min2.dist == 0.5);

            cout << "Binary Heap complex operations" << endl;
        }
    }

    void runPerformanceComparison(int numOperations = 10000) {
        cout << "\n" << string(60, '=') << endl;
        cout << "PERFORMANCE COMPARISON (" << numOperations << " operations)" << endl;
        cout << string(60, '=') << endl;

        cout << setw(15) << "Heap Type"
            << setw(15) << "Time (ms)"
            << setw(15) << "Memory (KB)"
            << setw(15) << "Ops/sec" << endl;
        cout << string(60, '-') << endl;

        // Test Binary Heap
        auto start = chrono::high_resolution_clock::now();
        BinaryHeap bh(numOperations * 2);
        for (int i = 0; i < numOperations; i++) {
            bh.insert(i, value_dist(gen));
        }
        for (int i = 0; i < numOperations; i++) {
            bh.extractMin();
        }
        auto end = chrono::high_resolution_clock::now();
        double bhTime = chrono::duration<double, milli>(end - start).count();
        size_t bhMemory = bh.calculateMemoryUsage();
        cout << setw(15) << "Binary_Heap"
            << setw(15) << fixed << setprecision(2) << bhTime
            << setw(15) << bhMemory / 1024.0
            << setw(15) << (int)(numOperations / (bhTime / 1000.0)) << endl;

        // Test Fibonacci Heap
        start = chrono::high_resolution_clock::now();
        FibonacciHeap fh;
        for (int i = 0; i < numOperations; i++) {
            fh.insert(i, value_dist(gen));
        }
        for (int i = 0; i < numOperations; i++) {
            fh.extractMin();
        }
        end = chrono::high_resolution_clock::now();
        double fhTime = chrono::duration<double, milli>(end - start).count();
        size_t fhMemory = fh.calculateMemoryUsage();
        cout << setw(15) << "Fibonacci_Heap"
            << setw(15) << fhTime
            << setw(15) << fhMemory / 1024.0
            << setw(15) << (int)(numOperations / (fhTime / 1000.0)) << endl;

        // Test Hollow Heap
        start = chrono::high_resolution_clock::now();
        HollowHeap hh;
        for (int i = 0; i < numOperations; i++) {
            hh.insert(i, value_dist(gen));
        }
        for (int i = 0; i < numOperations; i++) {
            hh.extractMin();
        }
        end = chrono::high_resolution_clock::now();
        double hhTime = chrono::duration<double, milli>(end - start).count();
        size_t hhMemory = hh.calculateMemoryUsage();
        cout << setw(15) << "Hollow_Heap"
            << setw(15) << hhTime
            << setw(15) << hhMemory / 1024.0
            << setw(15) << (int)(numOperations / (hhTime / 1000.0)) << endl;
    }

    void runAllTests() {
        cout << "HEAP CORRECTNESS TEST SUITE" << endl;
        cout << string(50, '=') << endl;

        testHollowHeapBasic();
        testFibonacciHeapBasic();
        testBinaryHeapBasic();
        testAllHeapsComplex();

        cout << "\nALL CORRECTNESS TESTS PASSED SUCCESSFULLY!" << endl;
    }
};

void runTestSuite() {
    HeapTestSuite testSuite;

    cout << "\n" << string(60, '=') << endl;
    cout << "HEAP TEST SUITE" << endl;
    cout << string(60, '=') << endl;

    cout << "Select test mode:" << endl;
    cout << "1. Run all correctness tests" << endl;
    cout << "2. Run performance comparison" << endl;
    cout << "3. Run both correctness and performance" << endl;
    cout << "Enter choice (1-3): ";

    int choice;
    cin >> choice;

    switch (choice) {
    case 1:
        testSuite.runAllTests();
        break;
    case 2:
        testSuite.runPerformanceComparison(10000);
        break;
    case 3:
        testSuite.runAllTests();
        testSuite.runPerformanceComparison(5000);
        break;
    default:
        cout << "Invalid choice. Running all tests." << endl;
        testSuite.runAllTests();
        testSuite.runPerformanceComparison(5000);
    }
}

// ==========================================================
//                           MAIN
// ==========================================================

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cout << "SELECT MODE:" << endl;
    cout << "1. Run Experiments (Dijkstra on graphs)" << endl;
    cout << "2. Run Test Suite (Heap correctness tests)" << endl;
    cout << "Enter choice (1 or 2): ";

    int mainChoice;
    cin >> mainChoice;

    if (mainChoice == 1) {
        cout << "Enter dataset filename: ";
        string file;
        cin >> file;

        int maxNode = 0, edgeCount = 0;
        cout << "Loading graph..." << endl;
        vector<vector<pair<int, double>>> graph = loadGraph(file, maxNode, edgeCount);
        int nodeCount = (int)graph.size();

        cout << "Graph loaded:" << endl;
        cout << "  Nodes: " << nodeCount << " (max node ID: " << maxNode << ")" << endl;
        cout << "  Edges: " << edgeCount << endl;
        cout << "  Density: " << (double)edgeCount / nodeCount << " edges per node" << endl;

        int totalDegree = 0;
        for (int i = 0; i < nodeCount; i++) {
            totalDegree += (int)graph[i].size();
        }
        cout << "  Actual average degree: " << (double)totalDegree / nodeCount << endl;

        cout << "\nSelect heap type:" << endl;
        cout << "1. Binary Heap" << endl;
        cout << "2. Fibonacci Heap" << endl;
        cout << "3. Hollow Heap" << endl;
        cout << "4. All Three (Comparative Analysis)" << endl;
        cout << "5. Experiment B - Operation Profiling" << endl;
        cout << "Enter choice (1-5): ";

        int choice;
        cin >> choice;

        string outputFile = "apsp_performance_results.csv";

        if (choice == 4) {
            printTableHeader();
        }

        if (choice == 1) {
            printTableHeader();
            runAllPairsDijkstra(graph, outputFile, nodeCount, edgeCount, "Binary_Heap");
        }
        else if (choice == 2) {
            printTableHeader();
            runAllPairsDijkstra(graph, outputFile, nodeCount, edgeCount, "Fibonacci_Heap");
        }
        else if (choice == 3) {
            printTableHeader();
            runAllPairsDijkstra(graph, outputFile, nodeCount, edgeCount, "Hollow_Heap");
        }
        else if (choice == 4) {
            runAllPairsDijkstra(graph, outputFile, nodeCount, edgeCount, "Binary_Heap");
            runAllPairsDijkstra(graph, outputFile, nodeCount, edgeCount, "Fibonacci_Heap");
            runAllPairsDijkstra(graph, outputFile, nodeCount, edgeCount, "Hollow_Heap");
            cout << string(120, '=') << endl;
        }
        else if (choice == 5) {
            runExperimentB();
        }
        else {
            cout << "Invalid choice. Using Binary Heap by default." << endl;
            printTableHeader();
            runAllPairsDijkstra(graph, outputFile, nodeCount, edgeCount, "Binary_Heap");
        }

        cout << "\nResults saved to: " << outputFile << endl;

    }
    else if (mainChoice == 2) {
        runTestSuite();
    }
    else {
        cout << "Invalid choice. Running experiments mode." << endl;
    }

    return 0;
}