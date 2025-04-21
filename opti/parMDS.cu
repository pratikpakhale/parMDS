// parMDS.cu
//~~~START:Thu, 16-Jun-2022, 12:43:32 IST
// For GECCO'23 Submission.
// nvcc -O3 -std=c++14 parMDS.cu -o parMDS.out && time ./parMDS.out toy.vrp 32

/*
 * Rajesh Pandian M | https://mrprajesh.co.in
 * Somesh Singh     | https://ssomesh.github.io
 * Rupesh Nasre     | www.cse.iitm.ac.in/~rupesh
 * N.S.Narayanaswamy| www.cse.iitm.ac.in/~swamy
 * MIT LICENSE
 */

#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cmath>    // for sqrt, round


using namespace std;

//~ Define types
using point_t  = double;
using weight_t = double;
using demand_t = double;
using node_t   = int;  // nodes ids: 0 to n-1

const node_t DEPOT = 0; // depot is always assumed to be zero.

// To hold all command line parameters in one struct.
class Params {
public:
  Params() : toRound(true), nThreads(20) {} // default: round (true) and 20 threads
  bool toRound;
  short nThreads;
};

// The Point class; note that on the host we use std::vector<Point> and on device we copy a simple POD.
struct Point {
  point_t x;
  point_t y;
  demand_t demand;
};

// (Host) edge class for the graph.
class Edge {
public:
  node_t to;
  weight_t length;
  Edge() : to(0), length(0) {}
  Edge(node_t t, weight_t l) : to(t), length(l) {}
  bool operator<(const Edge &e) const {
    return length < e.length;
  }
};

// VRP class holds the instance details.
class VRP {
public:
  size_t size;
  demand_t capacity;
  string type;

  vector<Point> node;            // List of nodes
  vector<weight_t> dist;         // Triangular array of distances (n*(n-1)/2 elements)
  Params params;

  VRP() : size(0), capacity(0) {}

  unsigned read(string filename);
  void print();
  void print_dist();

  // This routine computes distances between every pair of nodes.
  // The heavy inner loop is offloaded to a CUDA kernel.
  std::vector<std::vector<Edge>> cal_graph_dist();

  // Return distance between nodes i and j based on triangular index.
  weight_t get_dist(node_t i, node_t j) const {
    if (i == j)
      return 0.0;
    node_t a = i, b = j;
    if (a > b) std::swap(a, b);
    // Mapping: index = a*size - a*(a+1)/2 + (b - a - 1)
    size_t index = a * size - (a * (a + 1)) / 2 + (b - a - 1);
    return dist[index];
  }

  size_t getSize() const { return size; }
  demand_t getCapacity() const { return capacity; }
};

// ---------------------------------------------------------------------
// Device–side version of Point (a lightweight POD for the CUDA kernel).
struct DevicePoint {
  point_t x;
  point_t y;
  demand_t demand; // not used in the kernel, but kept for completeness.
};

template <bool toRound>
__global__ void computeDistancesKernel2D(const DevicePoint *d_nodes, double *d_dist, size_t size) {
  const int TILE = 32;
  // Compute global indices: i, j
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int i = bx * TILE + tx;
  int j = by * TILE + ty;

  // Only work on valid indices with i<j.
  // For blocks along the diagonal, enforce tx < ty.
  if (i < size && j < size && (bx < by || (bx == by && tx < ty))) {
    // Allocate shared memory for the block.
    __shared__ double shX_i[TILE];
    __shared__ double shY_i[TILE];
    __shared__ double shX_j[TILE];
    __shared__ double shY_j[TILE];

    // Each block loads the i-th values (from the row of the tile) and
    // the j-th values (from the column). (We use one thread per element in the row/column.)
    if (ty == 0 && i < size) {
      shX_i[tx] = d_nodes[i].x;
      shY_i[tx] = d_nodes[i].y;
    }
    if (tx == 0 && j < size) {
      shX_j[ty] = d_nodes[j].x;
      shY_j[ty] = d_nodes[j].y;
    }
    __syncthreads();

    // Now each thread computes the distance between point i and point j
    double dx = shX_i[tx] - shX_j[ty];
    double dy = shY_i[tx] - shY_j[ty];
    double dval = sqrt(dx * dx + dy * dy);
    if (toRound)
      dval = round(dval);

    // The complete graph distances are stored in a 1D array.
    // With mapping: linearIndex = i*size - (i*(i+1))/2 + (j - i - 1)
    size_t linearIndex = i * size - (i * (i + 1)) / 2 + (j - i - 1);
    d_dist[linearIndex] = dval;
  }
}
std::vector<std::vector<Edge>> VRP::cal_graph_dist() {
  size_t totalPairs = (size * (size - 1)) / 2;
  dist.resize(totalPairs);

  // Allocate device memory for the nodes and the distance array.
  DevicePoint *d_nodes = nullptr;
  double *d_dist = nullptr;
  cudaMalloc(&d_nodes, size * sizeof(DevicePoint));
  cudaMalloc(&d_dist, totalPairs * sizeof(double));

  // Copy our node data into a device-friendly array.
  DevicePoint *h_dnodes = new DevicePoint[size];
  for (size_t i = 0; i < size; i++) {
    h_dnodes[i].x      = node[i].x;
    h_dnodes[i].y      = node[i].y;
    h_dnodes[i].demand = node[i].demand;
  }
  cudaMemcpy(d_nodes, h_dnodes, size * sizeof(DevicePoint), cudaMemcpyHostToDevice);
  delete[] h_dnodes;

  // Choose a 2D block: TILE x TILE threads.
  const int TILE = 32;
  dim3 blockDim(TILE, TILE);
  // Grid dimensions based on the number of points, note we cover whole square [0,size)x[0,size)
  dim3 gridDim((size + TILE - 1) / TILE, (size + TILE - 1) / TILE);

  // Launch the kernel with compile-time constant for rounding.
  if (params.toRound) {
    computeDistancesKernel2D<true><<<gridDim, blockDim>>>(d_nodes, d_dist, size);
  } else {
    computeDistancesKernel2D<false><<<gridDim, blockDim>>>(d_nodes, d_dist, size);
  }
  cudaDeviceSynchronize();

  // Copy computed distances back to the host.
  cudaMemcpy(dist.data(), d_dist, totalPairs * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_nodes);
  cudaFree(d_dist);

  // Rebuild the complete graph as an adjacency list.
  std::vector<std::vector<Edge>> nG(size);
  size_t k = 0;
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = i + 1; j < size; ++j) {
      double w = dist[k];
      nG[i].push_back(Edge(j, w));
      nG[j].push_back(Edge(i, w));
      k++;
    }
  }
  return nG;
}
// ---------------------------------------------------------------------
// Read the input .vrp file.
unsigned VRP::read(string filename) {
  ifstream in(filename);
  if (!in.is_open()) {
    std::cerr << "Could not open the file \"" << filename << "\"" << std::endl;
    exit(1);
  }
  string line;
  for (int i = 0; i < 3; ++i)
    getline(in, line);

  // DIMENSION
  getline(in, line);
  size = stof(line.substr(line.find(":") + 2));

  // DISTANCE TYPE (not further used)
  getline(in, line);
  type = line;

  // CAPACITY
  getline(in, line);
  capacity = stof(line.substr(line.find(":") + 2));

  // Skip NODE_COORD_SECTION header.
  getline(in, line);

  node.resize(size);
  for (size_t i = 0; i < size; ++i) {
    getline(in, line);
    stringstream iss(line);
    size_t id;
    string xStr, yStr;
    iss >> id >> xStr >> yStr;
    node[i].x = stof(xStr);
    node[i].y = stof(yStr);
  }

  // Skip DEMAND_SECTION header.
  getline(in, line);
  for (size_t i = 0; i < size; ++i) {
    getline(in, line);
    stringstream iss(line);
    size_t id;
    string dStr;
    iss >> id >> dStr;
    node[i].demand = stof(dStr);
  }
  in.close();
  return capacity;
}

void VRP::print() {
  std::cout << "DIMENSION:" << size << '\n';
  std::cout << "CAPACITY:" << capacity << '\n';
  for (size_t i = 0; i < size; ++i) {
    std::cout << i << ':' << std::setw(6) << node[i].x << ' '
              << std::setw(6) << node[i].y << ' '
              << std::setw(6) << node[i].demand << std::endl;
  }
}

void VRP::print_dist() {
  for (size_t i = 0; i < size; ++i) {
    std::cout << i << ":";
    for (size_t j = 0; j < size; ++j) {
      cout << std::setw(10) << get_dist(i, j) << ' ';
    }
    std::cout << std::endl;
  }
}

// ---------------------------------------------------------------------
// The following functions (PrimsAlgo, DFS shortcircuittest, conversion routines,
// tsp_approx, 2opt etc.) remain essentially the same as in the original code.

std::vector<std::vector<Edge>> PrimsAlgo(const VRP &vrp, std::vector<std::vector<Edge>> &graph) {
  auto N = graph.size();
  const node_t INIT = -1;
  std::vector<weight_t> key(N, INT_MAX);
  std::vector<weight_t> toEdges(N, -1);
  std::vector<bool> visited(N, false);
  std::set<std::pair<weight_t, node_t>> active;
  std::vector<std::vector<Edge>> nG(N);
  node_t src = 0;
  key[src] = 0.0;
  active.insert({0.0, src});
  while (!active.empty()) {
    auto where = active.begin()->second;
    active.erase(active.begin());
    if (visited[where])
      continue;
    visited[where] = true;
    for (Edge E : graph[where]) {
      if (!visited[E.to] && E.length < key[E.to]) {
        key[E.to] = E.length;
        active.insert({key[E.to], E.to});
        toEdges[E.to] = where;
      }
    }
  }
  node_t u = 0;
  for (auto v : toEdges) {
    if (v != INIT) {
      weight_t w = vrp.get_dist(u, v);
      nG[u].push_back(Edge(v, w));
      nG[v].push_back(Edge(u, w));
    }
    u++;
  }
  return nG;
}

void printAdjList(const std::vector<std::vector<Edge>> &graph) {
  int i = 0;
  for (auto vec : graph) {
    std::cout << i << ": ";
    for (auto e : vec) {
      std::cout << e.to << " ";
    }
    i++;
    std::cout << std::endl;
  }
}

void ShortCircutTour(std::vector<std::vector<Edge>> &g, std::vector<bool> &visited, node_t u, std::vector<node_t> &out) {
  visited[u] = true;
  out.push_back(u);
  for (auto e : g[u]) {
    node_t v = e.to;
    if (!visited[v])
      ShortCircutTour(g, visited, v, out);
  }
}

std::vector<std::vector<node_t>> convertToVrpRoutes(const VRP &vrp, const std::vector<node_t> &singleRoute) {
  std::vector<std::vector<node_t>> routes;
  demand_t vCapacity = vrp.getCapacity();
  demand_t residueCap = vCapacity;
  std::vector<node_t> aRoute;
  for (auto v : singleRoute) {
    if (v == 0)
      continue;
    if (residueCap - vrp.node[v].demand >= 0) {
      aRoute.push_back(v);
      residueCap -= vrp.node[v].demand;
    } else {
      routes.push_back(aRoute);
      aRoute.clear();
      aRoute.push_back(v);
      residueCap = vCapacity - vrp.node[v].demand;
    }
  }
  routes.push_back(aRoute);
  return routes;
}

weight_t calRouteValue(const VRP &vrp, const std::vector<node_t> &aRoute, node_t depot = 1) {
  weight_t routeVal = 0;
  node_t prevPoint = 0; // starting from depot.
  for (auto aPoint : aRoute) {
    routeVal += vrp.get_dist(prevPoint, aPoint);
    prevPoint = aPoint;
  }
  routeVal += vrp.get_dist(prevPoint, 0);
  return routeVal;
}

void printOutput(const VRP &vrp, const std::vector<std::vector<node_t>> &final_routes) {
  weight_t total_cost = 0.0;
  for (unsigned ii = 0; ii < final_routes.size(); ++ii) {
    std::cout << "Route #" << ii + 1 << ":";
    for (unsigned jj = 0; jj < final_routes[ii].size(); ++jj) {
      std::cout << " " << final_routes[ii][jj];
    }
    std::cout << '\n';
  }
  for (unsigned ii = 0; ii < final_routes.size(); ++ii) {
    weight_t curr_route_cost = 0;
    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][0]);
    for (unsigned jj = 1; jj < final_routes[ii].size(); ++jj)
      curr_route_cost += vrp.get_dist(final_routes[ii][jj - 1], final_routes[ii][jj]);
    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii].back());
    total_cost += curr_route_cost;
  }
  std::cout << "Cost " << total_cost << std::endl;
}

void tsp_approx(const VRP &vrp, std::vector<node_t> &cities, std::vector<node_t> &tour, node_t ncities) {
  node_t i, j;
  node_t ClosePt = 0;
  weight_t CloseDist;
  for (i = 1; i < ncities; i++)
    tour[i] = cities[i - 1];
  tour[0] = cities[ncities - 1];
  for (i = 1; i < ncities; i++) {
    weight_t ThisX = vrp.node[tour[i - 1]].x;
    weight_t ThisY = vrp.node[tour[i - 1]].y;
    CloseDist = DBL_MAX;
    for (j = ncities - 1;; j--) {
      weight_t ThisDist = (vrp.node[tour[j]].x - ThisX) * (vrp.node[tour[j]].x - ThisX);
      if (ThisDist <= CloseDist) {
        ThisDist += (vrp.node[tour[j]].y - ThisY) * (vrp.node[tour[j]].y - ThisY);
        if (ThisDist <= CloseDist) {
          if (j < i)
            break;
          CloseDist = ThisDist;
          ClosePt = j;
        }
      }
    }
    unsigned temp = tour[i];
    tour[i] = tour[ClosePt];
    tour[ClosePt] = temp;
  }
}

std::vector<std::vector<node_t>> postprocess_tsp_approx(const VRP &vrp, std::vector<std::vector<node_t>> &solRoutes) {
  std::vector<std::vector<node_t>> modifiedRoutes;
  unsigned nroutes = solRoutes.size();
  for (unsigned i = 0; i < nroutes; ++i) {
    unsigned sz = solRoutes[i].size();
    std::vector<node_t> cities(sz + 1);
    std::vector<node_t> tour(sz + 1);
    for (unsigned j = 0; j < sz; ++j)
      cities[j] = solRoutes[i][j];
    cities[sz] = 0; // depot as last node.
    tsp_approx(vrp, cities, tour, sz + 1);
    std::vector<node_t> curr_route;
    for (unsigned kk = 1; kk < sz + 1; ++kk)
      curr_route.push_back(tour[kk]);
    modifiedRoutes.push_back(curr_route);
  }
  return modifiedRoutes;
}

void tsp_2opt(const VRP &vrp, std::vector<node_t> &cities, std::vector<node_t> &tour, unsigned ncities) {
  unsigned improve = 0;
  while (improve < 2) {
    double best_distance = 0.0;
    best_distance += vrp.get_dist(DEPOT, cities[0]);
    for (unsigned jj = 1; jj < ncities; ++jj)
      best_distance += vrp.get_dist(cities[jj - 1], cities[jj]);
    best_distance += vrp.get_dist(DEPOT, cities[ncities - 1]);
    for (unsigned i = 0; i < ncities - 1; i++) {
      for (unsigned k = i + 1; k < ncities; k++) {
        for (unsigned c = 0; c < i; ++c)
          tour[c] = cities[c];
        unsigned dec = 0;
        for (unsigned c = i; c < k + 1; ++c) {
          tour[c] = cities[k - dec];
          dec++;
        }
        for (unsigned c = k + 1; c < ncities; ++c)
          tour[c] = cities[c];
        double new_distance = 0.0;
        new_distance += vrp.get_dist(DEPOT, tour[0]);
        for (unsigned jj = 1; jj < ncities; ++jj)
          new_distance += vrp.get_dist(tour[jj - 1], tour[jj]);
        new_distance += vrp.get_dist(DEPOT, tour[ncities - 1]);
        if (new_distance < best_distance) {
          improve = 0;
          for (unsigned jj = 0; jj < ncities; jj++)
            cities[jj] = tour[jj];
          best_distance = new_distance;
        }
      }
    }
    improve++;
  }
}

std::vector<std::vector<node_t>> postprocess_2OPT(const VRP &vrp, std::vector<std::vector<node_t>> &final_routes) {
  std::vector<std::vector<node_t>> postprocessed_final_routes;
  unsigned nroutes = final_routes.size();
  for (unsigned i = 0; i < nroutes; ++i) {
    unsigned sz = final_routes[i].size();
    std::vector<node_t> cities(sz);
    std::vector<node_t> tour(sz);
    for (unsigned j = 0; j < sz; ++j)
      cities[j] = final_routes[i][j];
    std::vector<node_t> curr_route;
    if (sz > 2)
      tsp_2opt(vrp, cities, tour, sz);
    for (unsigned kk = 0; kk < sz; ++kk)
      curr_route.push_back(cities[kk]);
    postprocessed_final_routes.push_back(curr_route);
  }
  return postprocessed_final_routes;
}

weight_t get_total_cost_of_routes(const VRP &vrp, std::vector<std::vector<node_t>> &final_routes) {
  weight_t total_cost = 0.0;
  for (unsigned ii = 0; ii < final_routes.size(); ++ii) {
    weight_t curr_route_cost = 0;
    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][0]);
    for (unsigned jj = 1; jj < final_routes[ii].size(); ++jj)
      curr_route_cost += vrp.get_dist(final_routes[ii][jj - 1], final_routes[ii][jj]);
    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii].back());
    total_cost += curr_route_cost;
  }
  return total_cost;
}

//
// MAIN POST PROCESS ROUTINE
//
std::vector<std::vector<node_t>> postProcessIt(const VRP &vrp, std::vector<std::vector<node_t>> &final_routes, weight_t &minCost) {
  std::vector<std::vector<node_t>> postprocessed_final_routes;
  auto postprocessed_final_routes1 = postprocess_tsp_approx(vrp, final_routes);
  auto postprocessed_final_routes2 = postprocess_2OPT(vrp, postprocessed_final_routes1);
  auto postprocessed_final_routes3 = postprocess_2OPT(vrp, final_routes);
  for (unsigned zzz = 0; zzz < final_routes.size(); ++zzz) {
    std::vector<node_t> postprocessed_route2 = postprocessed_final_routes2[zzz];
    std::vector<node_t> postprocessed_route3 = postprocessed_final_routes3[zzz];
    unsigned sz2 = postprocessed_route2.size();
    unsigned sz3 = postprocessed_route3.size();
    weight_t postprocessed_route2_cost = 0.0;
    postprocessed_route2_cost += vrp.get_dist(DEPOT, postprocessed_route2[0]);
    for (unsigned jj = 1; jj < sz2; ++jj)
      postprocessed_route2_cost += vrp.get_dist(postprocessed_route2[jj - 1], postprocessed_route2[jj]);
    postprocessed_route2_cost += vrp.get_dist(DEPOT, postprocessed_route2.back());
    weight_t postprocessed_route3_cost = 0.0;
    postprocessed_route3_cost += vrp.get_dist(DEPOT, postprocessed_route3[0]);
    for (unsigned jj = 1; jj < sz3; ++jj)
      postprocessed_route3_cost += vrp.get_dist(postprocessed_route3[jj - 1], postprocessed_route3[jj]);
    postprocessed_route3_cost += vrp.get_dist(DEPOT, postprocessed_route3.back());
    if (postprocessed_route3_cost > postprocessed_route2_cost)
      postprocessed_final_routes.push_back(postprocessed_route2);
    else
      postprocessed_final_routes.push_back(postprocessed_route3);
  }
  auto postprocessed_final_routes_cost = get_total_cost_of_routes(vrp, postprocessed_final_routes);
  minCost = postprocessed_final_routes_cost;
  return postprocessed_final_routes;
}

std::pair<weight_t, std::vector<std::vector<node_t>>> calCost(const VRP &vrp, const std::vector<std::vector<node_t>> &final_routes) {
  weight_t total_cost = 0.0;
  for (unsigned ii = 0; ii < final_routes.size(); ++ii) {
    weight_t curr_route_cost = 0;
    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][0]);
    for (unsigned jj = 1; jj < final_routes[ii].size(); ++jj)
      curr_route_cost += vrp.get_dist(final_routes[ii][jj - 1], final_routes[ii][jj]);
    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii].back());
    total_cost += curr_route_cost;
  }
  return {total_cost, final_routes};
}

bool verify_sol(const VRP &vrp, std::vector<std::vector<node_t>> final_routes, unsigned capacity) {
  unsigned *hist = new unsigned[vrp.getSize()];
  memset(hist, 0, sizeof(unsigned) * vrp.getSize());
  for (unsigned i = 0; i < final_routes.size(); ++i) {
    unsigned route_sum_of_demands = 0;
    for (unsigned j = 0; j < final_routes[i].size(); ++j) {
      route_sum_of_demands += vrp.node[final_routes[i][j]].demand;
      hist[final_routes[i][j]]++;
    }
    if (route_sum_of_demands > capacity) {
      delete[] hist;
      return false;
    }
  }
  for (unsigned i = 1; i < vrp.getSize(); ++i) {
    if (hist[i] > 1 || hist[i] == 0) {
      delete[] hist;
      return false;
    }
  }
  delete[] hist;
  return true;
}

// ---------------------------------------------------------------------
// MAIN
int main(int argc, char *argv[]) {
  VRP vrp;
  if (argc < 2) {
    std::cout << "parMDS CUDA version\n";
    std::cout << "Usage: " << argv[0]
              << " toy.vrp [-nthreads <n> DEFAULT is 20] [-round 0 or 1 DEFAULT:1]\n";
    exit(1);
  }
  for (int ii = 2; ii < argc; ii += 2) {
    if (std::string(argv[ii]) == "-round")
      vrp.params.toRound = atoi(argv[ii + 1]);
    else if (std::string(argv[ii]) == "-nthreads")
      vrp.params.nThreads = atoi(argv[ii + 1]);
    else {
      std::cerr << "INVALID Arguments!\nUsage:" << argv[0]
                << " toy.vrp -nthreads 20 -round 1\n";
      exit(1);
    }
  }
  vrp.read(argv[1]);

  // START TIMER
  auto start = std::chrono::high_resolution_clock::now();

  // Compute complete graph distances using the CUDA kernel.
  auto cG = vrp.cal_graph_dist();

  // Compute the MST using Prim's algorithm on the complete graph.
  auto mstG = PrimsAlgo(vrp, cG);

  // DFS from MST to get a single route.
  std::vector<bool> visited(mstG.size(), false);
  visited[0] = true;
  std::vector<int> singleRoute;
  weight_t minCost = INT_MAX * 1.0;
  std::vector<std::vector<node_t>> minRoute;

  auto mstCopy = mstG; // make a copy as the DFS routine will modify it.

  // One initial iteration (could be repeated later).
  for (int i = 0; i < 1; i++) {
    for (auto &list : mstCopy) {
      // Randomize using a default random engine.
      std::shuffle(list.begin(), list.end(), std::default_random_engine(0));
    }
    std::vector<int> singleRouteLocal;
    std::vector<bool> visitedLocal(mstCopy.size(), false);
    visitedLocal[0] = true;
    ShortCircutTour(mstCopy, visitedLocal, 0, singleRouteLocal);
    auto aRoutes = convertToVrpRoutes(vrp, singleRouteLocal);
    auto aCostRoute = calCost(vrp, aRoutes);
    if (aCostRoute.first < minCost) {
      minCost = aCostRoute.first;
      minRoute = aCostRoute.second;
    }
  }
  auto minCost1 = minCost;

  auto end = std::chrono::high_resolution_clock::now();
  uint64_t elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  double timeUpto1 = elapsed * 1.E-9;
  short PARLIMIT = vrp.params.nThreads;

  // Instead of using OpenMP, we perform serial iterations here.
  for (int i = 0; i < 100000; i += PARLIMIT) {
    for (auto &list : mstCopy) {
      std::shuffle(list.begin(), list.end(), std::default_random_engine(rand()));
    }
    std::vector<int> singleRouteLocal;
    std::vector<bool> visitedLocal(mstCopy.size(), false);
    visitedLocal[0] = true;
    ShortCircutTour(mstCopy, visitedLocal, 0, singleRouteLocal);
    auto aRoutes = convertToVrpRoutes(vrp, singleRouteLocal);
    auto aCostRoute = calCost(vrp, aRoutes);
    if (aCostRoute.first < minCost) {
      minCost = aCostRoute.first;
      minRoute = aCostRoute.second;
    }
  }
  auto minCost2 = minCost;
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  double timeUpto2 = elapsed * 1.E-9;

  auto postRoutes = postProcessIt(vrp, minRoute, minCost);

  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  double total_time = elapsed * 1.E-9;

  bool verified = verify_sol(vrp, postRoutes, vrp.getCapacity());

  std::cerr << argv[1] << " Cost " << minCost1 << ' ' << minCost2 << ' ' << minCost;
  std::cerr << " Time(seconds) " << timeUpto1 << ' ' << timeUpto2 << ' ' << total_time;
  std::cerr << " parLimit " << PARLIMIT;
  if (verified)
    std::cerr << " VALID" << std::endl;
  else
    std::cerr << " INVALID" << std::endl;

  printOutput(vrp, postRoutes);

  return 0;
}