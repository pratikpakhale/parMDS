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
#include <omp.h>

using namespace std;

unsigned DEBUGCODE = 0;
#define DEBUG if (DEBUGCODE)

using point_t = double;
using weight_t = double;
using demand_t = double;
using node_t = int;

const node_t DEPOT = 0; // The depot (always node 0)

//
// Parameters class: to store command-line options
//
class Params
{
public:
  Params()
  {
    toRound = true; // DEFAULT: round computed distances
    nThreads = 20;  // DEFAULT: 20 OpenMP threads
  }
  bool toRound;
  short nThreads;
};

//
// Edge class: represents an edge to a neighbor node and its length.
//
class Edge
{
public:
  node_t to;
  weight_t length;
  Edge() : to(0), length(0) {}
  Edge(node_t t, weight_t l) : to(t), length(l) {}
  bool operator<(const Edge &e) const
  {
    return length < e.length;
  }
};

//
// Point class: holds x and y coordinates plus demand.
//
class Point
{
public:
  point_t x;
  point_t y;
  demand_t demand;
};

//
// VRP class: holds instance information including list of nodes, capacity,
// the complete (triangular) distance array and additional methods.
//
class VRP
{
  size_t size;
  demand_t capacity;
  string type;

public:
  VRP() : size(0), capacity(0) {}
  unsigned read(string filename);
  void print();
  void print_dist();
  std::vector<std::vector<Edge>> cal_graph_dist();

  // Returns the distance between two nodes from the triangular distance array.
  weight_t get_dist(node_t i, node_t j) const
  {
    if (i == j)
      return 0.0;
    if (i > j)
      swap(i, j);
    // Mapping: myoffset = ((2*i*size) - (i*i) + i)/2, then index = myoffset + (j - (2*i+1))
    size_t myoffset = ((2 * i * size) - (i * i) + i) / 2;
    size_t correction = 2 * i + 1;
    return dist[myoffset + j - correction];
  }

  size_t getSize() const { return size; }
  demand_t getCapacity() const { return capacity; }

  vector<Point> node;
  vector<weight_t> dist;
  Params params;
};

//
// Computes the complete graph distances. Also builds an adjacency list (nG).
//
std::vector<std::vector<Edge>> VRP::cal_graph_dist()
{
  dist.resize((size * (size - 1)) / 2); // number of unique pairs
  std::vector<std::vector<Edge>> nG(size);
  size_t k = 0;
  for (size_t i = 0; i < size; ++i)
  {
    for (size_t j = i + 1; j < size; ++j)
    {
      weight_t w = sqrt(((node[i].x - node[j].x) * (node[i].x - node[j].x)) +
                        ((node[i].y - node[j].y) * (node[i].y - node[j].y)));
      weight_t final_w = (params.toRound ? round(w) : w);
      dist[k] = final_w;
      nG[i].push_back(Edge(j, final_w));
      nG[j].push_back(Edge(i, final_w));
      k++;
    }
  }
  return nG;
}

//
// Print the full distance matrix (for debugging).
//
void VRP::print_dist()
{
  for (size_t i = 0; i < size; ++i)
  {
    cout << i << ":";
    for (size_t j = 0; j < size; ++j)
    {
      cout << setw(10) << get_dist(i, j) << ' ';
    }
    cout << endl;
  }
}

//
// Reads a VRP instance from a file (expects a header with DIMENSION, CAPACITY,
// then NODE_COORD_SECTION and DEMAND_SECTION).
//
unsigned VRP::read(string filename)
{
  ifstream in(filename);
  if (!in.is_open())
  {
    cerr << "Could not open the file \"" << filename << "\"" << endl;
    exit(1);
  }
  string line;
  // Skip first three lines (adjust if necessary)
  for (int i = 0; i < 3; ++i)
    getline(in, line);

  // Read DIMENSION line.
  getline(in, line);
  size = stof(line.substr(line.find(":") + 2));

  // Read DISTANCE TYPE line (ignored further).
  getline(in, line);
  type = line;

  // Read CAPACITY line.
  getline(in, line);
  capacity = stof(line.substr(line.find(":") + 2));

  // Skip NODE_COORD_SECTION header.
  getline(in, line);

  node.resize(size);
  for (size_t i = 0; i < size; ++i)
  {
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
  for (size_t i = 0; i < size; ++i)
  {
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

//
// Print the instance summary
//
void VRP::print()
{
  cout << "DIMENSION:" << size << "\n";
  cout << "CAPACITY:" << capacity << "\n";
  for (size_t i = 0; i < size; ++i)
  {
    cout << i << ':' << setw(6) << node[i].x << ' '
         << setw(6) << node[i].y << ' '
         << setw(6) << node[i].demand << "\n";
  }
}

//
// Prim’s algorithm to compute a Minimum Spanning Tree (MST) from the complete graph.
//
std::vector<std::vector<Edge>> PrimsAlgo(const VRP &vrp, std::vector<std::vector<Edge>> &graph)
{
  size_t N = graph.size();
  const node_t INIT = -1;
  vector<weight_t> key(N, INT_MAX);
  vector<weight_t> toEdges(N, -1);
  vector<bool> visited(N, false);
  set<pair<weight_t, node_t>> active;
  vector<vector<Edge>> nG(N);
  node_t src = 0;
  key[src] = 0.0;
  active.insert({0.0, src});

  while (!active.empty())
  {
    auto where = active.begin()->second;
    active.erase(active.begin());
    if (visited[where])
      continue;
    visited[where] = true;
    for (auto E : graph[where])
    {
      if (!visited[E.to] && E.length < key[E.to])
      {
        key[E.to] = E.length;
        active.insert({key[E.to], E.to});
        toEdges[E.to] = where;
      }
    }
  }
  node_t u = 0;
  for (auto v : toEdges)
  {
    if (v != INIT)
    {
      weight_t w = vrp.get_dist(u, v);
      nG[u].push_back(Edge(v, w));
      nG[v].push_back(Edge(u, w));
    }
    u++;
  }
  return nG;
}

//
// Utility: prints the graph’s adjacency list.
//
void printAdjList(const vector<vector<Edge>> &graph)
{
  int i = 0;
  for (auto vec : graph)
  {
    cout << i << ": ";
    for (auto e : vec)
    {
      cout << e.to << " ";
    }
    i++;
    cout << endl;
  }
}

//
// DFS routine to get a “short-circuit” tour of the MST.
//
void ShortCircutTour(vector<vector<Edge>> &g, vector<bool> &visited, node_t u, vector<node_t> &out)
{
  visited[u] = true;
  out.push_back(u);
  for (auto e : g[u])
  {
    node_t v = e.to;
    if (!visited[v])
      ShortCircutTour(g, visited, v, out);
  }
}

//
// Converts a single route (a permutation of node indices) into
// a set of VRP routes (partitioning the tour based on vehicle capacity).
//
vector<vector<node_t>> convertToVrpRoutes(const VRP &vrp,
                                          const vector<node_t> &singleRoute)
{
  vector<vector<node_t>> routes;
  demand_t vCapacity = vrp.getCapacity();
  demand_t residueCap = vCapacity;
  vector<node_t> aRoute;
  for (auto v : singleRoute)
  {
    if (v == DEPOT)
      continue;
    if (residueCap - vrp.node[v].demand >= 0)
    {
      aRoute.push_back(v);
      residueCap -= vrp.node[v].demand;
    }
    else
    {
      routes.push_back(aRoute);
      aRoute.clear();
      aRoute.push_back(v);
      residueCap = vCapacity - vrp.node[v].demand;
    }
  }
  routes.push_back(aRoute);
  return routes;
}

//
// Computes the total cost (length) of a given route (starting and ending at the depot).
//
weight_t calRouteValue(const VRP &vrp, const vector<node_t> &aRoute, node_t depot = DEPOT)
{
  weight_t routeVal = 0;
  node_t prevPoint = DEPOT;
  for (auto aPoint : aRoute)
  {
    routeVal += vrp.get_dist(prevPoint, aPoint);
    prevPoint = aPoint;
  }
  routeVal += vrp.get_dist(prevPoint, DEPOT);
  return routeVal;
}

//
// Prints the final routes and the total cost.
//
void printOutput(const VRP &vrp, const vector<vector<node_t>> &final_routes)
{
  weight_t totalCost = 0.0;
  for (unsigned ii = 0; ii < final_routes.size(); ++ii)
  {
    cout << "Route #" << ii + 1 << ":";
    for (unsigned jj = 0; jj < final_routes[ii].size(); ++jj)
      cout << " " << final_routes[ii][jj];
    cout << "\n";
  }
  for (unsigned ii = 0; ii < final_routes.size(); ++ii)
  {
    weight_t curr_route_cost = 0;
    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][0]);
    for (unsigned jj = 1; jj < final_routes[ii].size(); ++jj)
      curr_route_cost += vrp.get_dist(final_routes[ii][jj - 1],
                                      final_routes[ii][jj]);
    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii].back());
    totalCost += curr_route_cost;
  }
  cout << "Cost " << totalCost << endl;
}

//
// A nearest-neighbor based TSP approximation routine (with a “sweep” starting from the depot).
//
void tsp_approx(const VRP &vrp, vector<node_t> &cities, vector<node_t> &tour, node_t ncities)
{
  node_t i, j;
  node_t ClosePt = 0;
  weight_t CloseDist;
  for (i = 1; i < ncities; i++)
    tour[i] = cities[i - 1];
  tour[0] = cities[ncities - 1];
  for (i = 1; i < ncities; i++)
  {
    weight_t ThisX = vrp.node[tour[i - 1]].x;
    weight_t ThisY = vrp.node[tour[i - 1]].y;
    CloseDist = DBL_MAX;
    for (j = ncities - 1;; j--)
    {
      weight_t ThisDist = (vrp.node[tour[j]].x - ThisX) * (vrp.node[tour[j]].x - ThisX);
      if (ThisDist <= CloseDist)
      {
        ThisDist += (vrp.node[tour[j]].y - ThisY) * (vrp.node[tour[j]].y - ThisY);
        if (ThisDist <= CloseDist)
        {
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

//
// Postprocess routes using the TSP approximation.
//
vector<vector<node_t>> postprocess_tsp_approx(const VRP &vrp, vector<vector<node_t>> &solRoutes)
{
  vector<vector<node_t>> modifiedRoutes;
  unsigned nroutes = solRoutes.size();
  for (unsigned i = 0; i < nroutes; ++i)
  {
    unsigned sz = solRoutes[i].size();
    vector<node_t> cities(sz + 1);
    vector<node_t> tour(sz + 1);
    for (unsigned j = 0; j < sz; ++j)
      cities[j] = solRoutes[i][j];
    cities[sz] = DEPOT; // depot as the last node
    tsp_approx(vrp, cities, tour, sz + 1);
    vector<node_t> curr_route;
    for (unsigned kk = 1; kk < sz + 1; ++kk)
      curr_route.push_back(tour[kk]);
    modifiedRoutes.push_back(curr_route);
  }
  return modifiedRoutes;
}

//
// A simple 2‑opt TSP improvement routine.
//
void tsp_2opt(const VRP &vrp, vector<node_t> &cities, vector<node_t> &tour, unsigned ncities)
{
  unsigned improve = 0;
  while (improve < 2)
  {
    double best_distance = 0.0;
    best_distance += vrp.get_dist(DEPOT, cities[0]);
    for (unsigned jj = 1; jj < ncities; ++jj)
      best_distance += vrp.get_dist(cities[jj - 1], cities[jj]);
    best_distance += vrp.get_dist(DEPOT, cities[ncities - 1]);
    for (unsigned i = 0; i < ncities - 1; i++)
    {
      for (unsigned k = i + 1; k < ncities; k++)
      {
        for (unsigned c = 0; c < i; ++c)
          tour[c] = cities[c];
        unsigned dec = 0;
        for (unsigned c = i; c < k + 1; ++c)
        {
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
        if (new_distance < best_distance)
        {
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

//
// Postprocess routes further using 2‑opt improvement.
//
vector<vector<node_t>> postprocess_2OPT(const VRP &vrp, vector<vector<node_t>> &final_routes)
{
  vector<vector<node_t>> postprocessed_final_routes;
  unsigned nroutes = final_routes.size();
  for (unsigned i = 0; i < nroutes; ++i)
  {
    unsigned sz = final_routes[i].size();
    vector<node_t> cities(sz);
    vector<node_t> tour(sz);
    for (unsigned j = 0; j < sz; ++j)
      cities[j] = final_routes[i][j];
    vector<node_t> curr_route;
    if (sz > 2)
      tsp_2opt(vrp, cities, tour, sz);
    for (unsigned kk = 0; kk < sz; ++kk)
      curr_route.push_back(cities[kk]);
    postprocessed_final_routes.push_back(curr_route);
  }
  return postprocessed_final_routes;
}

//
// Returns the total cost of all routes.
//
weight_t get_total_cost_of_routes(const VRP &vrp, vector<vector<node_t>> &final_routes)
{
  weight_t total_cost = 0.0;
  for (unsigned ii = 0; ii < final_routes.size(); ++ii)
  {
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
// Main postprocessing routine: choose the better route from two 2‑opt schemes.
//
vector<vector<node_t>> postProcessIt(const VRP &vrp, vector<vector<node_t>> &final_routes, weight_t &minCost)
{
  vector<vector<node_t>> postprocessed_final_routes;
  auto postprocessed_final_routes1 = postprocess_tsp_approx(vrp, final_routes);
  auto postprocessed_final_routes2 = postprocess_2OPT(vrp, postprocessed_final_routes1);
  auto postprocessed_final_routes3 = postprocess_2OPT(vrp, final_routes);

#pragma omp parallel for num_threads(vrp.params.nThreads) schedule(dynamic)
  for (unsigned zzz = 0; zzz < final_routes.size(); ++zzz)
  {
    vector<node_t> postprocessed_route2 = postprocessed_final_routes2[zzz];
    vector<node_t> postprocessed_route3 = postprocessed_final_routes3[zzz];
    unsigned sz2 = postprocessed_route2.size();
    unsigned sz3 = postprocessed_route3.size();
    weight_t postprocessed_route2_cost = 0.0;
    postprocessed_route2_cost += vrp.get_dist(DEPOT, postprocessed_route2[0]);
    for (unsigned jj = 1; jj < sz2; ++jj)
      postprocessed_route2_cost += vrp.get_dist(postprocessed_route2[jj - 1], postprocessed_route2[jj]);
    postprocessed_route2_cost += vrp.get_dist(DEPOT, postprocessed_route2[sz2 - 1]);

    weight_t postprocessed_route3_cost = 0.0;
    postprocessed_route3_cost += vrp.get_dist(DEPOT, postprocessed_route3[0]);
    for (unsigned jj = 1; jj < sz3; ++jj)
      postprocessed_route3_cost += vrp.get_dist(postprocessed_route3[jj - 1], postprocessed_route3[jj]);
    postprocessed_route3_cost += vrp.get_dist(DEPOT, postprocessed_route3[sz3 - 1]);

#pragma omp critical
    {
      if (postprocessed_route3_cost > postprocessed_route2_cost)
        postprocessed_final_routes.push_back(postprocessed_route2);
      else
        postprocessed_final_routes.push_back(postprocessed_route3);
    }
  }
  auto post_cost = get_total_cost_of_routes(vrp, postprocessed_final_routes);
  minCost = post_cost;
  return postprocessed_final_routes;
}

//
// Computes the total cost of a set of routes (using OpenMP reduction).
//
pair<weight_t, vector<vector<node_t>>> calCost(const VRP &vrp, const vector<vector<node_t>> &final_routes)
{
  weight_t total_cost = 0.0;
#pragma omp parallel for reduction(+ : total_cost) num_threads(vrp.params.nThreads)
  for (unsigned ii = 0; ii < final_routes.size(); ++ii)
  {
    weight_t curr_route_cost = 0;
    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][0]);
    for (unsigned jj = 1; jj < final_routes[ii].size(); ++jj)
      curr_route_cost += vrp.get_dist(final_routes[ii][jj - 1], final_routes[ii][jj]);
    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii].back());
    total_cost += curr_route_cost;
  }
  return {total_cost, final_routes};
}

//
// Verifies that each customer (node 1..n-1) appears exactly once and that
// no vehicle’s capacity is exceeded.
//
bool verify_sol(const VRP &vrp, vector<vector<node_t>> final_routes, unsigned capacity)
{
  vector<unsigned> hist(vrp.getSize(), 0);
  for (unsigned i = 0; i < final_routes.size(); ++i)
  {
    unsigned route_sum_of_demands = 0;
    for (unsigned j = 0; j < final_routes[i].size(); ++j)
    {
      route_sum_of_demands += vrp.node[final_routes[i][j]].demand;
      hist[final_routes[i][j]]++;
    }
    if (route_sum_of_demands > capacity)
      return false;
  }
  for (unsigned i = 1; i < vrp.getSize(); ++i)
  {
    if (hist[i] != 1)
      return false;
  }
  return true;
}

//
// MAIN
//
int main(int argc, char *argv[])
{
  VRP vrp;
  if (argc < 2)
  {
    cout << "parMDS version 1.1" << "\n";
    cout << "Usage: " << argv[0] << " toy.vrp [-nthreads <n> DEFAULT is 20] [-round 0 or 1 DEFAULT:1]" << "\n";
    exit(1);
  }
  for (int ii = 2; ii < argc; ii += 2)
  {
    if (string(argv[ii]) == "-round")
      vrp.params.toRound = atoi(argv[ii + 1]);
    else if (string(argv[ii]) == "-nthreads")
      vrp.params.nThreads = atoi(argv[ii + 1]);
    else
    {
      cerr << "INVALID Arguments!\nUsage:" << argv[0] << " toy.vrp -nthreads 20 -round 1" << "\n";
      exit(1);
    }
  }
  vrp.read(argv[1]);

  auto start = chrono::high_resolution_clock::now();

  // Compute complete graph distances using a CPU routine.
  auto cG = vrp.cal_graph_dist();
  // Compute the MST from the complete graph.
  auto mstG = PrimsAlgo(vrp, cG);

  vector<bool> visited(mstG.size(), false);
  visited[0] = true;
  vector<int> singleRoute;

  weight_t minCost = INT_MAX * 1.0;
  vector<vector<node_t>> minRoute;

  // Make a copy of the MST (to be randomized in the DFS)
  auto mstCopy = mstG;

  // One initial iteration to establish a baseline solution.
  {
    for (auto &list : mstCopy)
    {
      // Use a fixed seed for reproducibility.
      unsigned seed = 0;
      shuffle(list.begin(), list.end(), default_random_engine(seed));
    }
    vector<int> singleRouteLocal;
    vector<bool> visitedLocal(mstCopy.size(), false);
    visitedLocal[0] = true;
    ShortCircutTour(mstCopy, visitedLocal, 0, singleRouteLocal);
    auto aRoutes = convertToVrpRoutes(vrp, singleRouteLocal);
    auto aCostRoute = calCost(vrp, aRoutes);
    if (aCostRoute.first < minCost)
    {
      minCost = aCostRoute.first;
      minRoute = aCostRoute.second;
    }
  }
  auto minCost1 = minCost;

  // Measure elapsed time up to baseline.
  auto end = chrono::high_resolution_clock::now();
  uint64_t elapsed = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
  double timeUpto1 = elapsed * 1.E-9;

  short PARLIMIT = vrp.params.nThreads;

// Parallelized iterative improvement: perform many DFS iterations in parallel.
#pragma omp parallel for num_threads(PARLIMIT) schedule(dynamic)
  for (int i = 0; i < 100000; i += PARLIMIT)
  {
    // Each thread shuffles the MST copy with a thread-specific seed.
    unsigned thread_seed = chrono::system_clock::now().time_since_epoch().count() + omp_get_thread_num();
    for (auto &list : mstCopy)
    {
      shuffle(list.begin(), list.end(), default_random_engine(thread_seed));
    }
    vector<int> singleRouteLocal;
    vector<bool> visitedLocal(mstCopy.size(), false);
    visitedLocal[0] = true;
    ShortCircutTour(mstCopy, visitedLocal, 0, singleRouteLocal);
    auto aRoutes = convertToVrpRoutes(vrp, singleRouteLocal);
    auto aCostRoute = calCost(vrp, aRoutes);
#pragma omp critical
    {
      if (aCostRoute.first < minCost)
      {
        minCost = aCostRoute.first;
        minRoute = aCostRoute.second;
      }
    }
  }
  auto minCost2 = minCost;
  end = chrono::high_resolution_clock::now();
  elapsed = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
  double timeUpto2 = elapsed * 1.E-9;

  // Postprocess the best found solution.
  auto postRoutes = postProcessIt(vrp, minRoute, minCost);

  end = chrono::high_resolution_clock::now();
  elapsed = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
  double total_time = elapsed * 1.E-9;

  bool verified = verify_sol(vrp, postRoutes, vrp.getCapacity());

  cerr << argv[1] << " Cost " << minCost1 << " " << minCost2 << " " << minCost;
  cerr << " Time(seconds) " << timeUpto1 << " " << timeUpto2 << " " << total_time;
  cerr << " parLimit " << PARLIMIT;
  if (verified)
    cerr << " VALID" << endl;
  else
    cerr << " INVALID" << endl;

  printOutput(vrp, postRoutes);

  return 0;
}