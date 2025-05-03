#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>
using namespace std;

struct Node {
    int id;
    double x, y;
    int demand;
};

struct Vehicle {
    int capacity;
    int load = 0;
    vector<int> route; // list of node IDs, starting and ending at depot (0)
};

double Distance(const Node& a, const Node& b) {
    return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}

double RouteCost(const vector<int>& route, const vector<Node>& nodes) {
    double cost = 0.0;
    for (int i = 0; i < route.size() - 1; ++i)
        cost += Distance(nodes[route[i]], nodes[route[i+1]]);
    return cost;
}

double TotalCost(const vector<Vehicle>& vehicles, const vector<Node>& nodes) {
    double total = 0.0;
    for (const auto& v : vehicles)
        total += RouteCost(v.route, nodes);
    return total;
}

// Create a simple initial solution (greedy split by capacity)
vector<Vehicle> CreateInitialSolution(const vector<Node>& nodes, int vehicleCount, int vehicleCap) {
    vector<Vehicle> vehicles(vehicleCount, {vehicleCap});
    int v = 0;
    for (int i = 1; i < nodes.size(); ++i) {
        if (vehicles[v].load + nodes[i].demand > vehicleCap) {
            v++;
            if (v >= vehicleCount) break;
        }
        vehicles[v].load += nodes[i].demand;
        vehicles[v].route.push_back(i);
    }
    for (auto& veh : vehicles) {
        veh.route.insert(veh.route.begin(), 0); // start at depot
        veh.route.push_back(0); // end at depot
    }
    return vehicles;
}

vector<Vehicle> SimulatedAnnealingVRP(const vector<Node>& nodes, int vehicleCount, int vehicleCap) {
    vector<Vehicle> current = CreateInitialSolution(nodes, vehicleCount, vehicleCap);
    vector<Vehicle> best = current;
    double bestCost = TotalCost(best, nodes);
    double temp = 1000.0;
    double cooling = 0.995;
    int maxIter = 10000;

    srand(time(0));

    for (int iter = 0; iter < maxIter; ++iter) {
        vector<Vehicle> neighbor = current;

        // Swap two customers between two random vehicles
        int v1 = rand() % vehicleCount;
        int v2 = rand() % vehicleCount;
        if (neighbor[v1].route.size() <= 2 || neighbor[v2].route.size() <= 2) continue;

        int i = 1 + rand() % (neighbor[v1].route.size() - 2);
        int j = 1 + rand() % (neighbor[v2].route.size() - 2);

        swap(neighbor[v1].route[i], neighbor[v2].route[j]);

        double newCost = TotalCost(neighbor, nodes);
        double delta = newCost - bestCost;

        if (delta < 0 || (rand() / (double)RAND_MAX) < exp(-delta / temp)) {
            current = neighbor;
            if (newCost < bestCost) {
                best = neighbor;
                bestCost = newCost;
            }
        }

        temp *= cooling;
    }

    return best;
}
