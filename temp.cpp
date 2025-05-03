#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>
#include <unordered_map>
using namespace std;

struct Node {
    int id;
    double x, y;
    int demand;
};

struct Vehicle {
    int capacity;
    int load = 0;
    int depotID;
    vector<int> route; // customer node IDs only
};

double Distance(const Node& a, const Node& b) {
    return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}

double RouteCost(const vector<int>& route, const vector<Node>& nodes, const Node& depot) {
    double cost = 0.0;
    cost += Distance(depot, nodes[route.front()]);
    for (int i = 0; i < route.size() - 1; ++i)
        cost += Distance(nodes[route[i]], nodes[route[i+1]]);
    cost += Distance(nodes[route.back()], depot);
    return cost;
}

double TotalCost(const vector<Vehicle>& vehicles, const vector<Node>& nodes, const unordered_map<int, Node>& depots) {
    double total = 0.0;
    for (const auto& v : vehicles)
        total += RouteCost(v.route, nodes, depots.at(v.depotID));
    return total;
}

vector<Vehicle> CreateInitialSolution(const vector<Node>& nodes, const vector<int>& depotIDs, int vehiclesPerDepot, int vehicleCap) {
    unordered_map<int, Node> depots;
    for (int id : depotIDs)
        depots[id] = nodes[id];

    vector<Node> customers;
    for (const auto& node : nodes)
        if (find(depotIDs.begin(), depotIDs.end(), node.id) == depotIDs.end())
            customers.push_back(node);

    vector<Vehicle> vehicles;
    int depotIdx = 0;
    for (int d : depotIDs) {
        for (int v = 0; v < vehiclesPerDepot; ++v)
            vehicles.push_back({vehicleCap, 0, d, {}});
    }

    int vi = 0;
    for (const auto& cust : customers) {
        while (vi < vehicles.size() && vehicles[vi].load + cust.demand > vehicleCap)
            vi++;
        if (vi >= vehicles.size()) break;

        vehicles[vi].load += cust.demand;
        vehicles[vi].route.push_back(cust.id);
    }

    return vehicles;
}

vector<Vehicle> SimulatedAnnealingVRP(const vector<Node>& nodes, const vector<int>& depotIDs, int vehiclesPerDepot, int vehicleCap) {
    unordered_map<int, Node> depots;
    for (int id : depotIDs)
        depots[id] = nodes[id];

    vector<Vehicle> current = CreateInitialSolution(nodes, depotIDs, vehiclesPerDepot, vehicleCap);
    vector<Vehicle> best = current;
    double bestCost = TotalCost(best, nodes, depots);
    double temp = 1000.0;
    double cooling = 0.995;
    int maxIter = 10000;

    srand(time(0));

    for (int iter = 0; iter < maxIter; ++iter) {
        vector<Vehicle> neighbor = current;

        int v1 = rand() % neighbor.size();
        int v2 = rand() % neighbor.size();
        if (neighbor[v1].route.empty() || neighbor[v2].route.empty()) continue;

        int i = rand() % neighbor[v1].route.size();
        int j = rand() % neighbor[v2].route.size();

        swap(neighbor[v1].route[i], neighbor[v2].route[j]);

        double newCost = TotalCost(neighbor, nodes, depots);
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


  vector<int> depotIDs = {0, 1};  // Regional depots
    int vehiclesPerDepot = 2;
    int vehicleCapacity = 10;

    vector<Vehicle> solution = SimulatedAnnealingVRP(nodes, depotIDs, vehiclesPerDepot, vehicleCapacity);

    // Print the solution
    for (int i = 0; i < solution.size(); ++i) {
        const Vehicle& v = solution[i];
        cout << "Vehicle " << i << " (Depot " << v.depotID << "): ";
        cout << v.depotID << " -> ";
        for (int cust : v.route) {
            cout << cust << " -> ";
        }
        cout << v.depotID << endl;
    }
