const int d = 10;
const int v = 10;
const int w = 10;
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <string>
#include <cstdlib>
#include <ctime>
#include <bits/stdc++.h>
#include <thread>
#include <regex>
#include <chrono>
#include <stream>
#include <algorithm>
#include <windows.h>

using namespace std;
using namespace chrono;

#ifdef _WIN32

#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif


struct City {
    int id;
    string name;
    int x, y;
    City(){}

    City(int id, const string& name, int x, int y)
        : id(id), name(name), x(x), y(y) {}
};

struct PPCity {
    int id;
    double demand;
    double supply;
    int orderId = -1;
    bool isPickup = false;
    PPCity() {}
    PPCity(int i, double d, double s = 0, int oid = -1, bool pickup = false) : id(i), demand(d), supply(s), orderId(oid), isPickup(pickup) {}
    bool operator==(const PPCity& other) const {
        return id == other.id && demand == other.demand && supply == other.supply;
    }
};

class Order {
public:
    static int n;
    int orderId;
    int sellerId;
    int source;
    int destination;
    double weight;
    double volume;

    Order(int sId,int src, int des, double w, double v):sellerId(sId),source(src), destination(des), weight(w), volume(v)
    {
        orderId=++n;
    }
};
int Order::n = 0;


class Seller {
public:
    int sellerId;
    int location;
    vector<Order> orders;

     Seller(int id, int loc)
        : sellerId(id), location(loc) {}

    void addOrder(const Order& o) {
        orders.push_back(o);
    }
};

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1); // combine hashes
    }
};

class HubHubCarrier {
public:
    int carrierId;
    int fromHubId;
    int toHubId;

    double maxWeight;
    double maxVolume;
    double remainingWeight;
    double remainingVolume;
    double speed;
    double pendingWeight;
    double pendingVolume;
    list<Order>pendingOrders;
    vector<Order> assignedOrders;

    HubHubCarrier(){}
    HubHubCarrier(int id, int fromHub, int toHub, double capWeight = 12000.0, double capVolume = 50.0, double spd = 60.0)
        : carrierId(id), fromHubId(fromHub), toHubId(toHub),
          maxWeight(capWeight), maxVolume(capVolume),
          remainingWeight(capWeight), remainingVolume(capVolume),
          speed(spd) {}

    bool canCarry(const Order& o) const {
        return /*o.source == fromHubId &&
               o.destination == toHubId &&*/
               o.weight <= remainingWeight &&
               o.volume <= remainingVolume;
    }

    void assignOrder(const Order& o) {
        assignedOrders.push_back(o);
        remainingWeight -= o.weight;
        remainingVolume -= o.volume;
    }
     void assignPendingOrder(const Order& o) {
        pendingOrders.push_back(o);
        pendingWeight+=o.weight;
        pendingVolume+=o.volume;
    }
};

unordered_map<pair<int,int>,HubHubCarrier,pair_hash>locToHHCarrier;
vector<vector<double>> distBtwCities;

pair<int,double> assignOrderHubHub(int src,int dest,Order order) {
    double dist=distBtwCities[src][dest];
    double cost=d*dist+w*order.weight+v*order.volume;
    int time=0;
    HubHubCarrier carrier=locToHHCarrier[{src,dest}];
    if (carrier.canCarry(order))
    {
        carrier.assignOrder(order);
        cout << "Assigned Order " << order.orderId
             << " (W: " << order.weight << ", V: " << order.volume << ") "
             << "to Carrier " << carrier.carrierId
             << " [Remaining W: " << carrier.remainingWeight
             << ", V: " << carrier.remainingVolume << "]\n";
        time++;
    }
    else
    {
        cout << "Order " << order.orderId << " could not be assigned (carrier capacity full).\n";
        carrier.assignPendingOrder(order);
        time+=ceil(max((double)carrier.pendingWeight/carrier.maxWeight,(double)carrier.pendingVolume/carrier.maxVolume));
    }
    return {time,cost};
}

class HubSpokeCarrier {
public:
    int carrierId;
    double maxWeight;
    double maxVolume;
    double speed;
    int hubLocationId;

    double remainingWeight;
    double remainingVolume;
    double pendingWeight;
    double pendingVolume;
    vector<Order>assignedOrders;
    list<Order>pendingOrders;

    HubSpokeCarrier(){}
    HubSpokeCarrier(int id, int hubLoc, double capWeight = 7000.0, double capVolume = 35.0, double spd = 40.0)
        : carrierId(id), hubLocationId(hubLoc), maxWeight(capWeight), maxVolume(capVolume), speed(spd),
          remainingWeight(capWeight), remainingVolume(capVolume) {}

    bool canCarry(const Order& o) const {
        return (o.weight <= remainingWeight) && (o.volume <= remainingVolume);
    }

    void assignOrder(const Order& o) {
        assignedOrders.push_back(o);
        remainingWeight -= o.weight;
        remainingVolume -= o.volume;
    }
    void assignPendingOrder(const Order& o) {
        pendingOrders.push_back(o);
        pendingWeight+=o.weight;
        pendingVolume+=o.volume;
    }
};

unordered_map<pair<int,int>,HubSpokeCarrier,pair_hash>locToHSCarrier;

pair<int,double> assignOrderSpokeHub(int src,int dest,Order order) {
    double dist=distBtwCities[src][dest];
    double cost=d*dist+w*order.weight+v*order.volume;
    int time=0;
    HubSpokeCarrier carrier=locToHSCarrier[{src,dest}];
    if (carrier.canCarry(order))
    {
        carrier.assignOrder(order);
        cout << "Assigned Order " << order.orderId
             << " (W: " << order.weight << ", V: " << order.volume << ") "
             << "to Carrier " << carrier.carrierId
             << " [Remaining W: " << carrier.remainingWeight
             << ", V: " << carrier.remainingVolume << "]\n";
        time++;
    }
    else
    {
        cout << "Order " << order.orderId << " could not be assigned (carrier capacity full).\n";
        carrier.assignPendingOrder(order);
        time+=ceil(max((double)carrier.pendingWeight/carrier.maxWeight,(double)carrier.pendingVolume/carrier.maxVolume));
    }
    return {time,cost};
}

double euclidean(const City& a, const City& b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

vector<vector<double>> floydWarshallFromAdjMatrix(vector<vector<double>>& adj_matrix) {
    int n = adj_matrix.size();
    vector<vector<double>>& adjMatrix=adj_matrix;
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (adjMatrix[i][k] < numeric_limits<double>::max() &&
                    adjMatrix[k][j] < numeric_limits<double>::max()) {
                    adjMatrix[i][j] = min(adjMatrix[i][j], adjMatrix[i][k] + adjMatrix[k][j]);
                }
            }
        }
    }
    return adjMatrix;
}

vector<vector<City>> kMeansClustering(const vector<City>& cities, int k, double& total_wcss) {
    int n = cities.size();
    vector<City> centroids(k);
    vector<vector<City>> clusters(k);
    vector<int> labels(n, -1);

    srand(time(0));

    // Initialize centroids randomly
    for (int i = 0; i < k; ++i)
        centroids[i] = cities[rand() % n];

    bool changed = true;
    int max_iters = 100;

    while (changed && max_iters--) {
        changed = false;

        for (int i = 0; i < k; ++i)
            clusters[i].clear();

        for (int i = 0; i < n; ++i) {
            double min_dist = numeric_limits<double>::max();
            int best_cluster = 0;

            for (int j = 0; j < k; ++j) {
                double dist = euclidean(cities[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }

            if (labels[i] != best_cluster) {
                changed = true;
                labels[i] = best_cluster;
            }

            clusters[best_cluster].push_back(cities[i]);
        }

        // Recalculate centroids
        for (int i = 0; i < k; ++i) {
            double sum_x = 0, sum_y = 0;
            int cluster_size = clusters[i].size();

            for (const City& c : clusters[i]) {
                sum_x += c.x;
                sum_y += c.y;
            }

            if (cluster_size > 0) {
                centroids[i].x = sum_x / cluster_size;
                centroids[i].y = sum_y / cluster_size;
            }
        }
    }

    // Compute total WCSS
    total_wcss = 0;
    for (int i = 0; i < k; ++i) {
        for (const City& c : clusters[i]) {
            total_wcss += pow(euclidean(c, centroids[i]), 2);
        }
    }

    return clusters;
}

City findHubs(vector<City>& clusterCities,vector<vector<double>>& adjMatrix) {
    City bestHub;
    double minTotalDist = 1e9;
    for (City city : clusterCities) {
        double totalDist = 0.0;

        for (City other : clusterCities) {
            if (city.id != other.id) {
                totalDist += adjMatrix[city.id-1][other.id-1];
            }
        }

        if (totalDist < minTotalDist) {
            minTotalDist = totalDist;
            bestHub = city;
        }

    }
    return bestHub;


}

unordered_map<int,int>spokeToHub;

void processOrder(Order& order)
{
    int srcHub=spokeToHub[order.source];
    int destHub=spokeToHub[order.destination];
    int time=0;
    double cost=0;
    pair<int,int>timeNdCost;
    if(order.source!=srcHub)
    {
        timeNdCost=assignOrderSpokeHub(order.source,srcHub,order);
        time+=timeNdCost.first;
        cost+=timeNdCost.second;
    }
    timeNdCost=assignOrderHubHub(srcHub,destHub,order);
    time+=timeNdCost.first;
    cost+=timeNdCost.second;
    if(order.destination!=destHub)
    {
        timeNdCost=assignOrderSpokeHub(destHub,order.destination,order);
        time+=timeNdCost.first;
        cost+=timeNdCost.second;
    }
    cout<<"For order id:"<<order.orderId<<endl;
    cout<<"Total time:"<<time<<"days"<<endl<<"total cost:"<<cost<<endl;
}

vector<Order> generateRandomOrders(int numOrders, int sellerIdStart = 1) {
    vector<Order> orders;
    random_device rd;
    mt19937 gen(rd());

    uniform_int_distribution<> locationDist(1, 49);     // Location IDs from 1 to 50
    uniform_real_distribution<> weightDist(300.0, 1000.0);  // weight in kg
    uniform_real_distribution<> volumeDist(1.0, 3.0);     // volume in m^3

    for (int i = 0; i < numOrders; ++i) {
        int sellerId = sellerIdStart + (i % 5);  // distribute among 5 sellers

        int pickup = locationDist(gen);
        int delivery = locationDist(gen);
        while (delivery == pickup) {
            delivery = locationDist(gen);  // Only re-roll delivery if it's same as pickup
        }

        double weight = weightDist(gen);
        double volume = volumeDist(gen);

        orders.push_back(*new Order(sellerId, pickup, delivery, weight, volume));
        cout<<"sid:"<<sellerId<<" "<<"oickup:"<<pickup<<" "<<"delivery"<<delivery<<" \n";
    }

return orders;
}



struct PPCarrier {
    double capacity = 6000;
    double load = 0;
    int depotID;
    vector<int> route; // sequence of PPCity indices
};


double RouteCost(const vector<int>& route, const vector<PPCity>& nodes, const PPCity& depot) {
    if (route.empty()) return 0;
    double cost = 0.0;
    cost += distBtwCities[depot.id-1][nodes[route.front()].id-1];
    for (int i = 0; i < route.size() - 1; ++i)
        cost += distBtwCities[nodes[route[i]].id-1][nodes[route[i+1]].id-1];
    cost += distBtwCities[nodes[route.back()].id-1][depot.id-1];
    return cost;
}

double TotalCost(const vector<PPCarrier>& vehicles, const vector<PPCity>& nodes, const unordered_map<int, PPCity>& depotMap) {
    double total = 0.0;
    for (const auto& v : vehicles)
        total += RouteCost(v.route, nodes, depotMap.at(v.depotID));
    return total;
}
vector<PPCarrier> CreateInitialSolution(const vector<PPCity>& nodes,
                                        const vector<pair<int, int>>& pdPairs,
                                        const vector<PPCity>& depots,
                                        int vehiclesPerDepot) {
    unordered_map<int, PPCity> depotMap;
    for (const auto& d : depots)
        depotMap[d.id] = d;

    vector<PPCarrier> vehicles;
    unordered_map<int, vector<int>> depotToVehicleIndices;

    // Create vehicles per depot and record their indices
    for (const auto& depot : depots) {
        for (int i = 0; i < vehiclesPerDepot; ++i) {
            PPCarrier v;
            v.capacity = 5000;
            v.load = 0;
            v.depotID = depot.id;
            vehicles.push_back(v);
            depotToVehicleIndices[depot.id].push_back(vehicles.size() - 1);
        }
    }
    vector<pair<double, int>> depotDistances;
    unordered_set<int> assignedOrders;

    for (const auto& pair : pdPairs) {
        int pid = pair.first;
        int did = pair.second;
        int orderId = nodes[pid].orderId;
        if (assignedOrders.count(orderId)) continue;

        double weight = nodes[pid].supply;
        bool assigned = false;

        // Step 1: sort depots by distance to pickup point

        for (const auto& depot : depots) {
            double d = distBtwCities[depot.id - 1][nodes[pid].id - 1];
            depotDistances.emplace_back(d, depot.id);
        }
        sort(depotDistances.begin(), depotDistances.end());

        // Step 2: for each depot (closest first), try assigning to a vehicle under that depot
        for (auto [_, depotID] : depotDistances) {
            for (int vidx : depotToVehicleIndices[depotID]) {
                PPCarrier& v = vehicles[vidx];
                if (v.load + weight <= v.capacity) {
                    v.route.push_back(pid);
                    v.route.push_back(did);
                    v.load += weight;
                    assignedOrders.insert(orderId);
                    assigned = true;
                    break;
                }
            }
            if (assigned) break;
        }

        if (!assigned) {
            cerr << "Warning: Order " << orderId << " could not be assigned to any vehicle.\n";
        }
        depotDistances.clear();
    }

    return vehicles;
}

// Validate route constraints: pickup before delivery, capacity not exceeded
bool IsValidRoute(const vector<int>& route, const vector<PPCity>& nodes, double capacity) {
    unordered_map<int, bool> pickedUp;
    double load = 0;

    for (int idx : route) {
        const PPCity& n = nodes[idx];
        if (n.isPickup) {
            load += n.supply;
            pickedUp[n.orderId] = true;
        } else {
            if (!pickedUp[n.orderId]) return false;
            load -= n.demand;
        }
        if (load > capacity) return false;
    }
    return true;
}

// Neighborhood: swap two full order pairs (pickup-delivery pair)
void SwapOrders(vector<PPCarrier>& vehicles, const vector<pair<int, int>>& pdPairs) {
    int maxAttempts = 100;
    while (maxAttempts--) {
        int v1 = rand() % vehicles.size();
        int v2 = rand() % vehicles.size();
        if (vehicles[v1].route.empty() || vehicles[v2].route.empty()) continue;
       int i = rand() % pdPairs.size();
    int j = rand() % pdPairs.size();
    if (i == j) continue;

    int pid1 = pdPairs[i].first, did1 = pdPairs[i].second;
    int pid2 = pdPairs[j].first, did2 = pdPairs[j].second;

    auto& r1 = vehicles[v1].route;
    auto& r2 = vehicles[v2].route;

    // Check existence
    bool r1_has_pair1 = find(r1.begin(), r1.end(), pid1) != r1.end() &&
                        find(r1.begin(), r1.end(), did1) != r1.end();
    bool r2_has_pair2 = find(r2.begin(), r2.end(), pid2) != r2.end() &&
                        find(r2.begin(), r2.end(), did2) != r2.end();
    if (!r1_has_pair1 || !r2_has_pair2) continue;

    // Safe swap
    r1.erase(remove(r1.begin(), r1.end(), pid1), r1.end());
    r1.erase(remove(r1.begin(), r1.end(), did1), r1.end());

    r2.erase(remove(r2.begin(), r2.end(), pid2), r2.end());
    r2.erase(remove(r2.begin(), r2.end(), did2), r2.end());

    r1.push_back(pid2);
    r1.push_back(did2);

    r2.push_back(pid1);
    r2.push_back(did1);

    return;
}
}

vector<PPCarrier> SimulatedAnnealingVRP(vector<PPCity>& nodes, vector<pair<int,int>>& pdPairs, vector<PPCity> depots, int vehiclesPerDepot) {
    unordered_map<int, PPCity> depotMap;
    for (auto& d : depots)
        depotMap[d.id] = d;

    vector<PPCarrier> current = CreateInitialSolution(nodes, pdPairs, depots, vehiclesPerDepot);
    for(auto v:current){
        for(auto r:v.route)
            cout<<r<<" ";
        cout<<endl;
    }
    vector<PPCarrier> best = current;
    double bestCost = TotalCost(best, nodes, depotMap);
    double temp = 1000.0, cooling = 0.995;
    int maxIter = 10000;

    srand(time(0));

    for (int iter = 0; iter < maxIter; ++iter) {
        vector<PPCarrier> neighbor = current;
        SwapOrders(neighbor, pdPairs);

        bool valid = true;
        for (auto& v : neighbor)
            if (!IsValidRoute(v.route, nodes, v.capacity)) valid = false;
        if (!valid) continue;

        double newCost = TotalCost(neighbor, nodes, depotMap);
        double delta = newCost - bestCost;

        if (delta < 0 || ((double)rand() / RAND_MAX) < exp(-delta / temp)) {
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


struct PersonalCarrier{
    static int n;
    int id;
    int hubId;
    double capacity;
    int sellerId;
    vector<Order> orders;
    PersonalCarrier(int h,int c=2000,int sid=-1):hubId(h)
    {
        id=++n;
    }

};
int PersonalCarrier::n=0;

void intro()
{
    clear_screen();
    auto start = chrono::high_resolution_clock::now();
    cout << "\n\n\n\n\n\n"
         << endl;
    cout << "                                                                                                                                                                                                     " << endl;                                                                                                                                                     
    cout << "                                                                                                                                                                                                     " << endl;
    cout << "    CCCCCCCCCCCCC                                                                                                                            iiii                            tttt                                     " << endl;
    cout << "  CCC::::::::::::C                                                                                                                          i::::i                        ttt:::t                                    " << endl;
    cout << "CC:::::::::::::::C                                                                                                                           iiii                         t:::::t                                    " << endl;
    cout << "C:::::CCCCCCCC::::C                                                                                                                                                       t:::::t                                   " << endl;
    cout << "C:::::C       CCCCCC      aaaaaaaaaaaaa       rrrrr   rrrrrrrrr          ggggggggg   ggggg     ooooooooooo        vvvvvvv           vvvvvvviiiiiii     ssssssssss   ttttttt:::::ttttttt      aaaaaaaaaaaaa         " << endl;
    cout << "C:::::C                   a::::::::::::a      r::::rrr:::::::::r        g:::::::::ggg::::g   oo:::::::::::oo       v:::::v         v:::::v i:::::i   ss::::::::::s  t:::::::::::::::::t      a::::::::::::a       " << endl;
    cout << "C:::::C                  aaaaaaaaa:::::a      r:::::::::::::::::r      g:::::::::::::::::g  o:::::::::::::::o       v:::::v       v:::::v   i::::i ss:::::::::::::s t:::::::::::::::::t      aaaaaaaaa:::::a      " << endl;
    cout << "C:::::C                           a::::a      rr::::::rrrrr::::::r    g::::::ggggg::::::gg  o:::::ooooo:::::o        v:::::v     v:::::v    i::::i s::::::ssss:::::stttttt:::::::tttttt               a::::a      " << endl;
    cout << "C:::::C                    aaaaaaa:::::a      r:::::r     r:::::r    g:::::g     g:::::g    o::::o     o::::o         v:::::v   v:::::v     i::::i  s:::::s  ssssss       t:::::t              aaaaaaa:::::a      " << endl;
    cout << "C:::::C                  aa::::::::::::a      r:::::r     rrrrrrr    g:::::g     g:::::g    o::::o     o::::o          v:::::v v:::::v      i::::i    s::::::s            t:::::t            aa::::::::::::a      " << endl;
    cout << " C:::::C                a::::aaaa::::::a      r:::::r                g:::::g     g:::::g    o::::o     o::::o           v:::::v:::::v       i::::i       s::::::s         t:::::t           a::::aaaa::::::a     " << endl;
    cout << "C:::::C       CCCCCC   a::::a    a:::::a      r:::::r                g::::::g    g:::::g    o::::o     o::::o            v:::::::::v        i::::i ssssss   s:::::s       t:::::t    tttttta::::a    a:::::a       " << endl;
    cout << "C:::::CCCCCCCC::::C    a::::a    a:::::a      r:::::r                g:::::::ggggg:::::g    o:::::ooooo:::::o             v:::::::v        i::::::is:::::ssss::::::s      t::::::tttt:::::ta::::a    a:::::a        " << endl;
    cout << "CC:::::::::::::::C     a:::::aaaa::::::a      r:::::r                 g::::::::::::::::g    o:::::::::::::::o              v:::::v         i::::::is::::::::::::::s       tt::::::::::::::ta:::::aaaa::::::a         " << endl;
    cout << " CCC::::::::::::C      a::::::::::aa:::a      r:::::r                  gg::::::::::::::g     oo:::::::::::oo                v:::v          i::::::i s:::::::::::ss          tt:::::::::::tt a::::::::::aa:::a         " << endl;
    cout << "    CCCCCCCCCCCCC       aaaaaaaaaa  aaaa      rrrrrrr                    gggggggg::::::g       ooooooooooo                   vvv           iiiiiiii  sssssssssss              ttttttttttt    aaaaaaaaaa  aaaa         " << endl;
    cout << "                                                                                 g:::::g                                                                                                                            " << endl;
    cout << "                                                                    gggggg       g:::::g                                                                                                                            " << endl;
    cout << "                                                                    g:::::gg    gg:::::g                                                                                                                            " << endl;
    cout << "                                                                      g::::::ggg:::::::g                                                                                                                           " << endl;
    cout << "                                                                        gg:::::::::::::g                                                                                                                            " << endl;
    cout << "                                                                          ggg::::::ggg                                                                                                                              " << endl;
    cout << "                                                                             gggggg                                                                                                                                 " << endl;                                                                                                                               " << endl;
   
    while (true)
    {
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::seconds>(end - start).count();
        if (duration >= 5)
        {
            clear_screen();
            break;
        }
        this_thread::sleep_for(chrono::milliseconds(100));
    }
}

// EXIT SCREEN
void exitscr()
{
    auto start = chrono::high_resolution_clock::now();
    cout << endl
         << endl
         << endl
         << endl
         << endl
         << endl
         << endl
         << endl
         << endl;
    cout << "                                          *                                 *                                                    \n";
    cout << "                                  *     **                                **                                                     \n";
    cout << "                                 **     **                                **                                                     \n";
    cout << "                                 **     **                                **                                                     \n";
    cout << "                               ******** **                                **            **   ****         ****    **   ****     \n";
    cout << "                              ********  **  ***      ****    ***  ****    **  ***        **    ***  *    * ***  *  **    ***  *  \n";
    cout << "                                 **     ** * ***    * ***  *  **** **** * ** * ***       **     ****    *   ****   **     ****  \n";
    cout << "                                 **     ***   ***  *   ****    **   ****  ***   *        **      **    **    **    **      **   \n";
    cout << "                                 **     **     ** **    **     **    **   **   *         **      **    **    **    **      **   \n";
    cout << "                                 **     **     ** **    **     **    **   **  *          **      **    **    **    **      **   \n";
    cout << "                                 **     **     ** **    **     **    **   ** **          **      **    **    **    **      **   \n";
    cout << "                                 **     **     ** **    **     **    **   ******         **      **    **    **    **      **   \n";
    cout << "                                 **     **     ** **    **     **    **   **  ***         *********     ******      ******* **  \n";
    cout << "                                  **    **     **  ***** **    ***   ***  **   *** *        **** ***     ****        *****   **  \n";
    cout << "                                         **    **   ***   **    ***   ***  **   ***               ***                           \n";
    cout << "                                               *                                           *****   ***                          \n";
    cout << "                                              *                                          ********  **                           \n";
    cout << "                                             *                                          *      ****                              \n";
    cout << "                                            *                                                                                   \n";
    cout << "                                                                                                                                \n";
    cout << "                                 ***                                                                                           \n";
    cout << "                               ** ***                                                        *                                 \n";
    cout << "                              **   ***                                                      ***                                \n";
    cout << "                              **                                                             *                                 \n";
    cout << "                              **          ****    ***  ****       **   ****        ****                                        \n";
    cout << "                              ******     * ***  *  **** **** *     **    ***  *   * **** * ***     ***  ****        ****      \n";
    cout << "                              *****     *   ****    **   ****      **     ****   **  ****   ***     **** **** *    *  ***  *  \n";
    cout << "                              **       **    **     **             **      **   ****         **      **   ****    *    ****   \n";
    cout << "                              **       **    **     **             **      **     ***        **      **    **    **     **    \n";
    cout << "                              **       **    **     **             **      **       ***      **      **    **    **     **    \n";
    cout << "                              **       **    **     **             **      **         ***    **      **    **    **     **    \n";
    cout << "                              **       **    **     **             **      **    ****  **    **      **    **    **     **    \n";
    cout << "                              **        ******      ***             ******* **  * **** *     **      **    **    **     **    \n";
    cout << "                              **         ****        ***             *****   **    ****      *** *   ***   ***    ********    \n";
    cout << "                               **                                                             ***     ***   ***     *** ***   \n";
    cout << "                                                                                                                         ***  \n";
    cout << "                                                                                                                   ****   *** \n";
    cout << "                                                                                                                 *******  **  \n";
    cout << "                                                                                                                *     ****    \n";

    while (true)
    {
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::seconds>(end - start).count();
        if (duration >= 3)
        {
            clear_screen();
            break;
        }
        this_thread::sleep_for(chrono::milliseconds(100));
    }

    clear_screen();
    intro();
    exit(0);
}


void get_console_size(int &width, int &height) {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    int columns, rows;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    columns = csbi.srWindow.Right - csbi.srWindow.Left + 1;
    rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
    width = columns;
    height = rows;
#else
    width = 80;
    height = 25;
#endif
}

void pause() {
    cout << "\nPress Enter to continue...";
    cin.ignore();
    cin.get();
}


// Dummy placeholders for actual logic
bool uploadOrderData() {
    cout << "\nUploading order data...\n";
    return true;
}

bool setPriorities() {
    cout << "\nSetting order priorities...\n";
    return true;
}

bool chooseOptimizationGoal() {
    cout << "\nChoosing optimization goal...\n";
    return true;
}

void runSimulation() {
    cout << "\nRunning simulation using provided preferences...\n";
    cout << "\nSimulation complete.\n";
}

void generateRecommendation() {
    cout << "\nAnalyzing results and generating recommendation...\n";
    cout << "\nRecommended model: Personalized Carrier (based on your preferences).\n";
}

void clear_screen() {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

void center_print(const string& text) {
    int width = 80;
    int pad = max((width - (int)text.length()) / 2, 0);
    cout << string(pad, ' ') << text << endl;
}

void main_menu() {
    clear_screen();
    center_print("+---------------------------------------------------------+");
    center_print("|                Welcome to CargoVista                   |");
    center_print("+---------------------------------------------------------+");
    center_print(" CargoVista is a simulation-based logistics tracker and   ");
    center_print(" advisor designed to empower sellers with strategic       ");
    center_print(" delivery insights. Our platform compares three core       ");
    center_print(" delivery models — Hub-and-Spoke, Point-to-Point,         ");
    center_print(" and Personalized Carrier — and recommends the best fit   ");
    center_print(" for your needs.                                          ");
    center_print(" Whether you're aiming to minimize costs, reduce delivery ");
    center_print(" time, or prioritize specific orders, CargoVista simulates");
    center_print(" all models for you.                                      ");
    center_print(" Analyze. Compare. Deliver smarter — every time.          ");
    center_print("                                                         ");
    center_print("+---------------------------------------------------------+");
    center_print("|                 CargoVista Roadmap                     |");
    center_print("+---------------------------------------------------------+");
    center_print("| Step 1: Orders & Preferences                           |");
    center_print("|         Upload orders, set delivery goals              |");
    center_print("|         - Upload Order Data                            |");
    center_print("|         - Set Delivery Priorities                      |");
    center_print("|         - Choose Optimization Goal                     |");
    center_print("|                                                        |");
    center_print("| Step 2: Simulate & Compare                             |");
    center_print("|         Run all delivery models & view outcomes        |");
    center_print("|                                                        |");
    center_print("| Step 3: Get Recommendation                             |");
    center_print("|         Our suggestion for optimal delivery strategy   |");
    center_print("+---------------------------------------------------------+\n");

    cout << "Press Enter to begin...";
    cin.ignore();
    cin.get();
}

void orders_preferences_flow() {
    clear_screen();
    center_print("------------------------------");
    center_print(" Order Preferences Setup ");
    center_print("------------------------------\n");

    bool step1 = false, step2 = false, step3 = false;

    do {
        center_print("Step 1: Upload Order Data");
        step1 = uploadOrderData();
        if (!step1) {
            cout << "\nFailed to upload order data. Try again.\n";
        }
    } while (!step1);

    do {
        center_print("\nStep 2: Set Order Priorities");
        step2 = setPriorities();
        if (!step2) {
            cout << "\nInvalid priority settings. Try again.\n";
        }
    } while (!step2);

    do {
        center_print("\nStep 3: Choose Optimization Goal");
        step3 = chooseOptimizationGoal();
        if (!step3) {
            cout << "\nInvalid selection. Try again.\n";
        }
    } while (!step3);

    cout << "\nAll preferences configured.\n";
    runSimulation();
    generateRecommendation();
}


void setConsoleColors() {
    // Set background to olive green and text to light yellow (beige approximation)
    system("color 6A"); 
    // '6' = olive/dark yellow background, 'A' = light green/yellow text
}


int main() {
    setConsoleColors();
    vector<City> cities = {
        {1, "Delhi", 700, 220},
    {2, "Amritsar", 640, 130},
    {3, "Chandigarh", 670, 150},
        {4, "Jaipur", 600, 350},
    {5, "Lucknow", 690, 320},
    {6, "Kanpur", 675, 340},
    {7, "Agra", 670, 275},
    {8, "Varanasi", 750, 375},
    {9, "Meerut", 720, 250},
        {10, "Aligarh", 690, 260},
    {11, "Patna", 770, 410},
    {12, "Ghaziabad", 715, 230},
        {13, "Moradabad", 730, 265},
    {14, "Bareilly", 705, 280},
    {15, "Saharanpur", 690, 205},
        {16, "Haridwar", 670, 190},
    {17, "Roorkee", 665, 180},
    {18, "Rishikesh", 660, 170},
        {19, "Nainital", 655, 150},
    {20, "Mathura", 680, 265},
    {21, "Hoshiarpur", 645, 140},
        {22, "Kullu", 655, 120},
    {23, "Shimla", 660, 110},
    {24, "Kangra", 650, 125},
        {25, "Solan", 660, 130},
    {26, "Srinagar", 610, 90},
    {27, "Jammu", 625, 100},
        {28, "Ludhiana", 660, 180},
    {29, "Patiala", 650, 170},
    {30, "Panipat", 705, 240},
        {31, "Sonipat", 710, 230},
    {32, "Muzaffarnagar", 720, 260},
    {33, "Fatehpur", 690, 310},
        {34, "Karnal", 705, 230},
    {35, "Bhiwani", 680, 245},
    {36, "Hisar", 670, 275},
        {37, "Jind", 660, 280},
    {38, "Kurukshetra", 715, 220},
    {39, "Rohtak", 720, 230},
        {40, "Faridabad", 700, 215},
    {41, "Barnala", 640, 150},
    {42, "Muktsar", 635, 135},
        {43, "Sangrur", 630, 140},
    {44, "Bhatinda", 625, 160},
    {45, "Jalandhar", 650, 160},
        {46, "Ambala", 675, 160},
    {47, "Gurugram", 710, 210},
    {48, "Noida", 715, 220},
        {49, "Farukhabad", 755, 360}
    };

    unordered_map<string,int>cityToId;
    for(int i=1;i<=49;i++)
    {
        cityToId[cities[i-1].name]=i;
    }


    vector<vector<double>> adj_matrix = {
    {0.00, 0.00, 99.01, 0.00, 130.65, 0.00, 0.00, 0.00, 0.00, 0.00, 263.23, 0.00, 0.00, 78.27, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 152.16, 139.56, 0.00, 0.00, 0.00, 73.54, 0.00, 0.00, 0.00, 58.14, 117.72, 0.00, 0.00, 81.44, 0.00, 0.00, 0.00, 6.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 19.50, 195.54},
    {0.00, 0.00, 0.00, 0.00, 255.41, 276.77, 192.49, 0.00, 187.49, 181.07, 401.32, 0.00, 0.00, 0.00, 0.00, 0.00, 72.67, 0.00, 32.50, 0.00, 14.53, 23.44, 36.77, 0.00, 26.00, 0.00, 43.60, 70.01, 53.60, 166.10, 0.00, 198.44, 0.00, 0.00, 158.29, 0.00, 0.00, 0.00, 0.00, 135.26, 26.00, 0.00, 0.00, 43.60, 0.00, 0.00, 138.19, 0.00, 0.00},
    {99.01, 0.00, 0.00, 0.00, 0.00, 247.09, 0.00, 310.44, 0.00, 0.00, 362.14, 0.00, 0.00, 0.00, 0.00, 52.00, 0.00, 0.00, 19.50, 0.00, 0.00, 43.60, 53.60, 0.00, 0.00, 110.31, 87.45, 41.11, 0.00, 0.00, 0.00, 0.00, 209.62, 0.00, 124.18, 0.00, 0.00, 0.00, 122.64, 93.07, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 133.37, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 269.65, 0.00, 0.00, 0.00, 0.00, 0.00, 296.45, 0.00, 0.00, 0.00, 0.00, 0.00, 211.62, 0.00, 0.00, 207.29, 171.60, 133.37, 0.00, 225.64, 220.62, 218.40, 0.00, 0.00, 0.00, 249.13, 0.00, 0.00, 231.46, 0.00, 201.92},
    {130.65, 255.41, 0.00, 0.00, 0.00, 32.50, 64.02, 0.00, 0.00, 78.00, 0.00, 0.00, 88.41, 0.00, 0.00, 170.99, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 298.22, 0.00, 0.00, 105.81, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 137.12, 0.00, 250.90, 246.66, 0.00, 0.00, 0.00, 145.34, 0.00, 0.00},
    {0.00, 276.77, 247.09, 0.00, 32.50, 0.00, 0.00, 0.00, 130.81, 105.81, 153.41, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 262.91, 0.00, 299.64, 0.00, 273.70, 0.00, 0.00, 0.00, 0.00, 135.72, 150.06, 0.00, 43.60, 0.00, 123.67, 0.00, 0.00, 0.00, 154.50, 0.00, 0.00, 271.53, 266.50, 0.00, 236.25, 0.00, 0.00, 164.44, 0.00},
    {0.00, 192.49, 0.00, 133.37, 64.02, 0.00, 0.00, 0.00, 0.00, 32.50, 218.40, 82.73, 79.08, 0.00, 94.64, 110.50, 123.67, 0.00, 0.00, 0.00, 178.48, 202.44, 0.00, 0.00, 188.95, 252.83, 234.90, 0.00, 0.00, 0.00, 78.27, 67.86, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 183.04, 0.00, 151.74, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 310.44, 0.00, 0.00, 0.00, 0.00, 0.00, 167.11, 0.00, 0.00, 193.91, 145.34, 0.00, 234.36, 0.00, 0.00, 0.00, 0.00, 169.50, 0.00, 0.00, 363.83, 0.00, 339.31, 0.00, 392.70, 0.00, 0.00, 0.00, 0.00, 0.00, 115.00, 0.00, 191.94, 0.00, 170.12, 0.00, 0.00, 217.92, 325.58, 0.00, 0.00, 323.31, 308.25, 296.02, 0.00, 0.00, 0.00},
    {0.00, 187.49, 0.00, 0.00, 0.00, 130.81, 0.00, 167.11, 0.00, 41.11, 0.00, 26.80, 0.00, 43.60, 70.31, 0.00, 115.73, 0.00, 0.00, 0.00, 173.08, 188.95, 198.01, 186.25, 0.00, 0.00, 0.00, 0.00, 0.00, 23.44, 0.00, 13.00, 0.00, 0.00, 0.00, 72.67, 87.21, 0.00, 0.00, 0.00, 0.00, 0.00, 184.76, 0.00, 0.00, 130.81, 0.00, 0.00, 150.06},
    {0.00, 181.07, 0.00, 0.00, 78.00, 105.81, 32.50, 0.00, 41.11, 0.00, 221.00, 50.77, 52.40, 0.00, 71.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 183.04, 173.44, 244.25, 0.00, 111.07, 0.00, 0.00, 46.87, 39.00, 0.00, 43.60, 0.00, 0.00, 0.00, 0.00, 55.15, 59.93, 0.00, 0.00, 0.00, 0.00, 0.00, 131.45, 0.00, 0.00, 0.00},
    {263.23, 401.32, 362.14, 0.00, 0.00, 153.41, 218.40, 0.00, 0.00, 221.00, 0.00, 244.68, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 221.86, 0.00, 0.00, 0.00, 0.00, 391.08, 0.00, 0.00, 0.00, 348.83, 0.00, 246.66, 205.55, 166.48, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 395.38, 0.00, 360.50, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 82.73, 193.91, 26.80, 50.77, 244.68, 0.00, 49.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 64.35, 148.22, 162.89, 0.00, 0.00, 0.00, 0.00, 205.55, 0.00, 0.00, 0.00, 0.00, 0.00, 108.96, 13.00, 0.00, 0.00, 96.63, 13.00, 6.50, 0.00, 0.00, 0.00, 0.00, 148.22, 124.18, 104.81, 26.80, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 88.41, 0.00, 79.08, 145.34, 0.00, 52.40, 0.00, 49.50, 0.00, 0.00, 93.74, 124.86, 0.00, 153.41, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 197.69, 0.00, 254.25, 143.15, 161.46, 45.96, 0.00, 14.53, 78.27, 0.00, 0.00, 0.00, 0.00, 61.66, 47.32, 75.80, 0.00, 0.00, 0.00, 0.00, 0.00, 154.09, 0.00, 0.00, 0.00},
    {78.27, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 43.60, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 154.50, 0.00, 0.00, 0.00, 0.00, 228.61, 0.00, 0.00, 276.15, 256.07, 0.00, 0.00, 0.00, 0.00, 32.50, 43.60, 0.00, 0.00, 0.00, 0.00, 0.00, 67.86, 84.75, 0.00, 0.00, 206.47, 0.00, 171.60, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 94.64, 234.36, 70.31, 71.50, 0.00, 0.00, 93.74, 0.00, 0.00, 32.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 50.77, 0.00, 0.00, 41.62, 0.00, 0.00, 0.00, 0.00, 94.64, 0.00, 37.90, 50.77, 18.38, 96.63, 115.73, 0.00, 0.00, 0.00, 0.00, 26.80, 0.00, 0.00},
    {0.00, 0.00, 52.00, 0.00, 170.99, 0.00, 110.50, 0.00, 0.00, 0.00, 0.00, 0.00, 124.86, 0.00, 32.50, 0.00, 0.00, 29.07, 0.00, 0.00, 0.00, 93.07, 104.81, 0.00, 79.08, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 158.15, 0.00, 0.00, 110.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 70.31, 0.00},
    {0.00, 72.67, 0.00, 0.00, 0.00, 0.00, 123.67, 0.00, 115.73, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 91.23, 0.00, 0.00, 0.00, 0.00, 6.50, 23.44, 0.00, 0.00, 0.00, 0.00, 0.00, 86.72, 123.67, 0.00, 0.00, 0.00, 0.00, 50.77, 0.00, 0.00, 0.00, 0.00, 29.07, 70.31, 0.00, 261.62},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 153.41, 154.50, 0.00, 29.07, 0.00, 0.00, 0.00, 126.21, 0.00, 65.32, 0.00, 0.00, 0.00, 0.00, 0.00, 13.00, 13.00, 0.00, 101.53, 0.00, 0.00, 0.00, 0.00, 137.12, 0.00, 0.00, 0.00, 78.27, 0.00, 0.00, 55.15, 0.00, 18.38, 0.00, 83.24, 0.00, 0.00},
    {0.00, 32.50, 19.50, 269.65, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 18.38, 39.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 212.92, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 32.50, 0.00, 41.11, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 169.50, 0.00, 0.00, 221.86, 64.35, 0.00, 0.00, 0.00, 0.00, 0.00, 126.21, 0.00, 0.00, 0.00, 0.00, 203.17, 0.00, 0.00, 0.00, 226.10, 113.52, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 26.00, 18.38, 0.00, 0.00, 0.00, 0.00, 0.00, 178.84, 0.00, 0.00, 0.00, 0.00, 0.00, 74.11, 0.00},
    {0.00, 14.53, 0.00, 0.00, 0.00, 262.91, 178.48, 0.00, 173.08, 0.00, 0.00, 148.22, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 18.38, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 144.32, 183.96, 0.00, 0.00, 0.00, 0.00, 0.00, 138.19, 0.00, 0.00, 0.00, 14.53, 0.00, 0.00, 26.80, 0.00, 124.18, 0.00, 0.00},
    {0.00, 23.44, 43.60, 0.00, 0.00, 0.00, 202.44, 0.00, 188.95, 0.00, 0.00, 162.89, 0.00, 0.00, 0.00, 93.07, 0.00, 65.32, 39.00, 0.00, 0.00, 0.00, 14.53, 9.19, 0.00, 70.31, 0.00, 0.00, 0.00, 0.00, 0.00, 200.66, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 166.10, 0.00, 0.00, 32.50, 41.62, 0.00, 52.40, 0.00, 0.00, 0.00, 0.00},
    {152.16, 36.77, 53.60, 0.00, 0.00, 299.64, 0.00, 363.83, 198.01, 0.00, 0.00, 0.00, 0.00, 228.61, 0.00, 104.81, 91.23, 0.00, 0.00, 203.17, 0.00, 14.53, 0.00, 0.00, 0.00, 70.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 177.42, 0.00, 0.00, 0.00, 174.41, 0.00, 0.00, 45.96, 0.00, 79.34, 0.00, 0.00, 0.00, 0.00, 0.00},
    {139.56, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 186.25, 183.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 9.19, 0.00, 0.00, 0.00, 69.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 55.92, 45.50, 0.00, 0.00, 149.64, 334.61},
    {0.00, 26.00, 0.00, 296.45, 0.00, 273.70, 188.95, 339.31, 0.00, 173.44, 391.08, 0.00, 197.69, 0.00, 0.00, 79.08, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 65.00, 0.00, 0.00, 0.00, 186.13, 0.00, 0.00, 0.00, 0.00, 0.00, 137.12, 151.60, 0.00, 0.00, 0.00, 0.00, 59.93, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 110.31, 0.00, 0.00, 0.00, 252.83, 0.00, 0.00, 244.25, 0.00, 0.00, 0.00, 276.15, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 70.31, 70.01, 69.10, 0.00, 0.00, 23.44, 0.00, 116.28, 230.82, 223.66, 0.00, 0.00, 219.95, 0.00, 0.00, 0.00, 0.00, 231.46, 0.00, 87.21, 0.00, 70.01, 0.00, 104.81, 0.00, 0.00, 0.00, 0.00},
    {0.00, 43.60, 87.45, 0.00, 298.22, 0.00, 234.90, 392.70, 0.00, 0.00, 0.00, 205.55, 254.25, 256.07, 0.00, 0.00, 0.00, 0.00, 0.00, 226.10, 0.00, 0.00, 0.00, 0.00, 0.00, 23.44, 0.00, 113.52, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 201.60, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 101.53, 0.00, 195.00, 0.00},
    {73.54, 70.01, 41.11, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 111.07, 0.00, 0.00, 143.15, 0.00, 50.77, 0.00, 6.50, 13.00, 0.00, 113.52, 0.00, 0.00, 0.00, 0.00, 65.00, 0.00, 113.52, 0.00, 0.00, 0.00, 0.00, 0.00, 173.44, 0.00, 0.00, 124.18, 0.00, 0.00, 0.00, 69.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 75.80, 0.00, 0.00},
    {0.00, 53.60, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 348.83, 0.00, 161.46, 0.00, 0.00, 0.00, 23.44, 13.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 116.28, 0.00, 0.00, 0.00, 0.00, 110.31, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 106.61, 119.85, 87.45, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 93.74, 0.00, 282.21},
    {0.00, 166.10, 0.00, 0.00, 105.81, 135.72, 0.00, 0.00, 23.44, 0.00, 0.00, 0.00, 45.96, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 230.82, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 23.44, 0.00, 0.00, 0.00, 162.50, 0.00, 0.00, 111.07, 0.00, 29.07, 0.00},
    {0.00, 0.00, 0.00, 211.62, 0.00, 150.06, 78.27, 0.00, 0.00, 46.87, 246.66, 0.00, 0.00, 0.00, 41.62, 0.00, 0.00, 101.53, 0.00, 0.00, 144.32, 0.00, 0.00, 0.00, 0.00, 223.66, 0.00, 0.00, 110.31, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 78.27, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 119.85, 0.00, 0.00, 0.00, 178.84},
    {58.14, 198.44, 0.00, 0.00, 0.00, 0.00, 67.86, 0.00, 13.00, 39.00, 205.55, 0.00, 14.53, 32.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 183.96, 200.66, 0.00, 0.00, 186.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 43.60, 0.00, 67.86, 0.00, 0.00, 39.00, 0.00, 0.00, 0.00, 0.00, 0.00, 158.69, 0.00, 66.29, 0.00, 137.73},
    {117.72, 0.00, 209.62, 0.00, 0.00, 43.60, 0.00, 115.00, 0.00, 0.00, 166.48, 108.96, 78.27, 43.60, 0.00, 158.15, 0.00, 0.00, 212.92, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 173.44, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 52.40, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 234.36, 212.52, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 207.29, 0.00, 0.00, 0.00, 0.00, 0.00, 43.60, 0.00, 13.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 219.95, 0.00, 0.00, 0.00, 0.00, 0.00, 43.60, 0.00, 0.00, 0.00, 0.00, 87.45, 0.00, 0.00, 0.00, 134.00, 0.00, 0.00, 0.00, 115.73, 0.00, 0.00, 18.38, 181.07},
    {0.00, 158.29, 124.18, 171.60, 0.00, 123.67, 0.00, 191.94, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 86.72, 0.00, 0.00, 26.00, 0.00, 0.00, 177.42, 0.00, 0.00, 0.00, 201.60, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 55.54, 46.87, 134.00, 0.00, 151.19, 0.00, 0.00, 0.00, 0.00, 0.00, 178.48},
    {81.44, 0.00, 0.00, 133.37, 0.00, 0.00, 0.00, 0.00, 72.67, 0.00, 0.00, 0.00, 0.00, 0.00, 94.64, 110.50, 123.67, 137.12, 0.00, 18.38, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 124.18, 0.00, 0.00, 78.27, 67.86, 52.40, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 183.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 170.12, 87.21, 0.00, 0.00, 96.63, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 87.45, 0.00, 0.00, 0.00, 0.00, 101.53, 0.00, 170.99, 0.00, 186.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 225.64, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 13.00, 61.66, 0.00, 37.90, 0.00, 0.00, 0.00, 0.00, 0.00, 138.19, 0.00, 0.00, 0.00, 137.12, 0.00, 0.00, 0.00, 106.61, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 14.53, 20.55, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 189.28},
    {0.00, 0.00, 122.64, 220.62, 0.00, 154.50, 0.00, 0.00, 0.00, 55.15, 0.00, 6.50, 47.32, 67.86, 50.77, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 166.10, 174.41, 0.00, 151.60, 231.46, 0.00, 0.00, 119.85, 23.44, 0.00, 39.00, 0.00, 0.00, 55.54, 0.00, 101.53, 14.53, 0.00, 32.50, 0.00, 0.00, 0.00, 153.41, 0.00, 108.18, 0.00, 0.00, 0.00},
    {6.50, 135.26, 93.07, 218.40, 137.12, 0.00, 0.00, 217.92, 0.00, 59.93, 0.00, 0.00, 75.80, 84.75, 18.38, 0.00, 0.00, 78.27, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 69.10, 87.45, 0.00, 0.00, 0.00, 0.00, 0.00, 46.87, 0.00, 0.00, 20.55, 32.50, 0.00, 0.00, 134.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 26.00, 0.00, 0.00, 0.00, 0.00, 0.00, 325.58, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 96.63, 0.00, 50.77, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 87.21, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 134.00, 134.00, 0.00, 170.99, 0.00, 0.00, 0.00, 0.00, 20.55, 18.38, 23.44, 18.38, 0.00, 0.00, 0.00, 311.25},
    {0.00, 0.00, 0.00, 0.00, 250.90, 271.53, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 115.73, 0.00, 0.00, 0.00, 32.50, 178.84, 14.53, 32.50, 45.96, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 134.00, 20.55, 0.00, 0.00, 0.00, 0.00, 0.00, 137.89, 0.00, 331.50},
    {0.00, 0.00, 0.00, 0.00, 246.66, 266.50, 183.04, 0.00, 184.76, 0.00, 395.38, 0.00, 0.00, 206.47, 0.00, 0.00, 0.00, 55.15, 0.00, 0.00, 0.00, 41.62, 0.00, 0.00, 0.00, 70.01, 0.00, 0.00, 0.00, 162.50, 0.00, 0.00, 234.36, 0.00, 151.19, 183.04, 186.13, 0.00, 0.00, 0.00, 18.38, 0.00, 0.00, 26.80, 36.77, 64.02, 0.00, 0.00, 328.94},
    {0.00, 43.60, 0.00, 249.13, 0.00, 0.00, 0.00, 323.31, 0.00, 0.00, 0.00, 148.22, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 41.11, 0.00, 0.00, 0.00, 79.34, 55.92, 59.93, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 212.52, 0.00, 0.00, 0.00, 0.00, 0.00, 153.41, 0.00, 23.44, 0.00, 26.80, 0.00, 0.00, 0.00, 0.00, 140.62, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 236.25, 151.74, 308.25, 0.00, 0.00, 360.50, 124.18, 0.00, 171.60, 0.00, 0.00, 0.00, 18.38, 0.00, 0.00, 26.80, 52.40, 0.00, 45.50, 0.00, 104.81, 0.00, 0.00, 0.00, 0.00, 119.85, 158.69, 0.00, 115.73, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 18.38, 0.00, 36.77, 0.00, 0.00, 0.00, 0.00, 115.00, 293.65},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 296.02, 130.81, 131.45, 0.00, 104.81, 154.09, 0.00, 0.00, 0.00, 29.07, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 101.53, 0.00, 0.00, 111.07, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 108.18, 0.00, 0.00, 0.00, 64.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 138.19, 0.00, 231.46, 145.34, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 26.80, 0.00, 0.00, 26.80, 0.00, 70.31, 83.24, 0.00, 0.00, 124.18, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 75.80, 93.74, 0.00, 0.00, 66.29, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 137.89, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 203.59},
    {19.50, 0.00, 0.00, 0.00, 0.00, 164.44, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 70.31, 0.00, 0.00, 0.00, 74.11, 0.00, 0.00, 0.00, 149.64, 0.00, 0.00, 195.00, 0.00, 0.00, 29.07, 0.00, 0.00, 0.00, 18.38, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 140.62, 115.00, 0.00, 0.00, 0.00, 0.00},
    {195.54, 0.00, 0.00, 201.92, 0.00, 0.00, 0.00, 0.00, 150.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 261.62, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 334.61, 0.00, 0.00, 0.00, 0.00, 282.21, 0.00, 178.84, 137.73, 0.00, 181.07, 178.48, 0.00, 0.00, 189.28, 0.00, 0.00, 311.25, 331.50, 328.94, 0.00, 293.65, 0.00, 203.59, 0.00, 0.00},
};
    
     main_menu();
    
    distBtwCities=floydWarshallFromAdjMatrix(adj_matrix);

    int k = 10;
    double wcss = 0;
    auto clusters = kMeansClustering(cities, k, wcss);
    vector<City>hubs;
    for(auto cluster:clusters)
    {
        hubs.push_back(findHubs(cluster,distBtwCities));
    }

    for (int i = 0; i < clusters.size(); ++i) {
        cout << "Cluster " << i + 1 << ":\n";
        for (const City& c : clusters[i]) {
            cout << "  " << c.id << " - " << c.name << " (" << c.x << ", " << c.y << ")\n";
            spokeToHub[c.id]=hubs[i].id;
        }
        cout<<"Hub:"<<hubs[i].name<<endl;
        cout << "\n";
    }

    int cnt1=1,cnt2=1;
    for(int i=1;i<=49;i++)
    {
        for(int j=i+1;j<=49;j++)
        {
            if(spokeToHub[i]!=i)
            {
                locToHSCarrier[{i,spokeToHub[i]}]=*(new HubSpokeCarrier(cnt1++,i));
                break;
            }
            else if(spokeToHub[j]==j)
            {
                locToHHCarrier[{i,j}]=*(new HubHubCarrier(cnt2++,i,j));
            }
        }
    }

    cout << "Total WCSS: " << wcss << endl;

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> locationDist(1, 50);

    vector<Seller> sellers;

    // Create 5 sellers with random location IDs
    for (int i = 1; i <= 5; ++i) {
        int locId = locationDist(gen);
        sellers.push_back(*(new Seller(i,locId)));
    }

    // Generate 30 random orders
    vector<Order> simulatedOrders = generateRandomOrders(30);

    // Assign orders to sellers
    for (const auto& order : simulatedOrders) {
        int index = order.sellerId - 1;
        if (index >= 0 && index < sellers.size()) {
            sellers[index].addOrder(order);
        }
    }

    // Print sample output
    for (const auto& seller : sellers) {
        cout << "Seller " << seller.sellerId << " (Location ID: " << seller.location << ") has " << seller.orders.size() << " orders.\n";
    }
int ch;
cout<<"Enter ch:";
cin>>ch;
switch(ch)
{
case 1:
      for(auto order:simulatedOrders)
    {
        processOrder(order);
    }

    int sid;
    cout<<"Enter seller id if existing seller, else 0";
    cin>>sid;
    if(sid<=sellers.size() && sid>0)
    {
        Seller curSeller=sellers[sid];
        int ch=1;
        string dest;
        double wt;
        double vol;
        int destId;
        while(ch)
        {
            cout<<"Enter destination(only first letter capital),weight,volume:";
            cin>>dest>>wt>>vol;
            while(!cityToId.count(dest))
            {
                cout<<"Invalid destination!\nRe-enter the destination:";
                cin>>dest;
            }
            destId=cityToId[dest];
            Order* order=new Order(sid,sellers[sid].location,destId,wt,vol);
            sellers[sid].addOrder(*order);
            processOrder(*order);
            cout<<"press 1 for more orders, else 0";
            cin>>ch;
        }
    }
    else
    {
        cout<<"invalid seller!";
    }
    break;

case 2:
      // Create depot list
    vector<PPCity> depots;
    for (City hub : hubs) {
        depots.push_back(PPCity(hub.id, 0, 0));
    }

 vector<PPCity> nodes;
vector<pair<int, int>> pdPairs;

for (auto& order : simulatedOrders) {
    int pickupIdx = nodes.size();
    nodes.push_back(PPCity(order.source, 0, order.weight, order.orderId, true));

    int deliveryIdx = nodes.size();
    nodes.push_back(PPCity(order.destination, order.weight, 0, order.orderId, false));

    pdPairs.push_back({pickupIdx, deliveryIdx});
    cout<<"order id:"<<order.orderId<<endl;
    cout<<"source:"<<cities[order.source-1].name<<endl;
    cout<<"destination:"<<cities[order.destination-1].name<<endl;
    cout<<endl;
}


    // Simulated Annealing to find the optimal routes
    int vehiclesPerDepot = 2;
    vector<PPCarrier> bestSolution = SimulatedAnnealingVRP(nodes, pdPairs, depots, vehiclesPerDepot);

    // Output the routes
    for (int v = 0; v < bestSolution.size(); ++v) {
        cout << "Vehicle " << v << " (Depot " << bestSolution[v].depotID << "): ";
        for (int i = 0; i < bestSolution[v].route.size(); ++i) {
            cout << nodes[bestSolution[v].route[i]].id;
            if (i < bestSolution[v].route.size() - 1)
                cout << " -> ";
        }
        cout << endl;
    }

case 3:

    break;

}
    return 0;
}
