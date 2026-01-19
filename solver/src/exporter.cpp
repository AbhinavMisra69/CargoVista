// exporter.cpp
int main() {
    // 1. Run your existing Floyd-Warshall function
    vector<vector<double>> distMatrix = floydWarshallFromAdjMatrix(adj_matrix);
    // Also perform Clustering and Hub finding here!
    double wcss = 0;
    auto clusters = kMeansClustering(cities, 10, wcss);

    // 2. Output as a JSON object
    json initData;
    initData["matrix"] = distMatrix;
    initData["spokeToHub"] = spokeToHub; // Your global map
    
    cout << initData.dump() << endl;
    return 0; // Exit after printing the init data
}
