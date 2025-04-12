#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <string>
#include <cstdlib>
#include <ctime>

using namespace std;

struct City {
    int id;
    string name;
    int x, y;

    City(int id, const string& name, int x, int y)
        : id(id), name(name), x(x), y(y) {}
};

double euclidean(const City& a, const City& b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
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

int main() {
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

    int k = 10;
    double wcss = 0;
    auto clusters = kMeansClustering(cities, k, wcss);

    for (int i = 0; i < clusters.size(); ++i) {
        cout << "Cluster " << i + 1 << ":\n";
        for (const City& c : clusters[i]) {
            cout << "  " << c.id << " - " << c.name << " (" << c.x << ", " << c.y << ")\n";
        }
        cout << "\n";
    }

    cout << "Total WCSS: " << wcss << endl;
    return 0;
}
