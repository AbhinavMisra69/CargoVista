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

void floydWarshallFromAdjMatrix(vector<vector<double>>& adjMatrix) {
    int n = adjMatrix.size();

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
