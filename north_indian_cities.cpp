#include <bits/stdc++.h>
#include <vector>
#include <string>

class Node {
public:
    int id;
    std::string name;
    int x, y;

    Node(int id, const std::string& name, int x, int y)
        : id(id), name(name), x(x), y(y) {}
};

double euclidean(const Point& a, const Point& b) {
    return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}

vector<int> kMeansHubSelection(vector<Node>& nodes, int k, int maxIterations = 100) {
    int n = nodes.size();
    vector<Node> centroids;
    vector<int> labels(n, -1);

    // Random initialization of centroids
    srand(time(0));
    for (int i = 0; i < k; ++i) {
        centroids.push_back(nodes[rand() % n]);
    }

    for (int iter = 0; iter < maxIterations; ++iter) {
        bool changed = false;

        // Assign each point to the nearest centroid
        for (int i = 0; i < n; ++i) {
            double minDist = DBL_MAX;
            int bestCluster = 0;
            for (int j = 0; j < k; ++j) {
                double dist = euclideanDistance(nodes[i], centroids[j]);
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = j;
                }
            }

            if (labels[i] != bestCluster) {
                changed = true;
                labels[i] = bestCluster;
            }
        }

        if (!changed) break; // Converged

        // Recalculate centroids
        vector<double> sumX(k, 0), sumY(k, 0);
        vector<int> count(k, 0);
        for (int i = 0; i < n; ++i) {
            sumX[labels[i]] += nodes[i].x;
            sumY[labels[i]] += nodes[i].y;
            count[labels[i]]++;
        }

        for (int j = 0; j < k; ++j) {
            if (count[j] == 0) continue;
            centroids[j].x = sumX[j] / count[j];
            centroids[j].y = sumY[j] / count[j];
        }
    }

    // Choose actual nodes closest to each centroid as hubs
    vector<int> hubs(k, -1);
    for (int j = 0; j < k; ++j) {
        double minDist = DBL_MAX;
        for (int i = 0; i < n; ++i) {
            if (labels[i] == j) {
                double dist = euclideanDistance(nodes[i], centroids[j]);
                if (dist < minDist) {
                    minDist = dist;
                    hubs[j] = nodes[i].id;
                }
            }
        }
    }

    return hubs;
}





int main() {
    Node delhi(1, "Delhi", 700, 220);
    Node amritsar(2, "Amritsar", 640, 130);
    Node chandigarh(3, "Chandigarh", 670, 150);
    Node jaipur(4, "Jaipur", 600, 350);
    Node lucknow(5, "Lucknow", 690, 320);
    Node kanpur(6, "Kanpur", 675, 340);
    Node agra(7, "Agra", 670, 275);
    Node varanasi(8, "Varanasi", 750, 375);
    Node meerut(9, "Meerut", 720, 250);
    Node aligarh(10, "Aligarh", 690, 260);
    Node patna(11, "Patna", 770, 410);
    Node ghaziabad(12, "Ghaziabad", 715, 230);
    Node moradabad(13, "Moradabad", 730, 265);
    Node bareilly(14, "Bareilly", 705, 280);
    Node saharanpur(15, "Saharanpur", 690, 205);
    Node haridwar(16, "Haridwar", 670, 190);
    Node roorkee(17, "Roorkee", 665, 180);
    Node rishikesh(18, "Rishikesh", 660, 170);
    Node nainital(19, "Nainital", 655, 150);
    Node mathura(20, "Mathura", 680, 265);
    Node hoshiarpur(21, "Hoshiarpur", 645, 140);
    Node kullu(22, "Kullu", 655, 120);
    Node shimla(23, "Shimla", 660, 110);
    Node kangra(24, "Kangra", 650, 125);
    Node solan(25, "Solan", 660, 130);
    Node srinagar(26, "Srinagar", 610, 90);
    Node jammu(27, "Jammu", 625, 100);
    Node ludhiana(28, "Ludhiana", 660, 180);
    Node patiala(29, "Patiala", 650, 170);
    Node panipat(30, "Panipat", 705, 240);
    Node sonipat(31, "Sonipat", 710, 230);
    Node muzaffarnagar(32, "Muzaffarnagar", 720, 260);
    Node fatehpur(33, "Fatehpur", 690, 310);
    Node karnal(34, "Karnal", 705, 230);
    Node bhiwani(35, "Bhiwani", 680, 245);
    Node hisar(36, "Hisar", 670, 275);
    Node jind(37, "Jind", 660, 280);
    Node kurukshetra(38, "Kurukshetra", 715, 220);
    Node rohtak(39, "Rohtak", 720, 230);
    Node faridabad(40, "Faridabad", 700, 215);
    Node barnala(41, "Barnala", 640, 150);
    Node muktsar(42, "Muktsar", 635, 135);
    Node sangrur(43, "Sangrur", 630, 140);
    Node bhatinda(44, "Bhatinda", 625, 160);
    Node jalandhar(45, "Jalandhar", 650, 160);
    Node ambala(46, "Ambala", 675, 160);
    Node gurugram(47, "Gurugram", 710, 210);
    Node noida(48, "Noida", 715, 220);
    Node farukhabad(49, "Farukhabad", 755, 360);

    std::vector<Node> cities = {delhi, amritsar, chandigarh, jaipur, lucknow, kanpur, agra, varanasi, meerut, aligarh, patna, ghaziabad, moradabad, bareilly, saharanpur, haridwar, roorkee, rishikesh, nainital, mathura, hoshiarpur, kullu, shimla, kangra, solan, srinagar, jammu, ludhiana, patiala, panipat, sonipat, muzaffarnagar, fatehpur, karnal, bhiwani, hisar, jind, kurukshetra, rohtak, faridabad, barnala, muktsar, sangrur, bhatinda, jalandhar, ambala, gurugram, noida, farukhabad};

    // Example: print city names
    for (const Node& city : cities) {
        std::cout << "City: " << city.name << " (ID: " << city.id
                  << ", X: " << city.x << ", Y: " << city.y << ")\n";
    }


    int k = 10; // Number of hubs you want
    vector<int> hubIds = kMeansHubSelection(cities, k);

    cout << "Selected Hubs:\n";
    for (int id : hubIds) {
        cout << nodes[id].name << " (ID: " << id << ")\n";
    }

    return 0;
}
