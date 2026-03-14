// @bot random
// @description Random bot for Winter Challenge 2026 (Snakebird)
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace std;

int main() {
    srand(time(nullptr));

    int myId, width, height;
    cin >> myId >> width >> height; cin.ignore();

    for (int i = 0; i < height; i++) {
        string row;
        getline(cin, row);
    }

    int snakebotsPerPlayer;
    cin >> snakebotsPerPlayer; cin.ignore();

    vector<int> myBots(snakebotsPerPlayer), oppBots(snakebotsPerPlayer);
    for (int i = 0; i < snakebotsPerPlayer; i++) cin >> myBots[i];
    cin.ignore();
    for (int i = 0; i < snakebotsPerPlayer; i++) cin >> oppBots[i];
    cin.ignore();

    const char* dirs[] = {"UP", "DOWN", "LEFT", "RIGHT"};

    while (true) {
        int powerSourceCount;
        cin >> powerSourceCount; cin.ignore();
        for (int i = 0; i < powerSourceCount; i++) {
            int x, y;
            cin >> x >> y; cin.ignore();
        }

        int snakebotCount;
        cin >> snakebotCount; cin.ignore();

        vector<int> ids;
        for (int i = 0; i < snakebotCount; i++) {
            int id;
            string body;
            cin >> id >> body; cin.ignore();
            ids.push_back(id);
        }

        // Move each of my snakebots in a random direction
        string output;
        for (int i = 0; i < snakebotsPerPlayer; i++) {
            if (i > 0) output += ";";
            output += to_string(myBots[i]) + " " + dirs[rand() % 4];
        }
        cout << output << endl;
    }
}