// Version 1.0

// Algorithms road map :
// v1 - Each snakes go to closest Energy cell using BFS
// v2 - Same as version 1 but 2 snakes can't refers to the same energy cell
//          Run BFS for each snakes and keep the closest energy
//          Pick the lowest Snake/Energy couple
//          Recompute BFS for snakes refering the Energy previously selected
//          Loop
// v3 - Evaluate snake moves independently using a fitness function : Score diff + sum (Snake/Energy distances)
// v4 - Evaluate all possible move combinaisons using a fitness function and physics simulation
// v5 - Beam search + Fitness function : Decreasing beam width, start at 3 ^ (3 + 3)
// v6 - Beam search + GA + Fitness function : Select 'beam_width' children with a small GA using fitness function already created

// No need to include physics in fitness function, tree iterations will take care of possible/impossible paths

#undef _GLIBCXX_DEBUG
#pragma GCC optimize("Ofast,unroll-loops,omit-frame-pointer,inline")
#pragma GCC option("arch=native", "tune=native", "no-zero-upper")
#pragma GCC target( \
    "movbe,aes,pclmul,avx,avx2,f16c,fma,sse3,ssse3,sse4.1,sse4.2,rdrnd,popcnt,bmi,bmi2,lzcnt")

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <climits>
#include <algorithm>
#include <sstream>
#include <map>
#include <set>
#include <unordered_set>

using namespace std;

using Pos = int; // 1 dimension coordinate in map (y * width + x)

/* --- MAPPROPERTIES --- */

constexpr int MAX_WIDTH = 45;
constexpr int MAX_HEIGHT = 30;
constexpr int MAX_CELL_COUNT = MAX_WIDTH * MAX_HEIGHT;

struct MapProperties
{
    int width;
    int height;
    int cell_count; // width * height
    int my_id;
    int opp_id;
    int max_snake_count;
    vector<Pos> adjacent_cells[MAX_CELL_COUNT]; // All valid adjacent cells + the current cell
                                                // (N, W, E, S)
};

static MapProperties map_properties;

/* --- POS --- */

constexpr int NORTH_POS_OFFSET = -MAX_WIDTH;
constexpr int WEST_POS_OFFSET = -1;
constexpr int EAST_POS_OFFSET = 1;
constexpr int SOUTH_POS_OFFSET = MAX_WIDTH;

int get_x(const Pos pos) { return pos % MAX_WIDTH; }
int get_y(const Pos pos) { return pos / MAX_WIDTH; }
Pos get_pos(const int x, const int y) { return y * MAX_WIDTH + x; }
Pos get_pos(const Pos pos) { return pos; }

/* --- SNAKE --- */

constexpr int MAX_SNAKE_SIZE = 50;

struct Snake
{
    int id;
    int player_id;
    Pos body_pos[MAX_SNAKE_SIZE];
    int body_length;
};

int get_snake_id(Snake *snake) { return snake->id; }
int get_snake_player_id(Snake *snake) { return snake->player_id; }
int get_snake_body_pos(Snake *snake, int index) { return snake->body_pos[index]; }
int get_snake_pos(Snake *snake) { return get_snake_body_pos(snake, 0); }

void add_body_pos(Snake *snake, Pos pos)
{
    // fprintf(stderr, "add_body_pos: idx %d - Pos %d\n", snake->body_length, pos);
    snake->body_pos[snake->body_length++] = pos;
    // fprintf(stderr, "add_body_pos: added Pos %d\n", snake->body_pos[snake->body_length - 1]);
}
void reset_snake_length(Snake *snake) { snake->body_length = 0; }

void initialize_snake_data(
    Snake *snake,
    int snake_id,
    int player_id)
{
    snake->id = snake_id;
    snake->player_id = player_id;
    bzero(snake->body_pos, sizeof(Pos) * MAX_SNAKE_SIZE);
    snake->body_length = 3;
}

void print_snake(Snake *snake)
{
    fprintf(stderr, "Snake %d (p=%d) of length %d:\n", snake->id, snake->player_id, snake->body_length);
    for (int i = 0; i < snake->body_length; i++)
    {
        fprintf(stderr, "  %d: (%d, %d)\n", i, get_x(snake->body_pos[i]), get_y(snake->body_pos[i]));
    }
}

/* --- ACTION --- */

constexpr int ACTION_UP = 1;
constexpr int ACTION_LEFT = 2;
constexpr int ACTION_RIGHT = 3;
constexpr int ACTION_DOWN = 4;

struct Action
{
    int snake_id;
    int action_type; // ACTION_UP | ACTION_LEFT | ACTION_RIGHT | ACTION_DOWN
};

string get_action_string(const Action &action)
{
    string dir_str = "UP";

    if (action.action_type == ACTION_LEFT)
    {
        dir_str = "LEFT";
    }
    else if (action.action_type == ACTION_RIGHT)
    {
        dir_str = "RIGHT";
    }
    else if (action.action_type == ACTION_DOWN)
    {
        dir_str = "DOWN";
    }

    return to_string(action.snake_id) + " " + dir_str;
}

/* --- STATE --- */

constexpr int MAX_SNAKE_COUNT = 8;
constexpr int MAX_PLAYER_SNAKE_COUNT = MAX_SNAKE_COUNT / 2;
constexpr int MIN_SNAKE_ID = 1;

constexpr int MAX_ACTION_COUNT = 3 ^ MAX_SNAKE_COUNT;

constexpr int CELL_EMPTY = 8;
constexpr int CELL_PLATFORM = 9;
constexpr int CELL_ENERGY = 10;

struct State
{
    int game_points;                   // Points difference between my id and opponent id
    int initial_cells[MAX_CELL_COUNT]; // 0-7: snake_id, 8: CELL_EMPTY, 9: CELL_PLATFORM
    int cells[MAX_CELL_COUNT];         // 0-7: snake_id, 8: CELL_EMPTY, 9: CELL_PLATFORM, 10: CELL_ENERGY

    Snake snakes[MIN_SNAKE_ID + MAX_SNAKE_COUNT]; // 1-16: snakes, 0: unused
    int alive_snake_count[2];
    int alive_snake_ids[2][MAX_PLAYER_SNAKE_COUNT];

    Pos energies[MAX_CELL_COUNT];
    int energy_count;

    Action actions[MAX_ACTION_COUNT]; // All available actions for one turn
    int action_count;                 // Number of actions available

    Action selected_actions[MAX_SNAKE_COUNT];
    int selected_actions_count;
    int snake_having_played[MIN_SNAKE_ID + MAX_SNAKE_COUNT]; // 0: not played, 1: played
};

int get_cell(State &state, Pos pos) { return state.cells[pos]; }
Snake *get_snake(State &state, int snake_id) { return &state.snakes[snake_id]; }
int get_player_alive_snake_count(State &state, int player_id) { return state.alive_snake_count[player_id]; }
int get_player_alive_snake_id(State &state, int player_id, int index) { return state.alive_snake_ids[player_id][index]; }

void set_initial_cell(State &state, int x, int y, int value) { state.initial_cells[get_pos(x, y)] = value; }
void set_cell(State &state, Pos pos, int value) { state.cells[pos] = value; }
void set_energy(State &state, int index, int x, int y) { state.energies[index] = get_pos(x, y); }
void set_alive_snake_count(State &state, int player_id, int count)
{
    state.alive_snake_count[player_id] = count;
}
void set_player_snake_ids(State &state, int player_id, int index, int snake_id)
{
    state.alive_snake_ids[player_id][index] = snake_id;
    // fprintf(stderr, "set_player_snake_ids : p %d - idx %d - snakeid %d\n", player_id, index, snake_id);
}

void initialize_snake_data(
    State &state,
    int snake_id,
    int player_id)
{
    Snake *snake = get_snake(state, snake_id);
    initialize_snake_data(
        snake, snake_id, player_id);
}

int find_closest_energy_cell(State &state, Pos start_pos, Pos &closest_energy_cell_pos);

void print_map(State &state)
{
    for (int y = 0; y < map_properties.height; y++)
    {
        for (int x = 0; x < map_properties.width; x++)
        {
            // Print emojis
            Pos pos = get_pos(x, y);
            int cell = get_cell(state, pos);
            if (cell == CELL_EMPTY)
                fprintf(stderr, "⬜");
            else if (cell == CELL_PLATFORM)
                fprintf(stderr, "🟦");
            else if (cell == CELL_ENERGY)
                fprintf(stderr, "🟨");
            else
            {
                Snake *snake = get_snake(state, cell);
                if (get_snake_player_id(snake) == map_properties.my_id)
                    fprintf(stderr, "🐍");
                else
                    fprintf(stderr, "🐉");
            }
        }
        fprintf(stderr, "\n");
    }
}

void print_map_ascii(State &state)
{
    Pos closest = -1;

    // int test = find_closest_energy_cell(state, 0, closest);
    // fprintf(stderr, "BFS result: %d | (%d, %d)\n", test, get_x(closest), get_y(closest));

    for (int y = 0; y < map_properties.height; y++)
    {
        for (int x = 0; x < map_properties.width; x++)
        {
            // Print emojis
            Pos pos = get_pos(x, y);
            int cell = get_cell(state, pos);
            if (cell == CELL_EMPTY)
            {
                int dist = find_closest_energy_cell(state, pos, closest);
                fprintf(stderr, "%d ", dist);
            }
            else if (cell == CELL_PLATFORM)
                fprintf(stderr, "# ");
            else if (cell == CELL_ENERGY)
                fprintf(stderr, "o ");
            else
            {
                fprintf(stderr, "S ");
                // Snake *snake = get_snake(state, cell);
                // if (snake->body_pos[0] == get_pos(x, y))
            }
        }
        fprintf(stderr, "\n");
    }
}

/* --- TOOL FUNCTIONS --- */

int get_opponent_id(const int player_id) { return 1 - player_id; }

bool is_out_of_bounds(const int x, const int y)
{
    return x < 0 || y < 0 || x >= map_properties.width || y >= map_properties.height;
}

bool is_north_cell_out_of_bounds(const Pos pos) { return get_y(pos) == 0; }
bool is_west_cell_out_of_bounds(const Pos pos) { return get_x(pos) == 0; }
bool is_east_cell_out_of_bounds(const Pos pos) { return get_x(pos) == map_properties.width - 1; }
bool is_south_cell_out_of_bounds(const Pos pos)
{
    return get_y(pos) == map_properties.height - 1;
}

Pos get_north_pos(const Pos pos) { return pos + NORTH_POS_OFFSET; }
Pos get_west_pos(const Pos pos) { return pos + WEST_POS_OFFSET; }
Pos get_east_pos(const Pos pos) { return pos + EAST_POS_OFFSET; }
Pos get_south_pos(const Pos pos) { return pos + SOUTH_POS_OFFSET; }

/* --- BFS --- */

int find_closest_energy_cell_recursive(State &state, queue<pair<Pos, int>> &bfs_queue, bool visited[], Pos &closest_energy_cell_pos)
{
    if (bfs_queue.empty())
    {
        return -1; // No energy cell found
    }

    pair<Pos, int> front = bfs_queue.front();
    Pos current_pos = front.first;
    int distance = front.second;
    bfs_queue.pop();

    // Explore neighbors in order: North, West, East, South
    Pos neighbors[4] = {
        get_north_pos(current_pos),
        get_west_pos(current_pos),
        get_east_pos(current_pos),
        get_south_pos(current_pos)};

    bool neighbor_out_of_bounds[4] = {
        is_north_cell_out_of_bounds(current_pos),
        is_west_cell_out_of_bounds(current_pos),
        is_east_cell_out_of_bounds(current_pos),
        is_south_cell_out_of_bounds(current_pos)};

    for (int i = 0; i < 4; i++)
    {

        // New valid cell if : In map & Not visited yet & Empty
        if (!neighbor_out_of_bounds[i] && !visited[neighbors[i]])
        {

            int neighbor = get_cell(state, neighbors[i]);

            // Check if the position is an energy cell
            if (neighbor == CELL_ENERGY)
            {
                closest_energy_cell_pos = neighbors[i];
                return distance + 1;
            }

            if (neighbor != CELL_EMPTY)
                continue;

            visited[neighbors[i]] = true;
            bfs_queue.push(make_pair(neighbors[i], distance + 1));
            // fprintf(stderr, "BFS: Push (%d, %d) with distance %d\n", get_x(neighbors[i]), get_y(neighbors[i]), distance + 1);
        }
    }

    // Recursively process the rest of the queue
    return find_closest_energy_cell_recursive(state, bfs_queue, visited, closest_energy_cell_pos);
}

int find_closest_energy_cell(State &state, Pos start_pos, Pos &closest_energy_cell_pos)
{
    // Check if the starting position is an energy cell
    if (get_cell(state, start_pos) == CELL_ENERGY)
    {
        closest_energy_cell_pos = start_pos;
        return 0;
    }

    bool visited[MAX_CELL_COUNT] = {false};
    queue<pair<Pos, int>> bfs_queue;
    visited[start_pos] = true;
    bfs_queue.push(make_pair(start_pos, 0));

    return find_closest_energy_cell_recursive(state, bfs_queue, visited, closest_energy_cell_pos);
}

/* --- DECISION MAKING --- */

int resolve_snake_dir(State &state, int snake_id)
{
    Snake *snake = get_snake(state, snake_id);
    Pos snake_pos = get_snake_pos(snake);

    // Explore neighbors in order: North, West, East, South
    Pos neighbors[4] = {
        get_east_pos(snake_pos),
        get_west_pos(snake_pos),
        get_north_pos(snake_pos),
        get_south_pos(snake_pos)};

    bool neighbor_out_of_bounds[4] = {
        is_east_cell_out_of_bounds(snake_pos),
        is_west_cell_out_of_bounds(snake_pos),
        is_north_cell_out_of_bounds(snake_pos),
        is_south_cell_out_of_bounds(snake_pos)};

    Pos closest = -1;
    Pos best_closest = -1;
    int best_dist = 100000;
    int best_i = -1;
    for (int i = 0; i < 4; i++)
    {

        // New valid cell if : In map & Not visited yet & Empty
        if (!neighbor_out_of_bounds[i] && (get_cell(state, neighbors[i]) == CELL_EMPTY || get_cell(state, neighbors[i]) == CELL_ENERGY))
        {
            int dist = find_closest_energy_cell(state, neighbors[i], closest);
            // fprintf(stderr, "Snake %d try dir %d with dist %d ... (Actual best dist is %d on dir %d\n", snake_id, i, dist, best_dist, best_i);

            if (dist != -1 && dist < best_dist)
            {
                best_dist = dist;
                best_closest = closest;
                best_i = i;
                // fprintf(stderr, "Snake %d find new best dir %d with dist %d !\n", snake_id, i, dist);
            }
        }
    }

    if (best_i == -1)
    {
        fprintf(stderr, "Snake %d is stuck !\n", snake_id);
        return 0;
    }

    fprintf(stderr, "Snake %d must go %d !\n", snake_id, best_i);
    cout << "MARK " << get_x(best_closest) << " " << get_y(best_closest) << ";";
    return best_i;
}

/* --- PARSING --- */

void parse_initial_inputs(State &state)
{
    cin >> map_properties.my_id;
    map_properties.opp_id = get_opponent_id(map_properties.my_id);

    cin >> map_properties.width;
    cin >> map_properties.height;
    map_properties.cell_count = map_properties.width * map_properties.height;

    // fprintf(stderr, "Width/Height : %d/%d\n", map_properties.width, map_properties.height);

    memset(&state, 0, sizeof(state));
    cin.ignore();
    for (int y = 0; y < map_properties.height; y++)
    {
        string row;
        getline(cin, row);

        for (int x = 0; x < map_properties.width; x++)
        {
            if (row[x] == '.')
                set_initial_cell(state, x, y, CELL_EMPTY);
            else if (row[x] == '#')
                set_initial_cell(state, x, y, CELL_PLATFORM);
        }
    }

    int snakebots_per_player;
    cin >> snakebots_per_player;
    // fprintf(stderr, "snakebots_per_player : %d\n", snakebots_per_player);

    set_alive_snake_count(state, map_properties.my_id, snakebots_per_player);
    set_alive_snake_count(state, map_properties.opp_id, snakebots_per_player);

    int snakebot_id;
    for (int i = 0; i < snakebots_per_player; i++)
    {
        cin >> snakebot_id;
        initialize_snake_data(state, snakebot_id, map_properties.my_id);
        set_player_snake_ids(state, map_properties.my_id, i, snakebot_id);
    }
    for (int i = 0; i < snakebots_per_player; i++)
    {
        cin >> snakebot_id;
        initialize_snake_data(state, snakebot_id, map_properties.opp_id);
        set_player_snake_ids(state, map_properties.opp_id, i, snakebot_id);
    }
}

Pos parse_pos_from_segment(string segment)
{
    size_t commaPos = segment.find(',');
    int x = stoi(segment.substr(0, commaPos));
    int y = stoi(segment.substr(commaPos + 1));

    // fprintf(stderr, "parse_pos_from_segment: %d %d\n", x, y);
    return get_pos(x, y);
}

void parse_snakebot(State &state, int *my_snake_count, int *opp_snake_count, int snakebotId, string bodyStr)
{

    Snake *snake = get_snake(state, snakebotId);
    int player_id = get_snake_player_id(snake);

    int *snake_count;
    if (player_id == map_properties.my_id)
        snake_count = my_snake_count;
    else
        snake_count = opp_snake_count;

    set_player_snake_ids(state, player_id, *snake_count, snakebotId);
    (*snake_count)++;

    reset_snake_length(snake);

    // Parse the body string into coordinates
    size_t start = 0;
    size_t end = bodyStr.find(':');

    while (end != string::npos)
    {
        string segment = bodyStr.substr(start, end - start);

        Pos body_pos = parse_pos_from_segment(segment);
        add_body_pos(snake, body_pos);
        set_cell(state, body_pos, snakebotId);

        start = end + 1;
        end = bodyStr.find(':', start);
    }

    // Add the last segment
    string lastSegment = bodyStr.substr(start);
    Pos body_pos = parse_pos_from_segment(lastSegment);
    add_body_pos(snake, body_pos);
    set_cell(state, body_pos, snakebotId);
}

void parse_turn_inputs(State &state)
{
    memcpy(state.cells, state.initial_cells, MAX_CELL_COUNT * sizeof(int));

    cin >> state.energy_count;
    for (int i = 0; i < state.energy_count; i++)
    {
        int x, y;
        cin >> x >> y;
        set_energy(state, i, x, y);
        set_cell(state, state.energies[i], CELL_ENERGY);
    }

    state.alive_snake_count[0] = 0;
    state.alive_snake_count[1] = 0;

    int my_snake_count = 0;
    int opp_snake_count = 0;

    int snakebotCount;
    cin >> snakebotCount;
    for (int i = 0; i < snakebotCount; i++)
    {
        int snakebotId;
        string bodyStr;
        cin >> snakebotId >> bodyStr;

        parse_snakebot(state, &my_snake_count, &opp_snake_count, snakebotId, bodyStr);
    }

    set_alive_snake_count(state, map_properties.my_id, my_snake_count);
    set_alive_snake_count(state, map_properties.opp_id, opp_snake_count);
}

int main()
{
    State state;
    parse_initial_inputs(state);

    while (true)
    {
        parse_turn_inputs(state);
        print_map(state);
        // print_map_ascii(state);

        for (int i = 0; i < get_player_alive_snake_count(state, map_properties.my_id); i++)
        {
            int snake_id = get_player_alive_snake_id(state, map_properties.my_id, i);
            int dir = resolve_snake_dir(state, snake_id);

            if (dir == 0)
                cout << snake_id << " RIGHT";
            else if (dir == 1)
                cout << snake_id << " LEFT";
            else if (dir == 2)
                cout << snake_id << " UP";
            else
                cout << snake_id << " DOWN";

            if (i != get_player_alive_snake_count(state, map_properties.my_id) - 1)
                cout << ";";
        }
        cout << endl;
    }

    return 0;
}
