// Version 1.1

// Algorithms road map :
// v1 - Each snakes go to closest Energy cell using BFS
//  v1.1 - Apply Action + Gravity first, and then use BFS value
// v2 - Evaluate all possible move combinaisons using physics simulation (gravity + collisions) first and then fitness function (Score diff + sum (Snake/Energy distances))
// v3 - Beam search + Fitness function : Decreasing beam width, start at 3 ^ (3 + 3)
// v4 - Beam search + GA + Fitness function : Select 'beam_width' children with a small GA using fitness function already created

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
Pos get_snake_body_pos(Snake *snake, int index) { return snake->body_pos[index]; }
Pos get_snake_head_pos(Snake *snake) { return get_snake_body_pos(snake, 0); }
Pos get_snake_body_length(Snake *snake) { return snake->body_length; }

Pos set_snake_body_pos(Snake *snake, int index, Pos pos) { return snake->body_pos[index] = pos; }

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

void move_snake_to(Snake *snake, Pos new_head_pos)
{
    memcpy(&snake->body_pos[1], &snake->body_pos[0], sizeof(Pos) * (snake->body_length - 1));
    set_snake_body_pos(snake, 0, new_head_pos);
}

/* --- ACTION --- */

enum ActionType
{
    ACTION_UP,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_DOWN
};

struct Action
{
    int snake_id;
    ActionType action_type; // ACTION_UP | ACTION_LEFT | ACTION_RIGHT | ACTION_DOWN
};

int get_action_snake_id(Action &action) { return action.snake_id; }
ActionType get_action_type(Action &action) { return action.action_type; }

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

void print_map(State &state, string title = "")
{
    if (title != "")
        fprintf(stderr, "%s\n", title.c_str());

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

int find_closest_energy_cell(State &state, Pos start_pos, Pos &closest_energy_cell_pos);
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

/* --- GAME PHYSICS --- */

bool is_cell_walkable(int cell)
{
    return cell == CELL_EMPTY || cell == CELL_ENERGY;
}

bool is_cell_solid(int cell, int snake_id)
{
    return cell != snake_id && cell != CELL_EMPTY;
}

int generate_snake_actions(State &state, Snake *snake, Pos actions[4])
{
    Pos head_pos = get_snake_head_pos(snake);

    // Explore neighbors in order: North, West, East, South
    Pos neighbors[4] = {
        get_north_pos(head_pos),
        get_west_pos(head_pos),
        get_east_pos(head_pos),
        get_south_pos(head_pos)};

    bool neighbor_out_of_bounds[4] = {
        is_north_cell_out_of_bounds(head_pos),
        is_west_cell_out_of_bounds(head_pos),
        is_east_cell_out_of_bounds(head_pos),
        is_south_cell_out_of_bounds(head_pos)};

    int action_count = 0;
    for (int i = 0; i < 4; i++)
    {
        int neighbor = get_cell(state, neighbors[i]);

        // New valid cell if : In map & Not visited yet & Empty
        if (!neighbor_out_of_bounds[i] && is_cell_walkable(neighbor))
        {
            actions[action_count++] = neighbors[i];
        }
    }

    return action_count;
}

void move_snake_in_state(State &state, Snake *snake, Pos new_head_pos)
{
    fprintf(stderr, "Move snake %d to %d %d\n", get_snake_id(snake), get_x(new_head_pos), get_y(new_head_pos));

    // Clear the tail position and set the snake id in the new head
    Pos tail_pos = get_snake_body_pos(snake, get_snake_body_length(snake) - 1);
    set_cell(state, tail_pos, CELL_EMPTY);
    set_cell(state, new_head_pos, get_snake_id(snake));

    // Move the whole snake toward the new position
    move_snake_to(snake, new_head_pos);
}

void apply_gravity(State &state, Snake *snake)
{
    int snake_id = get_snake_id(snake);
    int snake_body_length = get_snake_body_length(snake);
    Pos new_snake_positions[snake_body_length];

    while (true)
    {
        // Verify if snake has one solid cell under it
        for (int i = 0; i < snake_body_length; i++)
        {
            Pos pos = get_snake_body_pos(snake, i);
            Pos pos_below = get_south_pos(pos);
            int cell_below = get_cell(state, pos_below);

            // fprintf(stderr, "Snake %d: Is cell %d solid? (Pos=%d)\n", snake_id, cell_below, pos);
            if (is_cell_solid(cell_below, snake_id))
                return;

            new_snake_positions[i] = pos_below;
        }

        fprintf(stderr, "Snake %d is falling\n", snake_id);

        // If not, remove snake from state
        for (int i = 0; i < snake_body_length; i++)
            set_cell(state, get_snake_body_pos(snake, i), CELL_EMPTY);

        // And set it one cell below
        for (int i = 0; i < snake_body_length; i++)
        {
            set_cell(state, new_snake_positions[i], snake_id);
            set_snake_body_pos(snake, i, new_snake_positions[i]);
        }
    }
}

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

Pos choose_snake_dir(State &state, Snake *snake)
{
    Pos snake_pos = get_snake_head_pos(snake);

    Pos actions[4];
    int action_count = generate_snake_actions(state, snake, actions);
    fprintf(stderr, "choose_snake_dir(): Snake %d - %d actions generated\n", get_snake_id(snake), action_count);

    State next_state;
    Snake next_snake;

    Pos closest = -1;
    Pos best_closest = -1;
    Pos best_action = -1;
    int best_dist = 100000;
    for (int i = 0; i < action_count; i++)
    {
        // Reset states
        memcpy(&next_state, &state, sizeof(State));
        memcpy(&next_snake, snake, sizeof(Snake));

        move_snake_in_state(next_state, &next_snake, actions[i]);
        // print_map(next_state, "Map after applying action");

        apply_gravity(next_state, &next_snake);
        // print_map(next_state, "Map after gravity");

        int dist = find_closest_energy_cell(next_state, get_snake_head_pos(&next_snake), closest);
        if (dist != -1 && dist < best_dist)
        {
            best_dist = dist;
            best_closest = closest;
            best_action = actions[i];
            fprintf(stderr, "Snake %d find new best closest %d %d with dist %d !\n", get_snake_id(&next_snake), get_x(best_action), get_y(best_action), dist);
        }
    }

    if (best_action == -1)
    {
        fprintf(stderr, "Snake %d is stuck !\n", get_snake_id(snake));
        return 0;
    }

    // fprintf(stderr, "Snake %d must go %d !\n", get_snake_id(snake), best_closest);
    cout << "MARK " << get_x(best_closest) << " " << get_y(best_closest) << ";";
    return best_action;
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
        print_map(state, "Turn beginning");
        // print_map_ascii(state);

        for (int i = 0; i < get_player_alive_snake_count(state, map_properties.my_id); i++)
        {
            int snake_id = get_player_alive_snake_id(state, map_properties.my_id, i);
            Snake *snake = get_snake(state, snake_id);

            Pos dir = choose_snake_dir(state, snake);
            Pos snake_head = get_snake_head_pos(snake);
            int dir_offset = dir - snake_head;

            fprintf(stderr, "Snake %d: head %d %d, dir %d %d, dir_offset %d\n", snake_id, get_x(snake_head), get_y(snake_head), get_x(dir), get_y(dir), dir_offset);

            if (dir_offset == NORTH_POS_OFFSET)
                cout << snake_id << " UP";
            else if (dir_offset == SOUTH_POS_OFFSET)
                cout << snake_id << " DOWN";
            else if (dir_offset == WEST_POS_OFFSET)
                cout << snake_id << " LEFT";
            else
                cout << snake_id << " RIGHT";

            if (i != get_player_alive_snake_count(state, map_properties.my_id) - 1)
                cout << ";";
        }
        cout << endl;
    }

    return 0;
}
