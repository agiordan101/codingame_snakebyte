// Version 6

// Algorithms :
//  v1 - Each snakes go to closest Energy cell using BFS
//      v1.1 - Apply Action + Gravity first, and then use BFS value
//      v1.2 - Always select a valid action, even if no energy cell is found
//      v1.3 - Add padding around map so all simulations work outside the map (BFS, etc...)
//  v2 - Evaluate all possible move combinaisons at current depth, using physics simulation (gravity + collisions) and an improved fitness function (Score diff + sum (Snake/Energy distances))
//  v3 - Add an opponent move choice before with same algorithm
//      v3.1 - Apply opponent moveset in each simulated moveset
//           - Fix collisions simulation
//           - Fix cells application
//           - Correctly remove snake id from player alive list
//           - Snake weren't checking collisions with their own body
//  v4 - Beam search : Strategy explained in README.md
//      v4.1 - For depth > 1, stop simulating the opponent by choosing among all move combinaisons, but choose best snake move one by one (using previous snake moves)
//              So they still work together but the first ones have leading.
//              For 4 snakes, we're not simulating 3^4=81 movesets but 3*4=12 movesets.
//      v4.2 - Heuristic : Replace BFS by Manhattan distance
//      v4.3 - Heuristic : During first turn: Create a lookup table to know for all cells which energies are the closest and their BFS distance (without snakes)
//      v4.4 - v4.3 but debugged
//           - Fix snake parsing out of padded cells bounds
//           - Initialize with default distance because some energy cells may be innaccessible
//           - Reduce PADDING from 10 to 2 and reduce size of Pos energies from MAX_CELL_COUNT MAX_ENERGY_COUNT: From ~6500 to ~1750 integers (~ /4)
//      v4.5 - Remove time check each at state child/MoveSet: Now check once per beam state
//           - Snake collision refacto: Remove memcpy in apply_moveset by storing colliding_snakes in an array and removing their head after
//           - Platform collision refacto: Handle them at the same time than snake collisions
//           - Energy eating refacto: Remove energies later so multiple snakes can eat the same energy in the same turn
//      v4.6 - Simplify apply_moveset: Optimize snake mouvements in cells (Don't remove and reapply each turns)
//           - Add early return when collision are detected
//  v5 - Beam search use a min-heap storage for candidates instead of classic vector
//     - Reduce consider_state_to_be_candidate() from 67% to 15% of the time spent per turn
//     - Increase visited state per turn about 50%
//      v5.1 - Restore BFS search in heuristic, with new iterative implementation (Now the engine is faster, it could be worth it)
//      v5.2 - Add maluses when body pieces are out of map
//           - Align end game heuristics
//           - Take care of win/lose with ex-aequo game points, but different amount of losses
//           - refacto: Count game points in realtime instead of storing them
//      v5.3 - Add bonuses depending on the first body index on a platform. The closer to the head, the more bonus
//           - Weight win/lose heuristic by ally player points and turns.
//  v5 - Reword beam search state data structures

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
#include <chrono>

#include <csignal>
#include <cstdlib>
#include <execinfo.h> // For backtrace
#include <unistd.h>   // For STDERR_FILENO

using namespace std;

using Pos = int; // 1 dimension coordinate in map (y * width + x)

/* --- MAPPROPERTIES --- */

constexpr int MAX_MAP_WIDTH = 45;
constexpr int MAX_MAP_HEIGHT = 30;

constexpr int MAP_PADDING = 2;
constexpr int MAX_WIDTH = MAX_MAP_WIDTH + 2 * MAP_PADDING;
constexpr int MAX_HEIGHT = MAX_MAP_HEIGHT + 2 * MAP_PADDING;
constexpr int MAX_CELL_COUNT = MAX_WIDTH * MAX_HEIGHT;

constexpr int MAX_ENERGY_COUNT = 100;

struct MapProperties
{
    int width;
    int height;
    int my_id;
    int opp_id;
};

static MapProperties map_properties;

/* --- POS --- */

constexpr int NORTH_POS_OFFSET = -MAX_WIDTH;
constexpr int WEST_POS_OFFSET = -1;
constexpr int EAST_POS_OFFSET = 1;
constexpr int SOUTH_POS_OFFSET = MAX_WIDTH;

Pos get_pos(int x, int y) { return y * MAX_WIDTH + x; }
Pos get_pos_from_map_coord(int x, int y) { return (y + MAP_PADDING) * MAX_WIDTH + (x + MAP_PADDING); }

int get_x(Pos pos) { return pos % MAX_WIDTH; }
int get_y(Pos pos) { return pos / MAX_WIDTH; }
int get_map_x(Pos pos) { return get_x(pos) - MAP_PADDING; }
int get_map_y(Pos pos) { return get_y(pos) - MAP_PADDING; }

/* --- SNAKE --- */

constexpr int MAX_SNAKE_SIZE = 100;
constexpr int MAX_SNAKE_COUNT = 8;
constexpr int MAX_PLAYER_SNAKE_COUNT = MAX_SNAKE_COUNT / 2;

struct Snake
{
    int id;
    int player_id;
    Pos body_pos[MAX_SNAKE_SIZE];
    int body_length;
};

int get_snake_id(Snake &snake) { return snake.id; }
int get_snake_player_id(Snake &snake) { return snake.player_id; }
Pos get_snake_body_pos(Snake &snake, int index) { return snake.body_pos[index]; }
Pos get_snake_head_pos(Snake &snake) { return get_snake_body_pos(snake, 0); }
int get_snake_body_length(Snake &snake) { return snake.body_length; }

void set_snake_body_pos(Snake &snake, int index, Pos pos) { snake.body_pos[index] = pos; }
void set_snake_body_length(Snake &snake, int length) { snake.body_length = length; }

void add_body_pos(Snake &snake, Pos pos)
{
    snake.body_pos[snake.body_length++] = pos;
}
void remove_snake_head(Snake &snake)
{
    // Shift all body positions to the left, removing the head pos, and decreasing the body length by 1
    memmove(&snake.body_pos[0], &snake.body_pos[1], sizeof(Pos) * (--snake.body_length));
}
void reset_snake_length(Snake &snake) { snake.body_length = 0; }

void initialize_snake_data(
    Snake &snake,
    int snake_id,
    int player_id)
{
    snake.id = snake_id;
    snake.player_id = player_id;
    bzero(snake.body_pos, sizeof(Pos) * MAX_SNAKE_SIZE);
    snake.body_length = 3;
}

void print_snake(Snake &snake)
{
    fprintf(stderr, "Snake %d (p=%d) of length %d:\n", snake.id, snake.player_id, snake.body_length);
    for (int i = 0; i < snake.body_length; i++)
    {
        Pos body_pos = get_snake_body_pos(snake, i);
        fprintf(stderr, "  %d: (%d, %d)\n", i, get_map_x(body_pos), get_map_y(body_pos));
    }
}

/* --- ACTION / MOVE --- */

struct Move
{
    int snake_id;
    Pos dst_pos;
};

int get_move_snake_id(Move &move) { return move.snake_id; }
Pos get_move_dst_pos(Move &move) { return move.dst_pos; }

void set_move_snake_id(Move &move, int id) { move.snake_id = id; }
void set_move_dst_pos(Move &move, Pos pos) { move.dst_pos = pos; }

constexpr int constexpr_pow(int base, int exp) { return exp == 0 ? 1 : base * constexpr_pow(base, exp - 1); }
constexpr int MAX_PLAYER_MOVE_SETS = constexpr_pow(3, MAX_PLAYER_SNAKE_COUNT); // Max move combinaisons for one player

struct MoveSet
{
    Move moves[MAX_SNAKE_COUNT];
    int move_count;
};

Move &get_moveset_move(MoveSet &moveset, int i) { return moveset.moves[i]; }
int get_moveset_move_count(MoveSet &moveset) { return moveset.move_count; }

void set_moveset_move(MoveSet &moveset, Move &move, int i) { moveset.moves[i] = move; }
void set_moveset_move_count(MoveSet &moveset, int count) { moveset.move_count = count; }

void print_moveset(MoveSet moveset)
{
    fprintf(stderr, "MoveSet (move_count=%d) : ", moveset.move_count);
    for (int i = 0; i < moveset.move_count; i++)
        fprintf(stderr, "S=%d: %d %d | ", moveset.moves[i].snake_id, get_map_x(moveset.moves[i].dst_pos), get_map_y(moveset.moves[i].dst_pos));
    fprintf(stderr, "\n");
}

/* --- STATE --- */

constexpr int CELL_EMPTY = 8;
constexpr int CELL_PLATFORM = 9;
constexpr int CELL_ENERGY = 10;

struct State
{
    int turn;
    bool game_ended;
    int player_losses[2];

    int cells[MAX_CELL_COUNT]; // 0-7: snake_id, 8: CELL_EMPTY, 9: CELL_PLATFORM, 10: CELL_ENERGY

    Snake snakes[MAX_SNAKE_COUNT]; // 0-7: snakes

    int player_alive_snake_count[2];
    int player_alive_snake_ids[2][MAX_PLAYER_SNAKE_COUNT];

    int alive_snake_count;
    int alive_snake_ids[MAX_SNAKE_COUNT];

    Pos energies[MAX_ENERGY_COUNT];
    int energy_count;

    float heuristic;
    MoveSet first_depth_moveset;
};

int get_turn(State &state) { return state.turn; }
bool is_game_ended(State &state) { return state.game_ended; }
int get_player_losses(State &state, int player_id) { return state.player_losses[player_id]; }
int get_energy_count(State &state) { return state.energy_count; }
int get_cell(State &state, Pos pos) { return state.cells[pos]; }
Snake &get_snake(State &state, int snake_id) { return state.snakes[snake_id]; }
int get_player_alive_snake_count(State &state, int player_id) { return state.player_alive_snake_count[player_id]; }
int get_player_alive_snake_id(State &state, int player_id, int index) { return state.player_alive_snake_ids[player_id][index]; }
int get_alive_snake_count(State &state) { return state.alive_snake_count; }
int get_alive_snake_id(State &state, int index) { return state.alive_snake_ids[index]; }
Pos get_energy(State &state, int index) { return state.energies[index]; }
float get_heuristic(State &state) { return state.heuristic; }
MoveSet &get_first_depth_moveset(State &state) { return state.first_depth_moveset; }

void set_turn(State &state, int turn) { state.turn = turn; }
void set_game_ended(State &state) { state.game_ended = true; }
void set_player_losses(State &state, int player_id, int loss_count) { state.player_losses[player_id] = loss_count; }
void add_player_loss(State &state, int player_id, int loss_count)
{
    state.player_losses[player_id] += loss_count;
}
void set_energy_count(State &state, int count) { state.energy_count = count; }
void set_cell(State &state, Pos pos, int value) { state.cells[pos] = value; }
void set_energy(State &state, int index, Pos pos) { state.energies[index] = pos; }
void set_heuristic(State &state, float heuristic) { state.heuristic = heuristic; }
void set_first_depth_moveset(State &state, MoveSet &moveset) { state.first_depth_moveset = moveset; }

void initialize_cells(State &state)
{
    for (int i = 0; i < MAX_CELL_COUNT; i++)
    {
        state.cells[i] = CELL_EMPTY;
    }
}
void initialize_snake_data(
    State &state,
    int snake_id,
    int player_id)
{
    Snake &snake = get_snake(state, snake_id);
    initialize_snake_data(
        snake, snake_id, player_id);
}

void reset_alive_snake_count(State &state)
{
    state.alive_snake_count = 0;
    state.player_alive_snake_count[0] = 0;
    state.player_alive_snake_count[1] = 0;
}
void add_player_alive_snake_id(State &state, int player_id, int snake_id)
{
    state.alive_snake_ids[state.alive_snake_count++] = snake_id;
    state.player_alive_snake_ids[player_id][state.player_alive_snake_count[player_id]++] = snake_id;
}
void remove_snake_from_alive_snake_ids(State &state, int snake_id, int player_id)
{
    for (int i = 0; i < get_alive_snake_count(state); i++)
    {
        if (get_alive_snake_id(state, i) == snake_id)
        {
            if (i + 1 < get_alive_snake_count(state))
            {
                // Remove the snake id from alive_snake_ids by shifting the rest of the array
                memmove(&state.alive_snake_ids[i], &state.alive_snake_ids[i + 1], sizeof(int) * (get_alive_snake_count(state) - i - 1));
            }
            state.alive_snake_count--;
            break;
        }
    }

    // Remove the snake from the player's alive snake ids
    for (int i = 0; i < get_player_alive_snake_count(state, player_id); i++)
    {
        if (get_player_alive_snake_id(state, player_id, i) == snake_id)
        {
            if (i + 1 < get_player_alive_snake_count(state, player_id))
            {
                // Remove the snake id from alive_snake_ids by shifting the rest of the array
                memmove(&state.player_alive_snake_ids[player_id][i], &state.player_alive_snake_ids[player_id][i + 1], sizeof(int) * (get_player_alive_snake_count(state, player_id) - i - 1));
            }
            state.player_alive_snake_count[player_id]--;
            return;
        }
    }
}
void remove_energy(State &state, Pos energy)
{
    for (int i = 0; i < state.energy_count; i++)
    {
        if (get_energy(state, i) == energy)
        {
            if (i + 1 < state.energy_count)
                memmove(&state.energies[i], &state.energies[i + 1], sizeof(Pos) * (state.energy_count - i - 1));
            state.energy_count--;
            return;
        }
    }
}

int revert_last_move_time = 0;
int revert_last_move_count = 0;
void revert_last_move(State &last_state, State &state)
{
    auto start_chrono = chrono::high_resolution_clock::now();

    memcpy(&state, &last_state, sizeof(State));

    auto end_chrono = chrono::high_resolution_clock::now();
    revert_last_move_time += chrono::duration_cast<chrono::microseconds>(end_chrono - start_chrono).count();
    revert_last_move_count++;
}

int count_player_points(State &state, int player_id);

/* --- TOOL FUNCTIONS --- */

int get_opponent_id(const int player_id) { return 1 - player_id; }

bool is_north_cell_out_of_bounds(const Pos pos) { return get_y(pos) == 0; }
bool is_west_cell_out_of_bounds(const Pos pos) { return get_x(pos) == 0; }
bool is_east_cell_out_of_bounds(const Pos pos) { return get_x(pos) == MAX_WIDTH - 1; }
bool is_south_cell_out_of_bounds(const Pos pos)
{
    return get_y(pos) == MAX_HEIGHT - 1;
}

Pos get_north_pos(const Pos pos) { return pos + NORTH_POS_OFFSET; }
Pos get_west_pos(const Pos pos) { return pos + WEST_POS_OFFSET; }
Pos get_east_pos(const Pos pos) { return pos + EAST_POS_OFFSET; }
Pos get_south_pos(const Pos pos) { return pos + SOUTH_POS_OFFSET; }

int find_closest_energy_cell_bfs(State &state, Pos start_pos, Pos &closest_energy_cell_pos);
void print_cells(State &state, string title = "")
{
    if (title != "")
        fprintf(stderr, "(dPoint=%d) %s\n", count_player_points(state, map_properties.my_id) - count_player_points(state, map_properties.opp_id), title.c_str());

    for (int y = 0; y < MAX_HEIGHT; y++)
    {
        for (int x = 0; x < MAX_WIDTH; x++)
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
                Snake &snake = get_snake(state, cell);
                if (get_snake_player_id(snake) == map_properties.my_id)
                    fprintf(stderr, "🐍");
                else
                    fprintf(stderr, "🐉");
            }
        }
        fprintf(stderr, "\n");
    }
}
void print_map(State &state, string title = "")
{
    if (title != "")
        fprintf(stderr, "(dPoint=%d) %s\n", count_player_points(state, map_properties.my_id) - count_player_points(state, map_properties.opp_id), title.c_str());

    for (int y = 0; y < map_properties.height; y++)
    {
        for (int x = 0; x < map_properties.width; x++)
        {
            // Print emojis
            Pos pos = get_pos_from_map_coord(x, y);
            int cell = get_cell(state, pos);
            if (cell == CELL_EMPTY)
                fprintf(stderr, "⬜");
            else if (cell == CELL_PLATFORM)
                fprintf(stderr, "🟦");
            else if (cell == CELL_ENERGY)
                fprintf(stderr, "🟨");
            else
            {
                Snake &snake = get_snake(state, cell);
                if (get_snake_player_id(snake) == map_properties.my_id)
                    fprintf(stderr, "🐍");
                else
                    fprintf(stderr, "🐉");
            }
        }
        fprintf(stderr, "\n");
    }
}
void print_map_bfs_distances(State &state)
{
    Pos closest = -1;

    for (int y = 0; y < map_properties.height; y++)
    {
        for (int x = 0; x < map_properties.width; x++)
        {
            // Print emojis
            Pos pos = get_pos_from_map_coord(x, y);
            int cell = get_cell(state, pos);
            if (cell == CELL_EMPTY)
            {
                int dist = find_closest_energy_cell_bfs(state, pos, closest);
                fprintf(stderr, "%d ", dist);
            }
            else if (cell == CELL_PLATFORM)
                fprintf(stderr, "# ");
            else if (cell == CELL_ENERGY)
                fprintf(stderr, "o ");
            else
                fprintf(stderr, "S ");
        }
        fprintf(stderr, "\n");
    }
}

/* --- FIRST TURN COMPUTING --- */

struct BFSDistanceToEnergy
{
    int distance;
    Pos energy_pos;
};

// Global lookup table: For each cell, list of energies sorted by distance
BFSDistanceToEnergy cells_to_energy_lookup_table[MAX_CELL_COUNT][MAX_ENERGY_COUNT];

int lookup_initial_bfs_distance_to_closest_energy(State &state, Pos snake_head_pos)
{
    // Iterate from the closest to the farthest energy
    for (int e = 0; e < MAX_ENERGY_COUNT; e++)
    {
        BFSDistanceToEnergy bfsdistance = cells_to_energy_lookup_table[snake_head_pos][e];

        // Verify that the energy still exists
        if (get_cell(state, bfsdistance.energy_pos) == CELL_ENERGY)
            return bfsdistance.distance;
    }

    return -1;
}

void create_bfs_cells_to_energy_distance_lookup_table(State &state)
{
    // BFS from each energy cell outward, blocking only platforms
    for (int ei = 0; ei < state.energy_count; ei++)
    {
        Pos energy_pos = state.energies[ei];

        // Initialize with default distance because some energy cells may be innaccessible
        for (int i = 0; i < MAX_CELL_COUNT; i++)
            cells_to_energy_lookup_table[i][ei] = {MAX_WIDTH + MAX_HEIGHT, energy_pos};

        // Initialize BFS
        bool visited[MAX_CELL_COUNT] = {false};
        queue<pair<Pos, int>> bfs_queue;
        visited[energy_pos] = true;
        cells_to_energy_lookup_table[energy_pos][ei] = {0, energy_pos}; // Distance to self is 0
        bfs_queue.push({energy_pos, 0});

        // BFS loop
        while (!bfs_queue.empty())
        {
            auto [pos, dist] = bfs_queue.front();
            bfs_queue.pop();

            // Check all 4 neighbors
            Pos neighbors[4] = {
                get_north_pos(pos), get_west_pos(pos),
                get_east_pos(pos), get_south_pos(pos)};
            bool oob[4] = {
                is_north_cell_out_of_bounds(pos), is_west_cell_out_of_bounds(pos),
                is_east_cell_out_of_bounds(pos), is_south_cell_out_of_bounds(pos)};

            for (int i = 0; i < 4; i++)
            {
                if (oob[i] || visited[neighbors[i]])
                    continue;
                if (get_cell(state, neighbors[i]) == CELL_PLATFORM)
                    continue;

                visited[neighbors[i]] = true;
                cells_to_energy_lookup_table[neighbors[i]][ei] = {
                    dist + 1,
                    energy_pos};
                bfs_queue.push({neighbors[i], dist + 1});
            }
        }
    }

    // Sort distances for each cell
    for (int cell = 0; cell < MAX_CELL_COUNT; cell++)
    {
        sort(
            cells_to_energy_lookup_table[cell],
            cells_to_energy_lookup_table[cell] + state.energy_count,
            [](const BFSDistanceToEnergy &a, const BFSDistanceToEnergy &b)
            {
                return a.distance < b.distance;
            });
    }
}

void print_bfs_distances_per_energy(State &state)
{
    for (int ei = 0; ei < state.energy_count; ei++)
    {
        Pos energy_pos = state.energies[ei];
        fprintf(stderr, "\n--- BFS Distances for Energy at (%d, %d) ---\n", get_map_x(energy_pos), get_map_y(energy_pos));

        for (int y = 0; y < map_properties.height; y++)
        {
            for (int x = 0; x < map_properties.width; x++)
            {
                Pos pos = get_pos_from_map_coord(x, y);
                int cell = get_cell(state, pos);

                if (cell == CELL_PLATFORM)
                    fprintf(stderr, "#  "); // Platform
                else
                {
                    // Find distance from the cell to the current energy
                    bool displayed = false;
                    for (int e = 0; e < state.energy_count; e++)
                    {
                        if (cells_to_energy_lookup_table[pos][e].energy_pos == energy_pos)
                        {
                            displayed = true;
                            int dist = cells_to_energy_lookup_table[pos][e].distance;
                            fprintf(stderr, "%2d ", dist); // Distance
                            break;
                        }
                    }
                    if (!displayed)
                    {
                        fprintf(stderr, "Energy (%d, %d) not found for cell (%d, %d)\n", get_map_x(energy_pos), get_map_y(energy_pos), get_map_x(pos), get_map_y(pos)); // Unreachable cell
                        exit(0);
                    }
                }
            }
            fprintf(stderr, "\n");
        }
    }
}

void print_map_of_closest_energy_distances(State &state)
{
    fprintf(stderr, "Map of closest energy distances:\n");

    for (int y = 0; y < map_properties.height; y++)
    {
        for (int x = 0; x < map_properties.width; x++)
        {
            Pos pos = get_pos_from_map_coord(x, y);
            int cell = get_cell(state, pos);

            if (cell == CELL_PLATFORM)
                fprintf(stderr, "#  "); // Platform
            else
            {
                // Distance to closest energy
                int dist = cells_to_energy_lookup_table[pos][0].distance;
                fprintf(stderr, "%2d ", dist);
            }
        }
        fprintf(stderr, "\n");
    }
}

void create_lookup_tables(State &state)
{
    auto start_lookup_creation_chrono = chrono::high_resolution_clock::now();

    create_bfs_cells_to_energy_distance_lookup_table(state);

    auto end_lookup_creation_chrono = chrono::high_resolution_clock::now();

    fprintf(stderr, "Lookup tables created in %d ms\n",
            (int)chrono::duration_cast<chrono::milliseconds>(end_lookup_creation_chrono - start_lookup_creation_chrono).count());
}

/* --- GAME PHYSICS - GENERATION --- */

bool is_cell_solid(int cell, int snake_id)
{
    return cell != snake_id && cell != CELL_EMPTY;
}

int generate_snake_moves(Snake &snake, Pos moves[3])
{
    Pos head_pos = get_snake_head_pos(snake);

    // Explore neighbors in order: North, West, East, South
    Pos neighbors[4] = {
        get_west_pos(head_pos),
        get_east_pos(head_pos),
        get_north_pos(head_pos),
        get_south_pos(head_pos)};

    bool neighbor_out_of_bounds[4] = {
        is_west_cell_out_of_bounds(head_pos),
        is_east_cell_out_of_bounds(head_pos),
        is_north_cell_out_of_bounds(head_pos),
        is_south_cell_out_of_bounds(head_pos)};

    int move_count = 0;
    for (int i = 0; i < 4; i++)
    {
        // New valid cell if : In map & Not it neck
        if (!neighbor_out_of_bounds[i] && neighbors[i] != get_snake_body_pos(snake, 1))
        {
            moves[move_count++] = neighbors[i];
        }
    }

    return move_count;
}

int generate_player_movesets_time = 0;
int generate_player_movesets_count = 0;
int generate_player_movesets(State &state, int player_id, MoveSet movesets[MAX_PLAYER_MOVE_SETS])
{
    auto start_chrono = chrono::high_resolution_clock::now();

    int snake_count = get_player_alive_snake_count(state, player_id);
    int snake_ids[snake_count];

    int snake_move_counts[snake_count];
    Pos snake_moves[snake_count][3];

    // Generate moves for all snakes
    for (int i = 0; i < snake_count; i++)
    {
        int snake_id = get_player_alive_snake_id(state, player_id, i);
        Snake &snake = get_snake(state, snake_id);

        snake_ids[i] = snake_id;
        snake_move_counts[i] = generate_snake_moves(snake, snake_moves[i]);
    }

    // The idea is to generate all possible combinations of moves for each snake :
    // Example: snake 0 has 3 moves, snake 1 has 2 moves -> generates 3*2=6 movesets

    // Initialize indices to track which move to use for each snake
    int snake_move_indices[snake_count];
    for (int i = 0; i < snake_count; i++)
    {
        snake_move_indices[i] = 0; // Start with first move for each snake
    }

    int moveset_count = 0;
    while (moveset_count < MAX_PLAYER_MOVE_SETS)
    {
        // Create a moveset with the current combination of moves
        MoveSet moveset;
        moveset.move_count = snake_count;

        // For each snake, add its current move to the moveset
        for (int i = 0; i < snake_count; i++)
        {
            Move &move = get_moveset_move(moveset, i);
            set_move_snake_id(move, snake_ids[i]);
            set_move_dst_pos(move, snake_moves[i][snake_move_indices[i]]);
        }

        movesets[moveset_count++] = moveset;

        // Increment indices like a mixed-radix counter : Exactly like counting, but each digit has different base
        // The rightmost digit increments first, carrying left when it overflows

        // Example if snake 0 has 3 moves, snake 1 has 2 moves :
        // 00 -> 01 -> 10 -> 11 -> 20 -> 21
        int carry = 1;
        for (int i = snake_count - 1; i >= 0 && carry; i--)
        {
            snake_move_indices[i]++;
            // Check if this digit overflowed its base (snake_move_counts[i])
            if (snake_move_indices[i] >= snake_move_counts[i])
            {
                // Reset to 0 and carry to next digit because the digit overflowed
                snake_move_indices[i] = 0;
            }
            else
            {
                carry = 0; // No carry needed, done incrementing
            }
        }

        // If carry is still 1, we've cycled through all combinations
        if (carry)
            break;
    }

    // fprintf(stderr, "generate_player_movesets: %d snakes :\n", snake_count);
    // for (int i = 0; i < snake_count; i++)
    //     fprintf(stderr, "generate_player_movesets: - Snake %d: %d moves\n", i, snake_move_counts[i]);

    // fprintf(stderr, "generate_player_movesets: %d movesets generated :\n", moveset_count);
    // for (int moveset_index = 0; moveset_index < moveset_count; moveset_index++)
    // {
    //     for (int snake_index = 0; snake_index < snake_count; snake_index++)
    //     {
    //         fprintf(stderr, "generate_player_movesets: - Move %d: Snake %d: %d %d\n", moveset_index, snake_index, get_map_x(movesets[moveset_index].moves[snake_index].dst_pos), get_map_y(movesets[moveset_index].moves[snake_index].dst_pos));
    //     }
    // }

    auto end_chrono = chrono::high_resolution_clock::now();
    generate_player_movesets_time += chrono::duration_cast<chrono::microseconds>(end_chrono - start_chrono).count();
    generate_player_movesets_count++;
    return moveset_count;
}

/* --- GAME PHYSICS - APPLICATION --- */

MoveSet merge_movesets(MoveSet &moveset1, MoveSet &moveset2)
{
    MoveSet merged_moveset;
    int moveset1_move_count = get_moveset_move_count(moveset1);
    int moveset2_move_count = get_moveset_move_count(moveset2);

    // TODO: Use memcpy ?

    // Add moves from moveset1
    for (int i = 0; i < moveset1_move_count; i++)
        set_moveset_move(merged_moveset, get_moveset_move(moveset1, i), i);

    // Add moves from moveset2
    for (int i = 0; i < moveset2_move_count; i++)
    {
        set_moveset_move(merged_moveset, get_moveset_move(moveset2, i), moveset1_move_count + i);
    }

    set_moveset_move_count(merged_moveset, moveset1_move_count + moveset2_move_count);

    return merged_moveset;
}

void apply_move(State &state, Snake &snake, Move &move, Pos eaten_energies[MAX_SNAKE_COUNT], int *eaten_energy_count)
{
    // fprintf(stderr, "Applying move for snake %d: (%d, %d)\n", get_snake_id(snake), get_map_x(get_move_dst_pos(move)), get_map_y(get_move_dst_pos(move)));
    Pos new_head_pos = get_move_dst_pos(move);
    int body_length_to_move = get_snake_body_length(snake) - 1;

    // Grow when eating energy
    if (get_cell(state, new_head_pos) == CELL_ENERGY)
    {
        // Remove the energy cell (even if two snakes will collide on it)
        eaten_energies[(*eaten_energy_count)++] = new_head_pos;

        // Move the whole body positions, instead of loosing the tail position
        body_length_to_move = get_snake_body_length(snake);

        // Increase snake length
        set_snake_body_length(snake, get_snake_body_length(snake) + 1);
    }
    else
    {
        // We always remove the tail except when eating energy
        Pos tail_pos = get_snake_body_pos(snake, get_snake_body_length(snake) - 1);
        set_cell(state, tail_pos, CELL_EMPTY);
    }

    // Move the body positions (memmove because source and dest overlap)
    memmove(&snake.body_pos[1], &snake.body_pos[0], sizeof(Pos) * body_length_to_move);

    // Assign a new position to the head
    set_snake_body_pos(snake, 0, new_head_pos);
}

void apply_all_moves(State &state, MoveSet &moveset, Pos eaten_energies[MAX_SNAKE_COUNT], int *eaten_energy_count)
{
    for (int i = 0; i < get_moveset_move_count(moveset); i++)
    {
        Move &move = get_moveset_move(moveset, i);
        Snake &snake = get_snake(state, get_move_snake_id(move));

        // Move snake body & Increase length when eating & Remove tail in cells if not eating
        apply_move(state, snake, move, eaten_energies, eaten_energy_count);
    }
}

void remove_eaten_energies(State &state, Pos eaten_energies[MAX_SNAKE_COUNT], int eaten_energy_count)
{
    for (int i = 0; i < eaten_energy_count; i++)
    {
        set_cell(state, eaten_energies[i], CELL_EMPTY);
        remove_energy(state, eaten_energies[i]);
    }
}

int find_snake_collisions(State &state, Snake *colliding_snakes[MAX_SNAKE_COUNT])
{
    // Find collision & Set new snake pos in cells if not
    int colliding_snake_count = 0;

    for (int snake_index = 0; snake_index < get_alive_snake_count(state); snake_index++)
    {
        // Iter over all alive snakes
        bool collide = false;
        int snake_id = get_alive_snake_id(state, snake_index);
        Snake &snake = get_snake(state, snake_id);
        Pos snake_head_pos = get_snake_head_pos(snake);

        int head_cell = get_cell(state, snake_head_pos);
        if (head_cell == CELL_PLATFORM)
        {
            colliding_snakes[colliding_snake_count++] = &snake;

            // No need to check if it collide with a snake & Its head remain at the same position
            continue;
        }

        for (int snake2_index = 0; snake2_index < get_alive_snake_count(state) && !collide; snake2_index++)
        {
            // Iter over all other alive snakes
            int snake2_id = get_alive_snake_id(state, snake2_index);
            Snake &snake2 = get_snake(state, snake2_id);

            // Check if snake head is colliding with snake2 body
            for (int body_idx = 0; body_idx < get_snake_body_length(snake2) && !collide; body_idx++)
            {
                if (snake_index == snake2_index && body_idx == 0)
                    continue; // Don't check collision with its own head

                Pos snake2_body_pos = get_snake_body_pos(snake2, body_idx);
                if (snake_head_pos == snake2_body_pos)
                {
                    colliding_snakes[colliding_snake_count++] = &snake;

                    // No need to check if it collide with another snake body & Its head remain at the same position
                    collide = true;
                }
            }
        }

        // Snake head has always a new position, except when colliding with platform or snakes
        if (!collide)
            set_cell(state, snake_head_pos, snake_id);
    }

    return colliding_snake_count;
}

void apply_snake_collisions(State &state, Snake *colliding_snakes[MAX_SNAKE_COUNT], int colliding_snake_count)
{
    for (int i = 0; i < colliding_snake_count; i++)
    {
        Snake &snake = *colliding_snakes[i];

        if (get_snake_body_length(snake) - 1 < 3)
        {
            remove_snake_from_alive_snake_ids(state, get_snake_id(snake), get_snake_player_id(snake));
            add_player_loss(state, get_snake_player_id(snake), 3);
        }
        else
        {
            remove_snake_head(snake);
            add_player_loss(state, get_snake_player_id(snake), 1);
        }
    }
}

void kill_snake_immediately(State &state, Snake &snake)
{
    // If not, remove snake from state
    for (int i = 0; i < get_snake_body_length(snake); i++)
        set_cell(state, get_snake_body_pos(snake, i), CELL_EMPTY);

    remove_snake_from_alive_snake_ids(state, get_snake_id(snake), get_snake_player_id(snake));
}

bool apply_snake_gravity(State &state, Snake &snake)
{
    int snake_id = get_snake_id(snake);
    int snake_body_length = get_snake_body_length(snake);
    Pos new_snake_positions[snake_body_length];

    bool gravity_applied = false;
    int y = 0;
    while (y++ < MAX_HEIGHT)
    {
        int min_y = MAX_HEIGHT;

        // First, verify all body parts can move down (check out of bounds for all)
        for (int i = 0; i < snake_body_length; i++)
        {
            Pos pos = get_snake_body_pos(snake, i);
            if (is_south_cell_out_of_bounds(pos))
            {
                kill_snake_immediately(state, snake);
                return true;
            }
        }

        // Then, check if any part is blocked by solid ground
        for (int i = 0; i < snake_body_length; i++)
        {
            Pos pos = get_snake_body_pos(snake, i);
            Pos pos_below = get_south_pos(pos);
            int cell_below = get_cell(state, pos_below);

            // fprintf(stderr, "Snake %d: Is cell %d solid? (Pos=%d)\n", snake_id, cell_below, pos);
            if (is_cell_solid(cell_below, snake_id))
                return false;

            min_y = min(min_y, get_y(pos_below));
            new_snake_positions[i] = pos_below;
        }

        // If not, remove snake from state
        for (int i = 0; i < snake_body_length; i++)
            set_cell(state, get_snake_body_pos(snake, i), CELL_EMPTY);

        // Useless
        // If its maximum position is the map height, it's falling under the map
        // Do not add the snake in the cells
        if (min_y >= map_properties.height + MAP_PADDING)
        {
            kill_snake_immediately(state, snake);
            return true;
        }

        // And set it one cell below
        for (int i = 0; i < snake_body_length; i++)
        {
            set_cell(state, new_snake_positions[i], snake_id);
            set_snake_body_pos(snake, i, new_snake_positions[i]);
        }

        gravity_applied = true;
    }

    return gravity_applied;
}

int apply_gravity_time = 0;
int apply_gravity_count = 0;
void apply_gravity(State &state)
{
    auto start_chrono = chrono::high_resolution_clock::now();

    int gravity_finalized_count = 0;
    int snake_index = 0;

    // Iter until 'alive snake count' consecutive gravity is useless
    while (gravity_finalized_count < get_alive_snake_count(state))
    {
        int snake_id = get_alive_snake_id(state, snake_index);
        Snake &snake = get_snake(state, snake_id);

        bool gravity_applied = apply_snake_gravity(state, snake);
        if (gravity_applied)
            gravity_finalized_count = 0;
        else
            gravity_finalized_count++;

        // Loop over alive snakes indefinitely
        if (++snake_index >= get_alive_snake_count(state))
            snake_index = 0;
    }

    auto end_chrono = chrono::high_resolution_clock::now();
    apply_gravity_time += chrono::duration_cast<chrono::microseconds>(end_chrono - start_chrono).count();
    apply_gravity_count++;
}

int apply_moveset_time = 0;
int apply_moveset_count = 0;
void apply_moveset(State &state, MoveSet &moveset)
{
    auto start_chrono = chrono::high_resolution_clock::now();

    set_turn(state, get_turn(state) + 1);

    Pos eaten_energies[MAX_SNAKE_COUNT];
    int eaten_energies_count = 0;
    apply_all_moves(state, moveset, eaten_energies, &eaten_energies_count);

    remove_eaten_energies(state, eaten_energies, eaten_energies_count);

    // Find collision & Set new snake pos in cells if not
    Snake *colliding_snakes[MAX_SNAKE_COUNT];
    int colliding_snake_count = find_snake_collisions(state, colliding_snakes);

    // Reset snake positions if we found a collision & Kill snakes if needed
    apply_snake_collisions(state, colliding_snakes, colliding_snake_count);

    apply_gravity(state);

    if (get_turn(state) == 200 ||
        get_player_alive_snake_count(state, map_properties.my_id) == 0 ||
        get_player_alive_snake_count(state, map_properties.opp_id) == 0 ||
        get_energy_count(state) == 0)
        set_game_ended(state);

    auto end_chrono = chrono::high_resolution_clock::now();
    apply_moveset_time += chrono::duration_cast<chrono::microseconds>(end_chrono - start_chrono).count();
    apply_moveset_count++;
}

/* --- ENERGY DISTANCE - BFS --- */

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

            visited[neighbors[i]] = true;

            if (neighbor == CELL_EMPTY)
                bfs_queue.push(make_pair(neighbors[i], distance + 1));
            // fprintf(stderr, "BFS: Push (%d, %d) with distance %d\n", get_x(neighbors[i]), get_y(neighbors[i]), distance + 1);
        }
    }

    // Recursively process the rest of the queue
    return find_closest_energy_cell_recursive(state, bfs_queue, visited, closest_energy_cell_pos);
}

int bfs_recursive_time = 0;
int bfs_recursive_count = 0;
int find_closest_energy_cell_bfs(State &state, Pos start_pos, Pos &closest_energy_cell_pos)
{
    auto start_chrono = chrono::high_resolution_clock::now();

    // Check if the starting position is an energy cell
    // useless
    if (get_cell(state, start_pos) == CELL_ENERGY)
    {
        closest_energy_cell_pos = start_pos;
        return 0;
    }

    bool visited[MAX_CELL_COUNT] = {false};
    queue<pair<Pos, int>> bfs_queue;
    visited[start_pos] = true;
    bfs_queue.push(make_pair(start_pos, 0));

    int ret = find_closest_energy_cell_recursive(state, bfs_queue, visited, closest_energy_cell_pos);

    auto end_chrono = chrono::high_resolution_clock::now();
    bfs_recursive_time += chrono::duration_cast<chrono::microseconds>(end_chrono - start_chrono).count();
    bfs_recursive_count++;
    return ret;
}

int bfs_iterative_time = 0;
int bfs_iterative_count = 0;
int find_closest_energy_cell_bfs_iterative(State &state, Pos start_pos, Pos &closest_energy_cell_pos)
{
    auto start_chrono = chrono::high_resolution_clock::now();

    // Initialize BFS
    bool visited[MAX_CELL_COUNT] = {false};
    visited[start_pos] = true;

    queue<pair<Pos, int>> bfs_queue;
    bfs_queue.push({start_pos, 0});

    // BFS loop
    while (!bfs_queue.empty())
    {
        auto [pos, dist] = bfs_queue.front();
        bfs_queue.pop();

        // Check all 4 neighbors
        Pos neighbors[4] = {
            get_north_pos(pos), get_west_pos(pos),
            get_east_pos(pos), get_south_pos(pos)};
        bool oob[4] = {
            is_north_cell_out_of_bounds(pos), is_west_cell_out_of_bounds(pos),
            is_east_cell_out_of_bounds(pos), is_south_cell_out_of_bounds(pos)};

        for (int i = 0; i < 4; i++)
        {
            // Dot not go out of bounds or visit already visited cells
            if (oob[i] || visited[neighbors[i]])
                continue;

            // Closest energy cell found !
            if (get_cell(state, neighbors[i]) == CELL_ENERGY)
            {
                closest_energy_cell_pos = neighbors[i];
                auto end_chrono = chrono::high_resolution_clock::now();
                bfs_iterative_time += chrono::duration_cast<chrono::microseconds>(end_chrono - start_chrono).count();
                bfs_iterative_count++;
                return dist + 1;
            }

            // New cell found, add it to the queue if we can move on it
            visited[neighbors[i]] = true;
            if (get_cell(state, neighbors[i]) == CELL_EMPTY)
            {
                bfs_queue.push({neighbors[i], dist + 1});
            }
        }
    }

    auto end_chrono = chrono::high_resolution_clock::now();
    bfs_iterative_time += chrono::duration_cast<chrono::microseconds>(end_chrono - start_chrono).count();
    bfs_iterative_count++;
    return -1; // No energy cell found
}

/* --- ENERGY DISTANCE - MANHATTAN--- */

int find_closest_energy_cell_manhattan(State &state, Pos start_pos, Pos &closest_energy_cell_pos)
{
    int closest_distance = -1;

    for (int i = 0; i < get_energy_count(state); i++)
    {
        Pos energy_pos = get_energy(state, i);
        int distance = abs(get_x(energy_pos) - get_x(start_pos)) + abs(get_y(energy_pos) - get_y(start_pos));

        if (closest_distance == -1 || distance < closest_distance)
        {
            closest_distance = distance;
            closest_energy_cell_pos = energy_pos;
        }
    }

    return closest_distance;
}

/* --- DECISION MAKING --- */

int count_player_points(State &state, int player_id)
{
    // Sum lengths of alive snakes for my player
    int points = 0;
    for (int i = 0; i < get_player_alive_snake_count(state, player_id); i++)
    {
        int snake_id = get_player_alive_snake_id(state, player_id, i);
        Snake &snake = get_snake(state, snake_id);
        points += get_snake_body_length(snake);
    }

    return points;
}

int encode_lexicographic_priority(int a, int b, int b_max)
{
    // Multiply A by maximum B value to prioritize A changes over any B changes
    return a * b_max + b;
}

// int encode_lexicographic_priority(int a, int b, int b_max, int c, int c_max)
// {
//     b = encode_lexicographic_priority(b, c, c_max);
//     b_max = b_max * c_max;

//     // Multiply A by maximum BC value to prioritize A changes over any BC changes
//     return encode_lexicographic_priority(a, b, b_max);
// }

int evaluate_end_states(State &state, int player_id, int player_points, int opponent_points)
{
    bool player_win = false;
    bool opponent_win = false;

    // Game ends by turn count or missing energies
    if (get_turn(state) == 200 || get_energy_count(state) == 0)
    {
        // Wins by score
        if (player_points > opponent_points)
            player_win = true;
        else if (player_points < opponent_points)
            opponent_win = true;
        else if (get_player_losses(state, player_id) > get_player_losses(state, get_opponent_id(player_id)))
            opponent_win = true;
        else if (get_player_losses(state, player_id) < get_player_losses(state, get_opponent_id(player_id)))
            player_win = true;
    }
    // Wins by killing enemy snakes
    else if (opponent_points == 0)
        player_win = true;
    else if (player_points == 0)
        opponent_win = true;
    else if (get_player_losses(state, player_id) > get_player_losses(state, get_opponent_id(player_id)))
        opponent_win = true;
    else if (get_player_losses(state, player_id) < get_player_losses(state, get_opponent_id(player_id)))
        player_win = true;

    // End positions stay in beam search states, with different turn count
    // Start from a positive score if we win, and increase it by :
    // - player points: We should try to grow or not lose snakes
    // - turn: We should try to win as fast as possible
    if (player_win)
        return 1000 + encode_lexicographic_priority(player_points, (201 - get_turn(state)), 200);

    // Start from a negative score if we lose, and increase it by :
    // - player points: We should still try to grow or not lose snakes
    // - turn: We should try to delay the lose
    // 100 player point max * 200 turns max = 20 000 -> return -1000 maximum
    if (opponent_win)
        return -21000 + encode_lexicographic_priority(player_points, get_turn(state), 200);

    // Draw
    return 0;
}

int evaluate_state_time = 0;
int evaluate_state_count = 0;
float evaluate_state(State &state, int player_id)
{
    auto start_chrono = chrono::high_resolution_clock::now();

    int player_points;
    int opponent_points;
    if (player_id == map_properties.my_id)
    {
        player_points = count_player_points(state, map_properties.my_id);
        opponent_points = count_player_points(state, map_properties.opp_id);
    }
    else
    {
        player_points = count_player_points(state, map_properties.opp_id);
        opponent_points = count_player_points(state, map_properties.my_id);
    }

    if (is_game_ended(state))
        return evaluate_end_states(state, player_id, player_points, opponent_points);

    int dist_sum = 0;
    int platform_bonuses = 0;
    for (int i = 0; i < get_player_alive_snake_count(state, player_id); i++)
    {
        int snake_id = get_player_alive_snake_id(state, player_id, i);
        Snake &snake = get_snake(state, snake_id);
        Pos snake_head_pos = get_snake_head_pos(snake);

        // TODO: Add maluses/bonuses about surviving

        Pos closest;
        int dist;
        dist = lookup_initial_bfs_distance_to_closest_energy(state, snake_head_pos);
        // dist = find_closest_energy_cell_manhattan(state, snake_head_pos, closest);
        // dist = find_closest_energy_cell_bfs(state, snake_head_pos, closest);
        // dist = find_closest_energy_cell_bfs_iterative(state, snake_head_pos, closest);
        if (dist == -1)
            dist_sum += MAX_MAP_WIDTH + MAX_MAP_HEIGHT; // If no energy cell found, consider it very far
        else
            dist_sum += dist;

        for (int bi = 0; bi < get_snake_body_length(snake); bi++)
        {
            Pos snake_body_pos = get_snake_body_pos(snake, bi);

            // Add maluses for being out of map
            if (get_x(snake_body_pos) < MAP_PADDING ||
                get_x(snake_body_pos) >= MAX_WIDTH - MAP_PADDING ||
                get_y(snake_body_pos) < MAP_PADDING ||
                get_y(snake_body_pos) >= MAX_HEIGHT - MAP_PADDING)
            {
                dist_sum += 1;
            }
        }

        // Add bonuses depending on the first body index on a platform. The closer to the head, the more bonus.
        // The idea is that reaching a platform to navigate could be interesting even if we move away from the energy
        for (int bi = 0; bi < get_snake_body_length(snake); bi++)
        {
            Pos snake_body_pos = get_snake_body_pos(snake, bi);

            Pos cell_under = get_south_pos(snake_body_pos);
            if (get_cell(state, cell_under) == CELL_PLATFORM)
            {
                // Snake bonus in ]0, 1.5]
                // So: head on platform = -1.5 dist
                // Should be >1 to encourage sidesteps over staying at the same position due tu gravity
                // Should be <2 to discourage sidesteps over going directly onto the energy
                platform_bonuses += 1.5 * (get_snake_body_length(snake) - bi) / get_snake_body_length(snake);
                break;
            }
        }
    }

    dist_sum -= platform_bonuses;

    // Multiply dist so distance=1 doesn't be equivalent to a game point
    float dist_score = dist_sum != 0 ? 1.0 / (2 * dist_sum) : 0.0;

    auto end_chrono = chrono::high_resolution_clock::now();
    evaluate_state_time += chrono::duration_cast<chrono::microseconds>(end_chrono - start_chrono).count();
    evaluate_state_count++;

    // Game points is positive if it's good for the player, negative if it's good for its opponent, so we invert it for opponent evaluation
    return player_points - opponent_points + dist_score;
}

int choose_best_player_moveset_time = 0;
int choose_best_player_moveset_count = 0;
MoveSet choose_best_player_moveset(State &state, int player_id, MoveSet &previous_player_moveset)
{
    auto start_chrono = chrono::high_resolution_clock::now();

    State next_state;
    // fprintf(stderr, "Choosing move set for player %d ...\n", player_id);

    MoveSet movesets[MAX_PLAYER_MOVE_SETS];
    int moveset_count = generate_player_movesets(state, player_id, movesets);

    MoveSet best_moveset;
    float best_evaluation = -100000;
    for (int i = 0; i < moveset_count; i++)
    {
        revert_last_move(state, next_state);

        MoveSet turn_moveset = merge_movesets(previous_player_moveset, movesets[i]);
        apply_moveset(next_state, turn_moveset);

        float evaluation = evaluate_state(next_state, player_id);
        if (best_evaluation < evaluation)
        {
            // fprintf(stderr, "New best moveset found for player %d with evaluation %f: ", player_id, evaluation);
            // print_moveset(turn_moveset);
            // fprintf(stderr, "\n");

            best_evaluation = evaluation;
            best_moveset = movesets[i];
        }
    }

    auto end_chrono = chrono::high_resolution_clock::now();
    choose_best_player_moveset_time += chrono::duration_cast<chrono::microseconds>(end_chrono - start_chrono).count();
    choose_best_player_moveset_count++;
    return best_moveset;
}

int choose_best_snake_moves_time = 0;
int choose_best_snake_moves_count = 0;
MoveSet choose_best_snake_moves(State &state, int player_id)
{
    auto start_chrono = chrono::high_resolution_clock::now();

    MoveSet best_snake_moves;
    State next_state;

    int snake_count = get_player_alive_snake_count(state, player_id);

    for (int i = 0; i < snake_count; i++)
    {
        int snake_id = get_player_alive_snake_id(state, player_id, i);
        Snake &snake = get_snake(state, snake_id);

        Pos snake_moves[3];
        int snake_move_count = generate_snake_moves(snake, snake_moves);

        int best_move_evaluation = -100000;
        Move best_snake_move = {};
        for (int j = 0; j < snake_move_count; j++)
        {
            // Construct a moveset with only this move for this snake
            MoveSet snake_moveset;
            set_moveset_move_count(snake_moveset, 1);
            Move &move = get_moveset_move(snake_moveset, 0);
            set_move_snake_id(move, snake_id);
            set_move_dst_pos(move, snake_moves[j]);

            // Still include other player snake moves already selected, to not generate dump movesets
            MoveSet turn_moveset = merge_movesets(best_snake_moves, snake_moveset);

            memcpy(&next_state, &state, sizeof(State));
            apply_moveset(next_state, turn_moveset);

            float evaluation = evaluate_state(next_state, player_id);

            if (evaluation > best_move_evaluation)
            {
                best_move_evaluation = evaluation;
                best_snake_move = move;
            }
        }

        set_moveset_move(best_snake_moves, best_snake_move, i);
        set_moveset_move_count(best_snake_moves, i + 1);
    }

    auto end_chrono = chrono::high_resolution_clock::now();
    choose_best_snake_moves_time += chrono::duration_cast<chrono::microseconds>(end_chrono - start_chrono).count();
    choose_best_snake_moves_count++;
    return best_snake_moves;
}

void print_marks(State &state, MoveSet best_moveset)
{
    for (int i = 0; i < get_moveset_move_count(best_moveset); i++)
    {
        Move &move = get_moveset_move(best_moveset, i);
        int snake_id = get_move_snake_id(move);
        Snake &snake = get_snake(state, snake_id);

        Pos closest;
        int dist = find_closest_energy_cell_bfs_iterative(state, get_snake_head_pos(snake), closest);

        if (dist != -1)
            cout << "MARK " << get_map_x(closest) << " " << get_map_y(closest) << ";";
    }
}

/* --- ALGORITHM - BEAM SEARCH --- */

struct CompareState
{
    bool operator()(State *a, State *b) const
    {
        return get_heuristic(*a) > get_heuristic(*b);
    }
};

int beam_search_depth = 0;
int beam_search_execution_count = 0;
int beam_search_visited_states_count = 0;
float beam_search_sum_states_visited = 0;
float beam_search_average_states_visited = 0;

bool has_exceeded_time_limit(auto &start_chrono, int maximum_microseconds)
{
    auto current_chrono = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::microseconds>(current_chrono - start_chrono).count() >= maximum_microseconds;
}

int consider_state_to_be_candidate_time = 0;
int consider_state_to_be_candidate_count = 0;
void consider_state_to_be_candidate(State *state_addr, MoveSet &last_moveset, std::priority_queue<State *, std::vector<State *>, CompareState> *beam_candidates_queue, int beam_width)
{
    auto start_chrono = chrono::high_resolution_clock::now();
    beam_search_visited_states_count++;

    // fprintf(stderr, "consider_state_to_be_candidate (%p): Candidate queue size: %d\n", (int)beam_candidates_queue->size());

    State &state = *state_addr;
    if (beam_candidates_queue->size() < beam_width)
    {
        // fprintf(stderr, "consider_state_to_be_candidate (%p): Adding state to candidate queue\n", state_addr);

        // Save the candidate moveset for the current game turn choice
        // Can only happen here because max_moveset_count < beam_width, so we're not comparing heuristic on the first turn
        if (beam_search_depth == 1)
        {
            // fprintf(stderr, "consider_state_to_be_candidate (%p): Saving moveset for first depth\n", state_addr);

            // if (get_moveset_move_count(last_moveset) == 0)
            // {
            //     fprintf(stderr, "consider_state_to_be_candidate: Depth %d: last_moveset.move_count == 0: No moveset found for state %p\n", beam_search_depth, &state);
            //     exit(0);
            // }

            set_first_depth_moveset(state, last_moveset);

            // if (get_moveset_move_count(get_first_depth_moveset(state)) == 0)
            // {
            //     fprintf(stderr, "consider_state_to_be_candidate: Depth %d: get_moveset_move_count(get_first_depth_moveset(state)): No moveset found for state %p\n", beam_search_depth, &state);
            //     exit(0);
            // }
            // print_moveset(last_moveset);
        }

        beam_candidates_queue->push(state_addr);
    }
    // Save the new promising state if it's better than the current "worst best score"
    else if (get_heuristic(state) > get_heuristic((State &)beam_candidates_queue->top()))
    {
        // fprintf(stderr, "consider_state_to_be_candidate (%p): Replacing worst state in candidate queue\n", state_addr);
        if (beam_search_depth == 1)
            set_first_depth_moveset(state, last_moveset);

        beam_candidates_queue->pop();            // Remove the worst state
        beam_candidates_queue->push(state_addr); // Add the new state
    }
    // else
    // {
    //     fprintf(stderr, "consider_state_to_be_candidate: State %p not a candidate because its h=%f < h=%d from current candidate top\n", state_addr, get_heuristic(state), get_heuristic((State &)beam_candidates_queue->top()));
    // }

    // fprintf(stderr, "consider_state_to_be_candidate (%p): end\n");

    auto end_chrono = chrono::high_resolution_clock::now();
    consider_state_to_be_candidate_time += chrono::duration_cast<chrono::microseconds>(end_chrono - start_chrono).count();
    consider_state_to_be_candidate_count++;
}

int find_candidates_among_state_children_time = 0;
int find_candidates_among_state_children_count = 0;
void find_candidates_among_state_children(State &state, int player_id, State *beam_candidates, int *next_candidate_index, std::priority_queue<State *, std::vector<State *>, CompareState> *beam_candidates_queue, int beam_width)
{
    auto start_chrono = chrono::high_resolution_clock::now();

    // fprintf(stderr, "find_candidates_among_state_children (%p): player_id=%d\n", &state, player_id);

    MoveSet turn_beginning_moveset = {};
    set_moveset_move_count(turn_beginning_moveset, 0);

    MoveSet opponent_moveset = choose_best_player_moveset(state, get_opponent_id(player_id), turn_beginning_moveset);

    MoveSet ally_movesets[MAX_PLAYER_MOVE_SETS];
    int ally_moveset_count = generate_player_movesets(state, player_id, ally_movesets);

    // fprintf(stderr, "find_candidates_among_state_children: %d moves generated\n", ally_moveset_count);
    // if (ally_moveset_count == 0)
    // {
    //     print_map(state);
    //     exit(0);
    // }

    for (int i = 0; i < ally_moveset_count; i++)
    {
        // print_moveset(ally_movesets[i]);

        State *next_state_addr = &beam_candidates[*next_candidate_index];
        (*next_candidate_index)++;

        // fprintf(stderr, "find_candidates_among_state_children (%p): Pick new state from buffer index %d: %p\n", &state, *next_candidate_index - 1, next_state_addr);

        State &next_state = *next_state_addr;
        revert_last_move(state, next_state);

        // Preserve the first depth moveset from parent
        // if (beam_search_depth > 1)
        // {
        //     set_first_depth_moveset(next_state, get_first_depth_moveset(state));
        // }

        // if (beam_search_depth > 1 && get_moveset_move_count(get_first_depth_moveset(next_state)) == 0)
        // {
        //     fprintf(stderr, "find_candidates_among_state_children: Depth %d: No moveset found for state %p\n", beam_search_depth, &next_state);
        //     exit(0);
        // }

        MoveSet turn_moveset = merge_movesets(opponent_moveset, ally_movesets[i]);
        apply_moveset(next_state, turn_moveset);

        float heuristic = evaluate_state(next_state, player_id);
        set_heuristic(next_state, heuristic);

        consider_state_to_be_candidate(next_state_addr, ally_movesets[i], beam_candidates_queue, beam_width);
    }

    auto end_chrono = chrono::high_resolution_clock::now();
    find_candidates_among_state_children_time += chrono::duration_cast<chrono::microseconds>(end_chrono - start_chrono).count();
    find_candidates_among_state_children_count++;
}

State &get_best_candidate(State *beam_states, int beam_states_length, State *beam_candidates, int beam_candidates_length)
{
    State *best_state = &beam_states[0];
    int best_heuristic = get_heuristic(*best_state);

    for (int i = 1; i < beam_states_length; i++)
    {
        State &state = beam_states[i];
        if (get_heuristic(state) > best_heuristic)
        {
            best_heuristic = get_heuristic(state);
            best_state = &state;
        }
    }

    for (int i = 0; i < beam_candidates_length; i++)
    {
        State &state = beam_candidates[i];
        if (get_heuristic(state) > best_heuristic)
        {
            best_heuristic = get_heuristic(state);
            best_state = &state;
        }
    }

    return *best_state;
}

MoveSet beam_search(State &initial_state, int player_id, int depth_max, int beam_width, int maximum_microseconds, auto start_turn_chrono)
{
    fprintf(stderr, "Starting beam_search: player_id=%d, depth_max=%d, beam_width=%d, max_time_us=%d\n", player_id, depth_max, beam_width, maximum_microseconds);
    beam_search_visited_states_count = 0;
    beam_search_depth = 0;

    State *beam_states_1 = new State[beam_width * MAX_PLAYER_MOVE_SETS];
    State *beam_states_2 = new State[beam_width * MAX_PLAYER_MOVE_SETS];
    State *beam_states;
    State *beam_candidates;
    int beam_states_length = 0;
    int beam_candidates_length = 1;

    std::vector<State *> storage;
    std::vector<State *> storage2;
    storage.reserve(beam_width);
    storage2.reserve(beam_width);
    std::priority_queue<State *, std::vector<State *>, CompareState> beam_queue_1(CompareState(), std::move(storage));
    std::priority_queue<State *, std::vector<State *>, CompareState> beam_queue_2(CompareState(), std::move(storage2));
    std::priority_queue<State *, std::vector<State *>, CompareState> *beam_depth_queue;
    std::priority_queue<State *, std::vector<State *>, CompareState> *beam_candidates_queue;

    // Initialize the beam depth queue with the initial state children
    memcpy(&beam_states_1[0], &initial_state, sizeof(State));
    beam_queue_1.push(&beam_states_1[0]);
    bool use_second_queue_as_candidates = true;

    while (!has_exceeded_time_limit(start_turn_chrono, maximum_microseconds) && beam_search_depth < depth_max)
    {
        beam_search_depth++;
        fprintf(stderr, "Start beam search depth %d (%ld ym remaining)\n", beam_search_depth, maximum_microseconds - chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start_turn_chrono).count());

        if (use_second_queue_as_candidates)
        {
            beam_states = beam_states_1;
            beam_candidates = beam_states_2;
            beam_depth_queue = &beam_queue_1;
            beam_candidates_queue = &beam_queue_2;
        }
        else
        {
            beam_states = beam_states_2;
            beam_candidates = beam_states_1;
            beam_depth_queue = &beam_queue_2;
            beam_candidates_queue = &beam_queue_1;
        }
        use_second_queue_as_candidates = !use_second_queue_as_candidates;

        beam_states_length = beam_candidates_length;
        beam_candidates_length = 0;
        // fprintf(stderr, "beam_search: Depth %d start with %d states in queue, and %d states took from the previous buffer\n", beam_search_depth, beam_depth_queue->size(), beam_states_length);
        // for (int i = 0; i < beam_depth_queue->size(); i++)
        // {
        //     fprintf(stderr, "State %p: beam_candidates[%d].turn=%d, beam_candidates[%d].first_depth_moveset.size()=%d\n", &beam_states[i], i, beam_states[i].turn, i, beam_states[i].first_depth_moveset.move_count);
        // }
        while (!beam_depth_queue->empty())
        {
            State *state_ptr = beam_depth_queue->top();
            beam_depth_queue->pop();

            State &state = *state_ptr;

            // if (beam_search_depth > 1 && get_moveset_move_count(get_first_depth_moveset(state)) == 0)
            // {
            //     fprintf(stderr, "beam_search: Depth %d: No moveset found for state %p\n", beam_search_depth, &state);
            //     exit(0);
            // }
            // fprintf(stderr, "beam_search depth %d: Pop state from the %d available\n", beam_search_depth, beam_states_length);
            // print_map(state);

            // Skip ended states, but consider keeping it in candidates state
            if (is_game_ended(state))
            {
                // fprintf(stderr, "Game ended, considering state as candidate\n");
                consider_state_to_be_candidate(state_ptr, get_first_depth_moveset(state), beam_candidates_queue, beam_width);
                continue;
            }

            find_candidates_among_state_children(state, player_id, beam_candidates, &beam_candidates_length, beam_candidates_queue, beam_width);

            if (has_exceeded_time_limit(start_turn_chrono, maximum_microseconds))
            {
                fprintf(stderr, "Time limit exceeded during moveset generation. beam_states_length=%d, beam_candidates_length=%d\n", beam_states_length, beam_candidates_length);

                State &best_candidate = get_best_candidate(beam_states, beam_states_length, beam_candidates, beam_candidates_length);
                print_moveset(get_first_depth_moveset(best_candidate));
                print_map(best_candidate);
                return get_first_depth_moveset(best_candidate);
            }

            // Verify all states have first depth moveset and depth=2
            // fprintf(stderr, "beam_search: Depth %d end with %d candidates in queue, and %d state took from the buffer\n", beam_search_depth, beam_candidates_queue->size(), beam_candidates_length);
            // for (int i = 0; i < beam_candidates_length; i++)
            // {
            //     fprintf(stderr, "State %p: beam_candidates[%d].turn=%d, beam_candidates[%d].first_depth_moveset.size()=%d\n", &beam_candidates[i], i, beam_candidates[i].turn, i, beam_candidates[i].first_depth_moveset.move_count);
            // }
        }
    }

    State &best_candidate = get_best_candidate(beam_states, beam_states_length, beam_candidates, beam_candidates_length);
    print_moveset(get_first_depth_moveset(best_candidate));
    print_map(best_candidate);
    return get_first_depth_moveset(best_candidate);
}

/* --- PARSING --- */

void parse_initial_inputs(State &state)
{
    cin >> map_properties.my_id;
    map_properties.opp_id = get_opponent_id(map_properties.my_id);

    cin >> map_properties.width;
    cin >> map_properties.height;

    initialize_cells(state);
    cin.ignore();
    for (int y = 0; y < map_properties.height; y++)
    {
        string row;
        getline(cin, row);

        for (int x = 0; x < map_properties.width; x++)
        {
            // Convert map coordinates to internal data structure coordinates
            Pos pos = get_pos_from_map_coord(x, y);

            if (row[x] == '.')
                set_cell(state, pos, CELL_EMPTY);
            else if (row[x] == '#')
                set_cell(state, pos, CELL_PLATFORM);
        }
    }

    int snakebots_per_player;
    cin >> snakebots_per_player;

    reset_alive_snake_count(state);

    int snakebot_id;
    for (int i = 0; i < snakebots_per_player; i++)
    {
        cin >> snakebot_id;
        initialize_snake_data(state, snakebot_id, map_properties.my_id);
        add_player_alive_snake_id(state, map_properties.my_id, snakebot_id);
    }
    for (int i = 0; i < snakebots_per_player; i++)
    {
        cin >> snakebot_id;
        initialize_snake_data(state, snakebot_id, map_properties.opp_id);
        add_player_alive_snake_id(state, map_properties.opp_id, snakebot_id);
    }
}

bool parse_pos_from_segment(string segment, Pos &pos)
{
    size_t commaPos = segment.find(',');
    int x = stoi(segment.substr(0, commaPos));
    int y = stoi(segment.substr(commaPos + 1));

    if (x < -MAP_PADDING || x >= map_properties.width + MAP_PADDING || y < -MAP_PADDING || y >= map_properties.height + MAP_PADDING)
        return false;

    // Convert map coordinates to internal data structure coordinates
    pos = get_pos_from_map_coord(x, y);
    return true;
}

void parse_snakebot(State &state, Snake &snake, int snakebotId, string bodyStr)
{
    reset_snake_length(snake);

    // Parse the body string into coordinates
    size_t start = 0;
    size_t end = bodyStr.find(':');

    while (end != string::npos)
    {
        string segment = bodyStr.substr(start, end - start);

        Pos body_pos;
        if (parse_pos_from_segment(segment, body_pos))
        {
            add_body_pos(snake, body_pos);
            set_cell(state, body_pos, snakebotId);
        }

        start = end + 1;
        end = bodyStr.find(':', start);
    }

    // Add the last segment
    string lastSegment = bodyStr.substr(start);
    Pos body_pos;
    if (parse_pos_from_segment(lastSegment, body_pos))
    {
        add_body_pos(snake, body_pos);
        set_cell(state, body_pos, snakebotId);
    }
}

void parse_turn_inputs(State &state)
{
    int energy_count;
    cin >> energy_count;
    for (int i = 0; i < energy_count; i++)
    {
        int x, y;
        cin >> x >> y;

        // Convert map coordinates to internal data structure coordinates
        Pos pos = get_pos_from_map_coord(x, y);

        set_energy(state, i, pos);
        set_cell(state, state.energies[i], CELL_ENERGY);
    }
    set_energy_count(state, energy_count);

    reset_alive_snake_count(state);

    int snakebotCount;
    cin >> snakebotCount;
    for (int i = 0; i < snakebotCount; i++)
    {
        int snakebotId;
        string bodyStr;
        cin >> snakebotId >> bodyStr;

        Snake &snake = get_snake(state, snakebotId);
        int player_id = get_snake_player_id(snake);
        add_player_alive_snake_id(state, player_id, snakebotId);

        parse_snakebot(state, snake, snakebotId, bodyStr);
    }
}

/* --- MAIN LOOP --- */

void signal_handler(int signal)
{
    const char *signal_name = nullptr;
    switch (signal)
    {
    case SIGSEGV:
        signal_name = "SIGSEGV (Segmentation Fault)";
        break;
    case SIGABRT:
        signal_name = "SIGABRT (Abort)";
        break;
    case SIGFPE:
        signal_name = "SIGFPE (Floating Point Exception)";
        break;
    default:
        signal_name = "Unknown signal";
        break;
    }

    // Print error message to stderr
    std::cerr << "\n[ERROR] Caught signal: " << signal_name << " (" << signal << ")\n";

    // Exit gracefully
    exit(EXIT_FAILURE);
}

int main()
{
    // Register signal handlers
    signal(SIGSEGV, signal_handler); // Segmentation fault
    signal(SIGABRT, signal_handler); // Abort
    signal(SIGFPE, signal_handler);  // Floating point exception

    State initial_state;
    bzero(&initial_state, sizeof(initial_state));
    parse_initial_inputs(initial_state);

    // Log struct size
    fprintf(stderr, "State size: %zu bytes\n", sizeof(State));
    fprintf(stderr, "Snake size: %zu bytes\n", sizeof(Snake));
    fprintf(stderr, "MoveSet size: %zu bytes\n", sizeof(MoveSet));
    fprintf(stderr, "Move size: %zu bytes\n", sizeof(Move));

    // Save initial state for resetting cells each turn
    State state;
    while (true)
    {
        // Save internal data before overwrite
        int turn = state.turn;
        int player_losses = get_player_losses(state, map_properties.my_id);
        int opponent_losses = get_player_losses(state, map_properties.opp_id);

        // Reset state to initial state before parsing fresh dynamic data
        memcpy(&state, &initial_state, sizeof(state));

        // Restore internal data
        set_turn(state, turn);
        set_player_losses(state, map_properties.my_id, player_losses);
        set_player_losses(state, map_properties.opp_id, opponent_losses);

        parse_turn_inputs(state);
        auto start_turn_chrono = chrono::high_resolution_clock::now();

        if (turn == 0)
            create_lookup_tables(state);

        // print_cells(state, "Turn beginning");
        // print_map(state, "Turn beginning");
        // print_map_bfs_distances(state);

        MoveSet best_moveset;
        try
        {
            best_moveset = beam_search(state, map_properties.my_id, 20, 150, 30000, start_turn_chrono);

            beam_search_execution_count++;
            beam_search_sum_states_visited += beam_search_visited_states_count;
            beam_search_average_states_visited = beam_search_sum_states_visited / (float)beam_search_execution_count;
        }
        catch (const std::exception &e)
        {
            fprintf(stderr, "\n========== EXCEPTION IN BEAM_SEARCH ==========\n");
            fprintf(stderr, "Exception Type: std::exception\n");
            fprintf(stderr, "Exception Message: %s\n", e.what());
            fprintf(stderr, "\n--- PLAYER INFO ---\n");
            fprintf(stderr, "My ID: %d\n", map_properties.my_id);
            fprintf(stderr, "Opponent ID: %d\n", map_properties.opp_id);
            fprintf(stderr, "My Alive Snakes: %d\n", get_player_alive_snake_count(state, map_properties.my_id));
            fprintf(stderr, "Opponent Alive Snakes: %d\n", get_player_alive_snake_count(state, map_properties.opp_id));
            fprintf(stderr, "Total Alive Snakes: %d\n", get_alive_snake_count(state));
            fprintf(stderr, "===========================================\n\n");
            exit(1);
        }
        catch (...)
        {
            fprintf(stderr, "\n========== UNKNOWN EXCEPTION IN BEAM_SEARCH ==========\n");
            fprintf(stderr, "Exception Type: Unknown (not std::exception)\n");
            fprintf(stderr, "\n--- PLAYER INFO ---\n");
            fprintf(stderr, "My ID: %d\n", map_properties.my_id);
            fprintf(stderr, "Opponent ID: %d\n", map_properties.opp_id);
            fprintf(stderr, "My Alive Snakes: %d\n", get_player_alive_snake_count(state, map_properties.my_id));
            fprintf(stderr, "Opponent Alive Snakes: %d\n", get_player_alive_snake_count(state, map_properties.opp_id));
            fprintf(stderr, "Total Alive Snakes: %d\n", get_alive_snake_count(state));
            fprintf(stderr, "\n--- BEAM SEARCH STATS ---\n");
            fprintf(stderr, "Visited States: %d\n", beam_search_visited_states_count);
            fprintf(stderr, "Beam Search Depth Reached: %d\n", beam_search_depth);
            fprintf(stderr, "========================================================\n\n");
            exit(1);
        }
        // print_moveset(best_moveset);

        if (get_moveset_move_count(best_moveset) == 0)
        {
            fprintf(stderr, "No best moveset found this turn !!!!!\n");
            exit(0);
        }

        fprintf(stderr, "\nMax depth reached: %d\n", beam_search_depth);

        fprintf(stderr, "\nStates visited this turn: %d\n", beam_search_visited_states_count);
        fprintf(stderr, "Avg visited states: %d\n", (int)beam_search_average_states_visited);

        fprintf(stderr, "\nchoose_best_player_moveset() - time : %d ys\n", choose_best_player_moveset_time);
        fprintf(stderr, "choose_best_player_moveset() - count : %d\n", choose_best_player_moveset_count);
        fprintf(stderr, "choose_best_player_moveset() - t/call: %f ys\n", choose_best_player_moveset_time / (float)choose_best_player_moveset_count);

        fprintf(stderr, "\nchoose_best_snake_moves() - time : %d ys\n", choose_best_snake_moves_time);
        fprintf(stderr, "choose_best_snake_moves() - count : %d\n", choose_best_snake_moves_count);
        fprintf(stderr, "choose_best_snake_moves() - t/call: %f ys\n", choose_best_snake_moves_time / (float)choose_best_snake_moves_count);

        fprintf(stderr, "\ngenerate_player_movesets() - time : %d ys\n", generate_player_movesets_time);
        fprintf(stderr, "generate_player_movesets() - count : %d\n", generate_player_movesets_count);
        fprintf(stderr, "generate_player_movesets() - t/call: %f ys\n", generate_player_movesets_time / (float)generate_player_movesets_count);

        fprintf(stderr, "\nrevert_last_move() - time : %d ys\n", revert_last_move_time);
        fprintf(stderr, "revert_last_move() - count : %d\n", revert_last_move_count);
        fprintf(stderr, "revert_last_move() - t/call: %f ys\n", revert_last_move_time / (float)revert_last_move_count);

        fprintf(stderr, "\napply_moveset() - time : %d ys\n", apply_moveset_time);
        fprintf(stderr, "apply_moveset() - count : %d\n", apply_moveset_count);
        fprintf(stderr, "apply_moveset() - t/call: %f ys\n", apply_moveset_time / (float)apply_moveset_count);

        fprintf(stderr, "\napply_gravity() - time : %d ys\n", apply_gravity_time);
        fprintf(stderr, "apply_gravity() - count : %d\n", apply_gravity_count);
        fprintf(stderr, "apply_gravity() - t/call: %f ys\n", apply_gravity_time / (float)apply_gravity_count);

        // fprintf(stderr, "\nbfs_iterative() - time : %d ys\n", bfs_iterative_time);
        // fprintf(stderr, "bfs_iterative() - count : %d\n", bfs_iterative_count);
        // fprintf(stderr, "bfs_iterative() - t/call: %f ys\n", bfs_iterative_time / (float)bfs_iterative_count);

        // fprintf(stderr, "\nbfs_recursive() - time : %d ys\n", bfs_recursive_time);
        // fprintf(stderr, "bfs_recursive() - count : %d\n", bfs_recursive_count);
        // fprintf(stderr, "bfs_recursive() - t/call: %f ys\n", bfs_recursive_time / (float)bfs_recursive_count);

        fprintf(stderr, "\nevaluate_state() - time : %d ys\n", evaluate_state_time);
        fprintf(stderr, "evaluate_state() - count : %d\n", evaluate_state_count);
        fprintf(stderr, "evaluate_state() - t/call: %f ys\n", evaluate_state_time / (float)evaluate_state_count);

        // fprintf(stderr, "\nmove_candidates_in_beam_depth_queue() - time : %d ys\n", move_candidates_in_beam_depth_queue_time);
        // fprintf(stderr, "move_candidates_in_beam_depth_queue() - count : %d\n", move_candidates_in_beam_depth_queue_count);
        // fprintf(stderr, "move_candidates_in_beam_depth_queue() - t/call: %f ys\n", move_candidates_in_beam_depth_queue_time / (float)move_candidates_in_beam_depth_queue_count);

        fprintf(stderr, "\nconsider_state_to_be_candidate() - time : %d ys\n", consider_state_to_be_candidate_time);
        fprintf(stderr, "consider_state_to_be_candidate() - count : %d\n", consider_state_to_be_candidate_count);
        fprintf(stderr, "consider_state_to_be_candidate() - t/call: %f ys\n", consider_state_to_be_candidate_time / (float)consider_state_to_be_candidate_count);

        fprintf(stderr, "\nfind_candidates_among_state_children() - time : %d ys\n", find_candidates_among_state_children_time);
        fprintf(stderr, "find_candidates_among_state_children() - count : %d\n", find_candidates_among_state_children_count);
        fprintf(stderr, "find_candidates_among_state_children() - t/call: %f ys\n", find_candidates_among_state_children_time / (float)find_candidates_among_state_children_count);

        print_marks(state, best_moveset);

        for (int i = 0; i < get_moveset_move_count(best_moveset); i++)
        {
            Move &move = get_moveset_move(best_moveset, i);
            Pos dir = get_move_dst_pos(move);

            int snake_id = get_move_snake_id(move);
            Snake &snake = get_snake(state, snake_id);
            Pos snake_head = get_snake_head_pos(snake);

            int dir_offset = dir - snake_head;

            // fprintf(stderr, "Chosen move for snake %d: %d %d\n", get_move_snake_id(move), get_map_x(get_move_dst_pos(move)), get_map_y(get_move_dst_pos(move)));

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

        // Print microseconds
        auto end_turn_chrono = chrono::high_resolution_clock::now();
        int elapsed_time = chrono::duration_cast<chrono::microseconds>(end_turn_chrono - start_turn_chrono).count();
        fprintf(stderr, "Time elapsed after response: %d ys\n", elapsed_time);
        turn++;
    }

    return 0;
}
