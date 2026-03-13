// Version 2.1.1

// Algorithms :
// v1 - Each snakes go to closest Energy cell using BFS
//  v1.1 - Apply Action + Gravity first, and then use BFS value
//  v1.2 - Always select a valid action, even if no energy cell is found
//  v1.3 - Add padding around map so all simulations work outside the map (BFS, etc...)
// v2 - Evaluate all possible move combinaisons at current depth, using physics simulation (gravity + collisions) and an improved fitness function (Score diff + sum (Snake/Energy distances))
//  v2.1 - Find best move set for the opponent first, then for the player in consequence
//   v2.1.1 - Apply turn moveset in each simulation
//          - Fix collisions simulation
//          - Fix cells application
//          - Correctly remove snake id from player alive list
//          - Snake weren't checking collisions with their own body
// v3 - Add an opponent move choice before (The best base don heuristic)
// v4 - Beam search : Strategy explained in README.md

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

constexpr int MAX_MAP_WIDTH = 45;
constexpr int MAX_MAP_HEIGHT = 30;

constexpr int MAP_PADDING = 3;
constexpr int MAX_WIDTH = MAX_MAP_WIDTH + 2 * MAP_PADDING;
constexpr int MAX_HEIGHT = MAX_MAP_HEIGHT + 2 * MAP_PADDING;
constexpr int MAX_CELL_COUNT = MAX_WIDTH * MAX_HEIGHT;

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

constexpr int MAX_SNAKE_SIZE = 50;

struct Snake
{
    int id;
    int player_id;
    Pos body_pos[MAX_SNAKE_SIZE];
    int body_length;
    bool is_dying;
};

int get_snake_id(Snake &snake) { return snake.id; }
int get_snake_player_id(Snake &snake) { return snake.player_id; }
Pos get_snake_body_pos(Snake &snake, int index) { return snake.body_pos[index]; }
Pos get_snake_head_pos(Snake &snake) { return get_snake_body_pos(snake, 0); }
int get_snake_body_length(Snake &snake) { return snake.body_length; }

void set_snake_body_pos(Snake &snake, int index, Pos pos) { snake.body_pos[index] = pos; }
void set_snake_body_length(Snake &snake, int length) { snake.body_length = length; }
void set_snake_dying(Snake &snake) { snake.is_dying = true; }

bool is_snake_dying(Snake &snake) { return snake.is_dying; }

void reset_snake_state(Snake &snake)
{
    snake.is_dying = false;
}

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
    snake.is_dying = false;
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

/* --- STATE --- */

constexpr int MAX_SNAKE_COUNT = 8;
constexpr int MAX_PLAYER_SNAKE_COUNT = MAX_SNAKE_COUNT / 2;
constexpr int MIN_SNAKE_ID = 1;

constexpr int constexpr_pow(int base, int exp) { return exp == 0 ? 1 : base * constexpr_pow(base, exp - 1); }
constexpr int MAX_ACTION_COUNT = constexpr_pow(3, MAX_SNAKE_COUNT);

constexpr int CELL_EMPTY = 8;
constexpr int CELL_PLATFORM = 9;
constexpr int CELL_ENERGY = 10;

struct State
{
    int game_points;           // Points difference between my id and opponent id
    int cells[MAX_CELL_COUNT]; // 0-7: snake_id, 8: CELL_EMPTY, 9: CELL_PLATFORM, 10: CELL_ENERGY

    Snake snakes[MAX_SNAKE_COUNT]; // 0-7: snakes

    int player_alive_snake_count[2];
    int player_alive_snake_ids[2][MAX_PLAYER_SNAKE_COUNT];

    int alive_snake_count;
    int alive_snake_ids[MAX_SNAKE_COUNT];

    Pos energies[MAX_CELL_COUNT];
    int energy_count;

    // Pos snake_bodies[MAX_CELL_COUNT];
    // int snake_bodies_count;

    // Action actions[MAX_ACTION_COUNT]; // All available actions for one turn
    // int action_count;                 // Number of actions available

    // Action selected_actions[MAX_SNAKE_COUNT];
    // int selected_actions_count;
    // int snake_having_played[MIN_SNAKE_ID + MAX_SNAKE_COUNT]; // 0: not played, 1: played
};

int get_game_points(State &state) { return state.game_points; }
int get_cell(State &state, Pos pos) { return state.cells[pos]; }
Snake &get_snake(State &state, int snake_id) { return state.snakes[snake_id]; }
int get_player_alive_snake_count(State &state, int player_id) { return state.player_alive_snake_count[player_id]; }
int get_player_alive_snake_id(State &state, int player_id, int index) { return state.player_alive_snake_ids[player_id][index]; }
int get_alive_snake_count(State &state) { return state.alive_snake_count; }
int get_alive_snake_id(State &state, int index) { return state.alive_snake_ids[index]; }

void set_cell(State &state, Pos pos, int value) { state.cells[pos] = value; }
void set_energy(State &state, int index, Pos pos) { state.energies[index] = pos; }

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
    int snake_index = -1;
    for (int i = 0; i < state.alive_snake_count; i++)
    {
        if (state.alive_snake_ids[i] == snake_id)
        {
            // Remove the snake id from alive_snake_ids by shifting the rest of the array
            memmove(&state.alive_snake_ids[i], &state.alive_snake_ids[i + 1], sizeof(int) * (state.alive_snake_count - i - 1));
            state.alive_snake_count--;
        }
    }

    // Remove the snake from the player's alive snake ids
    for (int i = 0; i < state.player_alive_snake_count[player_id]; i++)
    {
        if (state.player_alive_snake_ids[player_id][i] == snake_id)
        {
            // Remove the snake id from alive_snake_ids by shifting the rest of the array
            memmove(&state.player_alive_snake_ids[player_id][i], &state.player_alive_snake_ids[player_id][i + 1], sizeof(int) * (state.player_alive_snake_count[player_id] - i - 1));
            state.player_alive_snake_count[player_id]--;
            return;
        }
    }
}

// void reset_snake_bodies(State &state)
// {
//     state.snake_bodies_count = 0;
// }
// void add_snake_body(State &state, Snake &snake)
// {
//     int snake_body_length = get_snake_body_length(snake);

//     memcpy(&state.snake_bodies[state.snake_bodies_count], &snake.body_pos, sizeof(Pos) * snake_body_length);
//     state.snake_bodies_count += snake_body_length;
// }

void update_game_points(State &state)
{
    // Positive game points is better for my player
    int my_total_length = 0;
    int opp_total_length = 0;

    // Sum lengths of alive snakes for my player
    int my_snake_count = get_player_alive_snake_count(state, map_properties.my_id);
    for (int i = 0; i < my_snake_count; i++)
    {
        int snake_id = get_player_alive_snake_id(state, map_properties.my_id, i);
        Snake &snake = get_snake(state, snake_id);
        my_total_length += get_snake_body_length(snake);
    }

    // Sum lengths of alive snakes for opponent
    int opp_snake_count = get_player_alive_snake_count(state, map_properties.opp_id);
    for (int i = 0; i < opp_snake_count; i++)
    {
        int snake_id = get_player_alive_snake_id(state, map_properties.opp_id, i);
        Snake &snake = get_snake(state, snake_id);
        opp_total_length += get_snake_body_length(snake);
    }

    // Calculate difference
    state.game_points = my_total_length - opp_total_length;
}

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

int find_closest_energy_cell(State &state, Pos start_pos, Pos &closest_energy_cell_pos);
void print_cells(State &state, string title = "")
{
    if (title != "")
        fprintf(stderr, "(dPoint=%d) %s\n", get_game_points(state), title.c_str());

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
        fprintf(stderr, "(dPoint=%d) %s\n", get_game_points(state), title.c_str());

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
                int dist = find_closest_energy_cell(state, pos, closest);
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
    for (int i = 0; i < moveset.move_count; i++)
        fprintf(stderr, "S=%d: %d %d | ", moveset.moves[i].snake_id, get_map_x(moveset.moves[i].dst_pos), get_map_y(moveset.moves[i].dst_pos));
}

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

/* --- GAME PHYSICS - GENERATION --- */

bool is_cell_solid(int cell, int snake_id)
{
    return cell != snake_id && cell != CELL_EMPTY;
}

int generate_snake_moves(State &state, Snake &snake, Pos moves[3])
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
        int neighbor = get_cell(state, neighbors[i]);

        // New valid cell if : In map & Not it neck
        if (!neighbor_out_of_bounds[i] && neighbors[i] != get_snake_body_pos(snake, 1))
        {
            moves[move_count++] = neighbors[i];
        }
    }

    if (move_count == 0)
    {
        fprintf(stderr, "Snake head pos: %d %d\n", get_map_x(head_pos), get_map_y(head_pos));
        for (int i = 0; i < 4; i++)
        {
            int neighbor = get_cell(state, neighbors[i]);
        }

        for (int body_idx = 0; body_idx < get_snake_body_length(snake); body_idx++)
        {
            Pos body_pos = snake.body_pos[body_idx];
            fprintf(stderr, "Body %d: %d %d\n", body_idx, get_map_x(body_pos), get_map_y(body_pos));
        }

        // TODO: If no move generated, create the default one: Continue forward
        print_map(state, "No move generated for snake");
    }

    return move_count;
}

int generate_player_movesets(State &state, int player_id, MoveSet movesets[])
{
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
        snake_move_counts[i] = generate_snake_moves(state, snake, snake_moves[i]);
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
    while (true)
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

    return moveset_count;
}

/* --- GAME PHYSICS - APPLICATION --- */

MoveSet merge_movesets(MoveSet &moveset1, MoveSet &moveset2)
{
    MoveSet merged_moveset;
    int moveset1_move_count = get_moveset_move_count(moveset1);
    int moveset2_move_count = get_moveset_move_count(moveset2);

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

void apply_move(State &state, Snake &snake, Move &move)
{
    // fprintf(stderr, "Applying move for snake %d: (%d, %d)\n", get_snake_id(snake), get_map_x(get_move_dst_pos(move)), get_map_y(get_move_dst_pos(move)));

    Pos new_head_pos = get_move_dst_pos(move);
    int cell_at_new_head_pos = get_cell(state, new_head_pos);

    if (cell_at_new_head_pos == CELL_PLATFORM)
    {
        // Decrease snake length
        set_snake_body_length(snake, get_snake_body_length(snake) - 1);

        // Body positions remain the same
    }
    else
    {
        int body_length_to_move = get_snake_body_length(snake) - 1;

        if (cell_at_new_head_pos == CELL_ENERGY)
        {
            set_cell(state, new_head_pos, CELL_EMPTY); // Remove the energy cel (even if two snakes will collide on it)

            // Move the whole body positions, instead of loosing the tail position
            body_length_to_move = get_snake_body_length(snake);

            // Increase snake length
            set_snake_body_length(snake, get_snake_body_length(snake) + 1);
        }

        // Move the body positions (memmove because source and dest overlap)
        memmove(&snake.body_pos[1], &snake.body_pos[0], sizeof(Pos) * body_length_to_move);

        // Assign a new position to the head
        set_snake_body_pos(snake, 0, new_head_pos);
    }
}

void apply_all_moves(State &state, MoveSet &moveset)
{
    for (int i = 0; i < get_moveset_move_count(moveset); i++)
    {
        Move &move = get_moveset_move(moveset, i);
        Snake &snake = get_snake(state, get_move_snake_id(move));

        reset_snake_state(snake);
        apply_move(state, snake, move);
    }
}

void handle_snake_collisions(State &colliding_state, State &resolved_state)
{
    // fprintf(stderr, "handle_snake_collisions() ...\n");

    for (int snake_index = 0; snake_index < get_alive_snake_count(colliding_state); snake_index++)
    {
        // Iter over all alive snakes
        int snake_id = get_alive_snake_id(colliding_state, snake_index);
        Snake &snake = get_snake(colliding_state, snake_id);
        Pos snake_head_pos = get_snake_head_pos(snake);

        for (int snake2_index = 0; snake2_index < get_alive_snake_count(colliding_state); snake2_index++)
        {
            // Iter over all other alive snakes
            int snake2_id = get_alive_snake_id(colliding_state, snake2_index);
            Snake &snake2 = get_snake(colliding_state, snake2_id);

            // Check if snake head is colliding with snake2 body
            for (int body_idx = 0; body_idx < get_snake_body_length(snake2); body_idx++)
            {
                if (snake_index == snake2_index && body_idx == 0)
                    continue; // Don't check collision with its own head

                Pos snake2_body_pos = get_snake_body_pos(snake2, body_idx);
                if (snake_head_pos == snake2_body_pos)
                {
                    // fprintf(stderr, "Snake %d is colliding with snake %d\n", snake_id, snake2_id);
                    Snake &future_snake = get_snake(resolved_state, snake_id);

                    if (get_snake_body_length(future_snake) - 1 < 3)
                    {
                        // fprintf(stderr, "Snake %d is dying\n", snake_id);
                        // We can't remove it from alive snakes list now, so mark it dying for later
                        set_snake_dying(future_snake);
                    }
                    else
                    {
                        remove_snake_head(future_snake);
                        // print_snake(future_snake);
                    }
                }
            }
        }
    }
}

void kill_dying_snakes(State &state)
{
    for (int i = 0; i < get_alive_snake_count(state); i++)
    {
        int snake_id = get_alive_snake_id(state, i);
        Snake &snake = get_snake(state, snake_id);

        if (is_snake_dying(snake))
            remove_snake_from_alive_snake_ids(state, snake_id, get_snake_player_id(snake));
    }
}

void remove_snake_in_cells_from_their_old_positions(State &previous_state, State &state)
{
    // State cells must be the same as previous state cells
    // But snake spositions are differents, so we use the previous state snakes to remove snkaes from actual state cells
    for (int i = 0; i < get_alive_snake_count(previous_state); i++)
    {
        int snake_id = get_alive_snake_id(previous_state, i);
        Snake &snake = get_snake(previous_state, snake_id);

        for (int j = 0; j < get_snake_body_length(snake); j++)
        {
            Pos body_pos = get_snake_body_pos(snake, j);
            set_cell(state, body_pos, CELL_EMPTY);
        }
    }
}

void set_snake_in_cells(State &state)
{
    for (int i = 0; i < get_alive_snake_count(state); i++)
    {
        int snake_id = get_alive_snake_id(state, i);
        Snake &snake = get_snake(state, snake_id);

        for (int j = 0; j < get_snake_body_length(snake); j++)
        {
            Pos body_pos = get_snake_body_pos(snake, j);
            set_cell(state, body_pos, snake_id);
        }
    }
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
        // Verify if snake has one solid cell under it
        for (int i = 0; i < snake_body_length; i++)
        {
            Pos pos = get_snake_body_pos(snake, i);
            Pos pos_below = get_south_pos(pos);
            int cell_below = get_cell(state, pos_below);

            // fprintf(stderr, "Snake %d: Is cell %d solid? (Pos=%d)\n", snake_id, cell_below, pos);
            if (is_cell_solid(cell_below, snake_id))
                return gravity_applied;

            new_snake_positions[i] = pos_below;
        }

        // TODO: Kill the snake if it fall under the map

        gravity_applied = true;

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

void apply_gravity(State &state)
{
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
        if (++snake_index == get_alive_snake_count(state))
            snake_index = 0;
    }
}

void apply_moveset(State &previous_state, State &state, MoveSet &moveset)
{
    // fprintf(stderr, "Applying moveset ...\n");
    apply_all_moves(state, moveset);

    // To not corrupt data we need to keep the collisions intact in colliding_state, and apply the collisions in another state
    State colliding_state;
    memcpy(&colliding_state, &state, sizeof(State));

    // Reset snake positions if we find a collision & Find dying snakes
    handle_snake_collisions(colliding_state, state);

    // Update cells with new snake positions (Removing dying snkaes from cells)
    remove_snake_in_cells_from_their_old_positions(previous_state, state);
    kill_dying_snakes(state);
    set_snake_in_cells(state);

    apply_gravity(state);
    
    update_game_points(state);

    // if (moveset.moves[0].dst_pos == get_pos_from_map_coord(9, 6))
    // {
    //     fprintf(stderr, "Game points != 0 after apply_gravity: %d\n", state.game_points);
    //     print_map(state, "State after apply_gravity:");
    // }
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

    return find_closest_energy_cell_recursive(state, bfs_queue, visited, closest_energy_cell_pos);
}

/* --- DECISION MAKING --- */

float evaluate_state(State &state, int player_id)
{
    int dist_sum = 0;
    for (int i = 0; i < get_player_alive_snake_count(state, player_id); i++)
    {
        int snake_id = get_player_alive_snake_id(state, player_id, i);
        Snake &snake = get_snake(state, snake_id);

        Pos closest;
        int dist = find_closest_energy_cell(state, get_snake_head_pos(snake), closest);
        if (dist == -1)
            dist_sum += MAX_MAP_WIDTH + MAX_MAP_HEIGHT; // If no energy cell found, consider it very far
        else
            dist_sum += dist;
    }

    if (player_id == map_properties.my_id)
        return get_game_points(state) + 1.0 / dist_sum;

    // Game points is positive if it's good for my player, negative if it's good for opponent, so we invert it for opponent evaluation
    return -get_game_points(state) + 1.0 / dist_sum;
}

MoveSet choose_player_move_set(State &state, int player_id, MoveSet &previous_player_moveset)
{
    State next_state;
    fprintf(stderr, "Choosing move set for player %d ...\n", player_id);

    MoveSet movesets[MAX_PLAYER_MOVE_SETS];
    int moveset_count = generate_player_movesets(state, player_id, movesets);

    MoveSet best_moveset;
    float best_evaluation = -100000;
    for (int i = 0; i < moveset_count; i++)
    {
        memcpy(&next_state, &state, sizeof(State));

        MoveSet turn_moveset = merge_movesets(previous_player_moveset, movesets[i]);
        apply_moveset(state, next_state, turn_moveset);

        float evaluation = evaluate_state(next_state, player_id);
        if (best_evaluation < evaluation)
        {
            fprintf(stderr, "New best moveset found for player %d with evaluation %f: ", player_id, evaluation);
            print_moveset(turn_moveset);
            fprintf(stderr, "\n");

            best_evaluation = evaluation;
            best_moveset = movesets[i];
        }
    }

    return best_moveset;
}

void print_marks(State &state, MoveSet best_moveset)
{
    for (int i = 0; i < get_moveset_move_count(best_moveset); i++)
    {
        Move &move = get_moveset_move(best_moveset, i);
        int snake_id = get_move_snake_id(move);
        Snake &snake = get_snake(state, snake_id);

        Pos closest;
        int dist = find_closest_energy_cell(state, get_snake_head_pos(snake), closest);

        cout << "MARK " << get_map_x(closest) << " " << get_map_y(closest) << ";";
    }
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

Pos parse_pos_from_segment(string segment)
{
    size_t commaPos = segment.find(',');
    int x = stoi(segment.substr(0, commaPos));
    int y = stoi(segment.substr(commaPos + 1));

    // Convert map coordinates to internal data structure coordinates
    return get_pos_from_map_coord(x, y);
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
    cin >> state.energy_count;
    for (int i = 0; i < state.energy_count; i++)
    {
        int x, y;
        cin >> x >> y;

        // Convert map coordinates to internal data structure coordinates
        Pos pos = get_pos_from_map_coord(x, y);

        set_energy(state, i, pos);
        set_cell(state, state.energies[i], CELL_ENERGY);
    }

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

int main()
{
    State initial_state;
    bzero(&initial_state, sizeof(initial_state));
    parse_initial_inputs(initial_state);

    // Save initial state for resetting cells each turn
    State state;
    while (true)
    {
        // Reset state to initial state before parsing fresh dynamic data
        memcpy(&state, &initial_state, sizeof(state));
        parse_turn_inputs(state);
        update_game_points(state);
        // print_cells(state, "Turn beginning");
        // print_map(state, "Turn beginning");
        // print_map_bfs_distances(state);

        MoveSet turn_beginning_movesets = {};
        set_moveset_move_count(turn_beginning_movesets, 0);

        MoveSet best_opponent_moveset = choose_player_move_set(state, map_properties.opp_id, turn_beginning_movesets);
        print_moveset(best_opponent_moveset);

        MoveSet best_moveset = choose_player_move_set(state, map_properties.my_id, best_opponent_moveset);

        print_marks(state, best_moveset);

        for (int i = 0; i < get_moveset_move_count(best_moveset); i++)
        {
            Move &move = get_moveset_move(best_moveset, i);
            Pos dir = get_move_dst_pos(move);

            int snake_id = get_move_snake_id(move);
            Snake &snake = get_snake(state, snake_id);
            Pos snake_head = get_snake_head_pos(snake);

            int dir_offset = dir - snake_head;

            fprintf(stderr, "Chosen move for snake %d: %d %d\n", get_move_snake_id(move), get_map_x(get_move_dst_pos(move)), get_map_y(get_move_dst_pos(move)));

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
