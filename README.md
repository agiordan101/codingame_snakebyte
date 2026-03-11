# codingame_snakebyte
Winter Challenge 2026

# TODO

## Strategies

- Create heuristics mixing Score + 1 snake BFS dist will help snakes to not crash on walls when no paths exist
- Consider actual tails as walkable

## Implementation

- Remove initial cells from State, and just save a initial_state. memcpy each turn the State
