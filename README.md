# codingame_snakebyte
Winter Challenge 2026

# TODO

## Strategies

- Create heuristics mixing Score + 1 snake BFS dist will help snakes to not crash on walls when no paths exist
- Consider actual tails as walkable

### Heuristics


### Turns simulation Me/Opponnet


## Implementation

- Remove initial cells from State, and just save a initial_state. memcpy each turn the State
- Mettre un padding de 3 minimum sur la gauche pour que les snakes puissent se déplacer en dehors de la map
    - S'assurer que les "out of bound" s'applique sur les dimensions max des arrays et pas la map
