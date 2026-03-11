# codingame_snakebyte
Winter Challenge 2026

# TODO

- Consider actual tails as walkable

## Strategies

### Heuristics

- Create heuristics mixing Score + 1 snake BFS dist will help snakes to not crash on walls when no paths exist
- No need to include physics in fitness function, tree iterations will take care of possible/impossible paths

### Beam search strategy

Their is at most 3^4=81 action combinaisons per player.

Move -> A snake moving in one direction
PlayerMoveSet -> A set of 'player snake count' moves
TurnMoveSet -> A set of 'total snake count' moves representing a turn
ActionSet -> A history of TurnMoveSet representing game turns

- Iter on each depth while 49ms isn't reached :
    - Clear beam_search_candidates
    - For each ActionSet in the beam_search_actionsets list (0 < n <= beam_width) :
        - Reset state to turn beginning
        - Apply the ActionSet
        - Generate all opponent PlayerMoveSet
        - For each opponent PlayerMoveSet :
            - Reset state to the current evaluated ActionSet
            - Apply the PlayerMoveSet
            - Simulate collisions and gravity
            - Run heuristic
        - Select the best opponent PlayerMoveSet
            <!-- - By running a beam search with small width and depthMax=2/3 ? -->
        - Generate all ally PlayerMoveSet
        - For each ally PlayerMoveSet :
            - Reset state to the current evaluated ActionSet
            - Apply the opponent PlayerMoveSet
            - Apply the ally PlayerMoveSet
            - Simulate collisions and gravity
            - Run heuristic
            - Create a TurnMoveSet with the two PlayerMoveSet
            - Create a candidate ActionSet from the current ActionSet and TurnMoveSet, and add it to beam_search_candidates list
    - Keep the best 'beam_width' ActionSet from beam_search_candidates, and move them in beam_search_actionsets

## History

# v1.3

+ Add padding around map so all simulations work outside the map (BFS, collisions, gravity, beam search)

League: Bronze (max)
Position: 192/1206

# v1.2

+ Always select a valid action, even if no energy cell is found

League: Bronze (max)
Position: 290/1206

# v1.1

+ Apply Action + Gravity first, and then use BFS value

League: Bronze (max)
Position: 350/1000

# v1.0

Each snakes go to closest Energy cell using BFS

League: Bronze (max)
Position: 210/800
