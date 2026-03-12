# codingame_snakebyte
Winter Challenge 2026

# TODO

- Consider actual tails as walkable

## Strategies

### Physics

Quand on bouge :

Pour chaque snake :
    Update le body du Snake
    Faire collisions entre le Snake et la plateforme (Update le flag du snake 'colliding')
    Faire collisions entre le Snake et les energies (Update le flag du snake 'eating')
    Append son body à un tableau représentant les positions de tous les snakes
Pour chaque snake :
    Parcourir larray, si la head pos est trouvé 2 fois alors il y a collision :
        La gérer dans le Snake uniquement (Update le flag du snake 'colliding')
Apply gravity :
     L'appliquer individuellement en bouclant sur la liste de snakes, incrémenter un compteur à chaque fois qu'un snake bouge pas, jusqu'à ce quil atteigne <n snake>
     Mettre un flag si le snake est posé sur une plateforme, pour ne pas le retester.
Update les cells
    3 cas : 
        colliding = true: tete change pas | queue disparait
        eating = true: tête avance | queue change pas
        else: tête avance | queue disparait


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
Begin at position : 192/1206
Ending at position: 300/1333

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
