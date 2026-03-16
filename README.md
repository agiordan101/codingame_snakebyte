# codingame_snakebyte
Winter Challenge 2026

## TODO

v4.3 timeout :
    - Each turn ends with "Time limit exceeded during moveset generation.", except when their is a timeout ! With 65000/69000 ms. Logs look fine, and satte visited too.
    - Error: No energy cell found in lookup table for snake head at (23, 1) with 2 energy left: 7644973333309270000 , player "Qoo". Energy cells are inacessible !!!!!!
    - [ERROR] Caught signal: SIGSEGV (Segmentation Fault) (11)
        Un snake est probablement sortis de la map avec padding !!
        Le beam search ne s'est même pas lancé ! Error dans le parsing ?

After publishing v4: Still rare crash - Happen on map edges, after an enemy snake fall

Lorsqu'on pert avec plus d'energy, il vaut mieux perdre que se prendre des murs en boucle (plus dexagération de la lose)

timeout solutions :
    remove tous les vector ?
    Replace recursive BFS with iterative BFS (must do)

- Tester d'autres heuristic :
    - Ajouter des bonus en fonction de ce qu'il y a autour du snake :
        - Platform: Bonus weighted par l'idex du PREMIER body qui est sur une platform (Encourage la tête à retrouver des platforms)
        - Snake allié: Pareil que pour les platforms, mais avec un bonus plus faible
        - Snake enemy: Pareil que pour les platforms, mais avec un malus
    - Avoir un BFS avec gravité qui determine si un snake peut ateindre une energy, sinon faire en sorte qu'il se raproche de la queue d'un allié
    - A faire après le beam search, pour correctement évaluer l'amélioration du ratio temps/précision de l'heuristic : Faire un nouveau DFS qui prends en compte la gravité et son body :
        On prends l'état actuel
        On fait bouger que ce snake
        en v1: On considère une seule fois les cases où la tête est passée
        en v1: On shift les pos du body ?
- Collisions :
  Plutot que d'avoir un state previous ou resolved et dying flag on ourrait :
  renvoyer une liste des snake qui collide dans handle_snake_collisions (donc avoir 1 seul state en param)
- apply_moveset: Plutot que d'avoir un previous state, on pourrait juste remove_snake_in_cells_from_their_old_positions au tout début de la fonction ?
. Ensuite remove head et kill les snake directement dans la même fonction 

POinters instead of state copy :
    Short-term minimal change: store pointers (or std::unique_ptr<State>) in your candidate list instead of copying whole State. Alternatively, only push the heuristic + MoveSet first_depth_moveset + a compact representation of the state (e.g., differences). The quickest patch is to store std::shared_ptr<State> or std::unique_ptr<State>.

Faster sorting :
    After generating all children for this depth (or after generating all children for a parent), if beam_search_candidates.size() > beam_width, call std::nth_element/std::partial_sort to keep only top beam_width by heuristic, then resize vector.

## Run commands

To compete 2 bots (100 silver games with 4 threads) :

```bash
make && \
cg-colosseum battle \
    --preset winter2026 -d league=2 -n 100 -t 4 \
    --p1 ./bin/snakebyte_v4 \
    --p2 ./bin/snakebyte_v4
```

To analyze a specific game :

```bash
make && \
cg-colosseum replay \
    --preset winter2026 -d league=2 \
    --p1 "./bin/snakebyte_v4" \
    --p2 ./bin/snakebyte_v4 \
    --seed=-7137723507467264000 \
    --view
```

To analyze time consumption (Generate file callgrind.out.*) :

```bash
make && \
cg-colosseum battle \
    --preset winter2026 -d league=2 -n 1 -t 1 \
    --p1 "valgrind --tool=callgrind ./bin/snakebyte_v4" \
    --p2 ./bin/snakebyte_4
```

For debug memory issues (Generate file valgrind.log) :

```bash
make && \
cg-colosseum battle \
    --preset winter2026 -d league=2 -n 1 -t 1 \
    --p1 "valgrind -s --leak-check=full --track-origins=yes --log-file=valgrind.log ./bin/snakebyte_v4" \
    --p2 ./bin/snakebyte_v4
```

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
        Handle la suppression du snke si il meurt
Update les cells
    3 cas : 
        colliding = true: tete change pas | queue disparait
        eating = true: tête avance | queue change pas
        else: tête avance | queue disparait
Apply gravity :
     L'appliquer individuellement en bouclant sur la liste de snakes, incrémenter un compteur à chaque fois qu'un snake bouge pas, jusqu'à ce quil atteigne <n snake>
     Mettre un flag si le snake est posé sur une plateforme, pour ne pas le retester.


### Heuristics

- Create heuristics mixing Score + 1 snake BFS dist will help snakes to not crash on walls when no paths exist
- No need to include physics in fitness function, tree iterations will take care of possible/impossible paths

### Beam search strategy

Their is at most 3^4=81 action combinaisons per player.

Move -> A snake moving to a neighboor position
MoveSet -> A list of Move

- Iter on each depth while 49ms isn't reached :
    - Clear beam_search_candidates
    - For each beam_state in the beam_search_states list (0 < n <= beam_width) :
        - Generate all combinaisons of opponent MoveSet from beam_state
        - For each opponent MoveSet :
            - Reset current_state to beam_state
            - Apply the opponent MoveSet
            - Simulate collisions and gravity
            - Run heuristic
        - Select the best opponent MoveSet
        - Generate all combinaisons of ally MoveSet from beam_state
        - For each ally MoveSet :
            - Reset current_state to beam_state
            - Apply the opponent MoveSet
            - Apply the ally MoveSet
            - Simulate collisions and gravity
            - Run and store heuristic in current_state
            - (For the first turn only: Save MoveSet as first_depth_moveset in current_state)
            - Add a copy of the final current_state in beam_search_candidates
    - Sort all states in beam_search_candidates
    - Move best 'beam_width' states from beam_search_candidates to beam_search_states

## History

# v4.3 (Bugged)

Heuristic : During first turn: Create a lookup table to know for all cells which energies are the closest and their BFS distance (without snakes)

League: Gold (max)
First publication : 190/1839
Last publication: 221/1867

# v4.2 (Worst than v4)

Heuristic : Replace BFS by Manhattan distance

League: Silver (max)
First publication : 230/1800
Last publication: 230/1839

# v4.1 (Worst than v4)

For depth > 1, stop simulating the opponent by choosing among all move combinaisons, but choose best snake move one by one (using previous snake moves).

League: Silver (max)
First publication : 320/1800
Last publication: 300/1800

# v4

Beam search : Strategy explained below

League: Silver (max)
First publication : 146/1732
Last publication: 210/1867

# v3.1

+ Apply opponent moveset in each simulated moveset
+ Fix collisions simulation
+ Fix cells application
+ Correctly remove snake id from player alive list
+ Snake weren't checking collisions with their own body

League: Silver (max)
First publication : 155/1580
Last publication: 235/1730

# v2.1 = v3 (lot of bugs)

Find opponent moveset with same algorithm before considering mine

League: Silver (max)
First publication : 330/1450
Last publication: 485/1563

# v2

Evaluate all possible move combinaisons at current depth, using physics simulation (gravity + collisions) and an improved fitness function (Score diff + sum (Snake closest energy distance))

League: Silver (max)
First publication : 330/1450
Last publication: 364/1450

# v1.3

+ Add padding around map so all simulations work outside the map (BFS, etc...)

League: Silver (max)
First publication : 192/1206
Last publication: 317/1450

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
