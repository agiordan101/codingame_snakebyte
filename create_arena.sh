#!/bin/bash
cg-colosseum env create myarena11 --preset winter2026

# cg-colosseum submit --env myarena11 --name v5.4 ./src/snakebyte_v5.4.cpp
# cg-colosseum submit --env myarena11 --name v5.5 ./src/snakebyte_v5.5.cpp
# cg-colosseum submit --env myarena11 --name v5.6 ./src/snakebyte_v5.6.cpp
# cg-colosseum submit --env myarena11 --name v5.7 ./src/snakebyte_v5.7.cpp
# cg-colosseum submit --env myarena11 --name v5.8 ./src/snakebyte_v5.8.cpp
# cg-colosseum submit --env myarena11 --name v5.9 ./src/snakebyte_v5.9.cpp
# cg-colosseum submit --env myarena11 --name v5.9.1 ./src/snakebyte_v5.9.1.cpp
cg-colosseum submit --env myarena11 --name v5.10 ./src/snakebyte_v5.10.cpp
# cg-colosseum submit --env myarena11 --name v7 ./src/snakebyte_v7.cpp
# cg-colosseum submit --env myarena11 --name v7.1 ./src/snakebyte_v7.1.cpp
# cg-colosseum submit --env myarena11 --name v7.2 ./src/snakebyte_v7.2.cpp
# cg-colosseum submit --env myarena11 --name v8 ./src/snakebyte_v8.cpp
# cg-colosseum submit --env myarena11 --name v8.1 ./src/snakebyte_v8.1.cpp
cg-colosseum submit --env myarena11 --name v9 ./src/snakebyte_v9.cpp
cg-colosseum submit --env myarena11 --name v9.1 ./src/snakebyte_v9.1.cpp
cg-colosseum submit --env myarena11 --name v9.2 ./src/snakebyte_v9.2.cpp

cg-colosseum arena myarena11 -t 6