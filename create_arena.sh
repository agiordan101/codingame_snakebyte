#!/bin/bash
cg-colosseum env create myarena18 --preset winter2026

# cg-colosseum submit --env myarena18 --name v5.10 ./src/snakebyte_v5.10.cpp
# cg-colosseum submit --env myarena18 --name v9.1 ./src/snakebyte_v9.1.cpp
# cg-colosseum submit --env myarena18 --name v9.2 ./src/snakebyte_v9.2.cpp
# cg-colosseum submit --env myarena18 --name v9.3 ./src/snakebyte_v9.3.cpp
# cg-colosseum submit --env myarena18 --name v9.4 ./src/snakebyte_v9.4.cpp
# cg-colosseum submit --env myarena18 --name v9.5 ./src/snakebyte_v9.5.cpp
# cg-colosseum submit --env myarena18 --name v9.6 ./src/snakebyte_v9.6.cpp
# cg-colosseum submit --env myarena18 --name v9.7 ./src/snakebyte_v9.7.cpp

cg-colosseum submit --env myarena18 --name rc-1 ./src/snakebyte_rc-1.cpp
cg-colosseum submit --env myarena18 --name rc-1.1 ./src/snakebyte_rc-1.1.cpp
cg-colosseum submit --env myarena18 --name rc-1.2 ./src/snakebyte_rc-1.2.cpp
cg-colosseum submit --env myarena18 --name rc-1.3 ./src/snakebyte_rc-1.3.cpp
cg-colosseum submit --env myarena18 --name rc-1.4 ./src/snakebyte_rc-1.4.cpp
cg-colosseum submit --env myarena18 --name rc-1.5 ./src/snakebyte_rc-1.5.cpp
cg-colosseum submit --env myarena18 --name rc-1.6 ./src/snakebyte_rc-1.6.cpp

cg-colosseum arena myarena18 -t 6