#!/bin/bash

## run different envs
#declare -a envs=("MiniGrid-Empty-6x6-v0" "MiniGrid-Empty-8x8-v0" "MiniGrid-DoorKey-5x5-v0" "MiniGrid-MultiRoom-N2-S4-v0")

#for env in "${envs[@]}"; do
#  echo -e "\nEnvironment $env\n"  
#  python ./gridworld_experiments.py --env $env&
#done
#wait


# run trials on one env
env='MiniGrid-Empty-6x6-v0'
echo $env
for i in {1..8}; do  
  echo -e "\nTrial $i $env\n" 
  python ./gridworld_experiments.py --env $env&
  sleep 1
done
wait