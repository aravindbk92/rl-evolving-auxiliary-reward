#!/bin/bash

## environments variable
declare -a envs=("MiniGrid-Empty-6x6-v0" "MiniGrid-Empty-8x8-v0" "MiniGrid-DoorKey-5x5-v0" "MiniGrid-MultiRoom-N2-S4-v0")

for env in "${envs[@]}"; do
  echo -e "\nEnvironment $env\n"  
  python ./gridworld_experiments.py --env $env&
done
wait