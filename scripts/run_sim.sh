#!/bin/bash

generate_plates='false'
spawn_pedestrians='false'

print_usage() {
  echo "Usage:"
  echo "-g to generate new license plates"
}

while getopts 'vpgl' flag; do
  case "${flag}" in
    g) generate_plates='true' ;;
    p) spawn_pedestrians='true' ;;
    *) print_usage
       exit 1 ;;
  esac
done

if $generate_plates = 'true'
then
	# DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
	RELATIVE_PATH="plate_generator.py"
	FULL_PATH="./$RELATIVE_PATH"
	python $FULL_PATH
fi

ln -sfn unlabelled ../media/materials/textures/license_plates

roslaunch competition_2019t2 my_launch.launch spawn_pedestrians:=$spawn_pedestrians