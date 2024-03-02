#!/bin/bash

# Assuming both scripts are in the same folder
python_script="./modelLive.py"
second_python_script="./reTrain.py"

# Run the first Python script
# 'ETHBUSD 1m 0'

while true; do
    # Wait for 5 minutes
    echo 'Re Training BTC'

    python3 "$second_python_script" "$2" "$3"
    sleep 1500  # 300 seconds = 5 minutes

    echo  'Re Training ETH'
    sleep 1500
    python3 "$second_python_script" "$1" "$3"
    echo 

    # Run the second Python script
    # 'ETHBUSD' '1m'
done
