#!/bin/bash

echo -e "Downloading\n"
python3 ./src/data/download_data.py
echo -e "Downloads complete\n"
echo -e "Unzipping\n"
source ./src/data/unzip.sh
echo -e "Complete\n"
