#!/bin/bash

echo Downloading
python3 /src/data/download_data.py
echo Downloads complete
echo Unzipping
source /src/data/unzip.sh
echo Unzipping complete\n
