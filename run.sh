#!/bin/bash

echo Downloading
python3 download_data.py
echo Downloads complete
echo Unzipping
source unzip.sh
echo Unzipping complete
