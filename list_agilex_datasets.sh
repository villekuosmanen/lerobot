#!/bin/bash
# list_datasets.sh

# Navigate to the directory
cd "/media/ville/T9/rdt-data/rdt_data"

# List all directories and save to datasets.txt
ls -d */ | sed 's#/##' > datasets.txt

