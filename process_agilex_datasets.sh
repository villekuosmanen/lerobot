#!/bin/bash
# process_datasets.sh

# Read datasets.txt line by line
while IFS= read -r dataset; do
    echo "Processing dataset: $dataset"
    
    # Run the Python command for this dataset
    python lerobot/scripts/push_dataset_to_hub.py \
        --raw-dir "/media/ville/T9/rdt-data/rdt_data/$dataset" \
        --raw-format aloha_hdf5 \
        --repo-id "villekuosmanen/agilex_$dataset"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully processed $dataset"
    else
        echo "Error processing $dataset"
    fi
    
    echo "----------------------------------------"
done < "datasets.txt"