#!/bin/bash

echo "=== Re-encoding dataset with fixed preprocessing ==="
echo "This will:"
echo "1. Delete the old latent dataset"
echo "2. Re-encode with contours downsampled to 16x16 during encoding"
echo "3. Ensure clean binary masks"
echo ""

# Check if we should proceed
read -p "Do you want to proceed? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo "Deleting old dataset..."
rm -rf ../data32

echo "Re-encoding dataset..."
./encode_dataset.sh

echo ""
echo "=== Re-encoding complete! ==="
echo "You can now restart training with:"
echo "./train_stable.sh"
echo ""
echo "The contours should now appear as clean binary masks in your visualizations." 