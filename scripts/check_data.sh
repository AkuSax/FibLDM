#!/bin/bash

echo "Checking contour data in latent dataset..."
python ../scripts/check_contour_data.py --latent_datapath ../data32

echo ""
echo "If contours are not properly binarized (should show only 0.0 and 1.0),"
echo "you need to delete the old dataset and re-encode:"
echo ""
echo "rm -rf ../data32"
echo "./encode_dataset.sh"
echo ""
echo "Then restart training with:"
echo "./train_stable.sh" 