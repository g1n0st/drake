#!/bin/sh

TARGET_FILE="/home/changyu/drake/dual_arm_folding.html"
rm $TARGET_FILE

while [ ! -f "$TARGET_FILE" ]; do
    cd examples/multibody/deformable
    bazel run dual_arm_folding --config omp -- -write_files
    cd ../../../
done

echo "$TARGET_FILE detected."
