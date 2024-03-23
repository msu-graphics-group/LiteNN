# NeuralCore

## How to build (standalone)
1) download and build kernel slicer
2) bash slicer_execute.sh absolute/path/to/slicer/folder absolute/path/to/slicer/executable
3) cmake CMakeLists.txt -DMODULE_VULKAN=ON -DSLICER_DIR=/absolute/path/to/slicer/folder
4) make -j8

## Run

./nn_test - it will run perform_tests function with a bunch of different tests