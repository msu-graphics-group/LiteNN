#!/bin/bash
# slicer_execute absolute/path/to/slicer/folder absolute/path/to/slicer/executable
start_dir=$PWD
cd $1
$2 $start_dir/tensor_processor_impl.cpp -mainClass TensorProcessorImpl -stdlibfolder $PWD/TINYSTL -pattern ipv -reorderLoops YX -I$PWD/apps/LiteMath ignore -I$PWD/apps/LiteMathAux ignore -I$PWD/TINYSTL ignore -shaderCC glsl -suffix _GPU -DKERNEL_SLICER -v 
cd $start_dir
rm -r ../../shaders_gpu
mkdir ../../shaders_gpu
cd shaders_gpu
bash build.sh
find -name "*.spv" | xargs cp --parents -t ../../../shaders_gpu
cd ..