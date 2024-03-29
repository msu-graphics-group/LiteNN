#!/bin/bash
# slicer_execute absolute/path/to/slicer/folder absolute/path/to/slicer/executable
#e.g. bash slicer_execute.sh ~/kernel_slicer/ ~/kernel_slicer/kslicer
start_dir=$PWD
cd $1

$2 $start_dir/tensor_processor_impl.cpp \
-mainClass TensorProcessorImpl \
-stdlibfolder $PWD/TINYSTL \
-pattern ipv \
-reorderLoops YX \
-I$PWD/apps/LiteMath ignore \
-I$PWD/apps/LiteMathAux ignore \
-I$PWD/TINYSTL ignore \
-shaderCC glsl \
-suffix _GPU \
-DKERNEL_SLICER -v 

cd $start_dir
cd shaders_gpu
bash build.sh
cd ..