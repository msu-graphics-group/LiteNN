# LiteNN

A simple neural networks library using Vulkan.

## How to build 
If you wish to use GPU, you need to download and build [Kernel Slicer](https://github.com/Ray-Tracing-Systems/kernel_slicer).

### Standalone
To build with Vulkan capabilities run:

    cmake -DLITENN_ENABLE_VULKAN=ON -DUSE_KSLICER_DIR=/path/to/slicer/directory -B ./build && cd build && make -j8

To build CPU-only version run

    cmake -DLITENN_ENABLE_VULKAN=OFF -B ./build && cd build && make -j8

### As library
Use it as CMake subproject.
Options:
* **LITENN_ENABLE_VULKAN** (ON/OFF) -- whether to enable GPU mode using Kernel Slicer (default: *ON*);
* **USE_KSLICER_DIR** (PATH) -- path to kernel slicer directory (default: *../kernel_slicer*).

## Run tests
Change working directory to build (if building as subproject open build directory of root project):
    
    cd build
    
To run tests use:

    ./litenn_test

or 
    ./path/to/LiteNN/litenn_test