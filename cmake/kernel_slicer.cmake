
if(NOT DEFINED USE_KSLICER_DIR)
	set(USE_KSLICER_DIR "../kernel_slicer" CACHE STRING "Kernel slicer directory")
endif()

if(NOT DEFINED KERNEL_SLICER) 
	get_filename_component(KERNEL_SLICER
                       	   ${USE_KSLICER_DIR}/cmake-build-release/kslicer
                       	   ABSOLUTE)
	get_filename_component(KERNEL_SLICER_DIR
						   ${USE_KSLICER_DIR}
                       	   ABSOLUTE)
endif()
