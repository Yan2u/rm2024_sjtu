# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yan2u/Desktop/rm2024_sjtu/2-231008/points

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yan2u/Desktop/rm2024_sjtu/2-231008/points/build

# Include any dependencies generated for this target.
include CMakeFiles/points.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/points.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/points.dir/flags.make

CMakeFiles/points.dir/main.cpp.o: CMakeFiles/points.dir/flags.make
CMakeFiles/points.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yan2u/Desktop/rm2024_sjtu/2-231008/points/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/points.dir/main.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/points.dir/main.cpp.o -c /home/yan2u/Desktop/rm2024_sjtu/2-231008/points/main.cpp

CMakeFiles/points.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/points.dir/main.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yan2u/Desktop/rm2024_sjtu/2-231008/points/main.cpp > CMakeFiles/points.dir/main.cpp.i

CMakeFiles/points.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/points.dir/main.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yan2u/Desktop/rm2024_sjtu/2-231008/points/main.cpp -o CMakeFiles/points.dir/main.cpp.s

# Object files for target points
points_OBJECTS = \
"CMakeFiles/points.dir/main.cpp.o"

# External object files for target points
points_EXTERNAL_OBJECTS =

points: CMakeFiles/points.dir/main.cpp.o
points: CMakeFiles/points.dir/build.make
points: /usr/local/lib/libopencv_stitching.so.4.4.0
points: /usr/local/lib/libopencv_alphamat.so.4.4.0
points: /usr/local/lib/libopencv_aruco.so.4.4.0
points: /usr/local/lib/libopencv_bgsegm.so.4.4.0
points: /usr/local/lib/libopencv_bioinspired.so.4.4.0
points: /usr/local/lib/libopencv_ccalib.so.4.4.0
points: /usr/local/lib/libopencv_dnn_objdetect.so.4.4.0
points: /usr/local/lib/libopencv_dnn_superres.so.4.4.0
points: /usr/local/lib/libopencv_dpm.so.4.4.0
points: /usr/local/lib/libopencv_face.so.4.4.0
points: /usr/local/lib/libopencv_freetype.so.4.4.0
points: /usr/local/lib/libopencv_fuzzy.so.4.4.0
points: /usr/local/lib/libopencv_hfs.so.4.4.0
points: /usr/local/lib/libopencv_img_hash.so.4.4.0
points: /usr/local/lib/libopencv_intensity_transform.so.4.4.0
points: /usr/local/lib/libopencv_line_descriptor.so.4.4.0
points: /usr/local/lib/libopencv_quality.so.4.4.0
points: /usr/local/lib/libopencv_rapid.so.4.4.0
points: /usr/local/lib/libopencv_reg.so.4.4.0
points: /usr/local/lib/libopencv_rgbd.so.4.4.0
points: /usr/local/lib/libopencv_saliency.so.4.4.0
points: /usr/local/lib/libopencv_sfm.so.4.4.0
points: /usr/local/lib/libopencv_stereo.so.4.4.0
points: /usr/local/lib/libopencv_structured_light.so.4.4.0
points: /usr/local/lib/libopencv_superres.so.4.4.0
points: /usr/local/lib/libopencv_surface_matching.so.4.4.0
points: /usr/local/lib/libopencv_tracking.so.4.4.0
points: /usr/local/lib/libopencv_videostab.so.4.4.0
points: /usr/local/lib/libopencv_xfeatures2d.so.4.4.0
points: /usr/local/lib/libopencv_xobjdetect.so.4.4.0
points: /usr/local/lib/libopencv_xphoto.so.4.4.0
points: /usr/local/lib/libopencv_highgui.so.4.4.0
points: /usr/local/lib/libopencv_shape.so.4.4.0
points: /usr/local/lib/libopencv_datasets.so.4.4.0
points: /usr/local/lib/libopencv_plot.so.4.4.0
points: /usr/local/lib/libopencv_text.so.4.4.0
points: /usr/local/lib/libopencv_dnn.so.4.4.0
points: /usr/local/lib/libopencv_ml.so.4.4.0
points: /usr/local/lib/libopencv_phase_unwrapping.so.4.4.0
points: /usr/local/lib/libopencv_optflow.so.4.4.0
points: /usr/local/lib/libopencv_ximgproc.so.4.4.0
points: /usr/local/lib/libopencv_video.so.4.4.0
points: /usr/local/lib/libopencv_videoio.so.4.4.0
points: /usr/local/lib/libopencv_imgcodecs.so.4.4.0
points: /usr/local/lib/libopencv_objdetect.so.4.4.0
points: /usr/local/lib/libopencv_calib3d.so.4.4.0
points: /usr/local/lib/libopencv_features2d.so.4.4.0
points: /usr/local/lib/libopencv_flann.so.4.4.0
points: /usr/local/lib/libopencv_photo.so.4.4.0
points: /usr/local/lib/libopencv_imgproc.so.4.4.0
points: /usr/local/lib/libopencv_core.so.4.4.0
points: CMakeFiles/points.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yan2u/Desktop/rm2024_sjtu/2-231008/points/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable points"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/points.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/points.dir/build: points

.PHONY : CMakeFiles/points.dir/build

CMakeFiles/points.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/points.dir/cmake_clean.cmake
.PHONY : CMakeFiles/points.dir/clean

CMakeFiles/points.dir/depend:
	cd /home/yan2u/Desktop/rm2024_sjtu/2-231008/points/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yan2u/Desktop/rm2024_sjtu/2-231008/points /home/yan2u/Desktop/rm2024_sjtu/2-231008/points /home/yan2u/Desktop/rm2024_sjtu/2-231008/points/build /home/yan2u/Desktop/rm2024_sjtu/2-231008/points/build /home/yan2u/Desktop/rm2024_sjtu/2-231008/points/build/CMakeFiles/points.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/points.dir/depend

