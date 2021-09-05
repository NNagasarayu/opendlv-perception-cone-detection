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
CMAKE_SOURCE_DIR = /home/johan/project/opendlv-perception-cone-detection

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/johan/project/opendlv-perception-cone-detection/build

# Include any dependencies generated for this target.
include CMakeFiles/opendlv-perception-cone-detection.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/opendlv-perception-cone-detection.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/opendlv-perception-cone-detection.dir/flags.make

opendlv-standard-message-set.hpp: ../src/opendlv-costum-message-set.odvd
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/johan/project/opendlv-perception-cone-detection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating opendlv-standard-message-set.hpp"
	cluon-msc --cpp --out=/home/johan/project/opendlv-perception-cone-detection/build/opendlv-standard-message-set.hpp /home/johan/project/opendlv-perception-cone-detection/src/opendlv-costum-message-set.odvd

cluon-complete.hpp: ../src/cluon-complete-v0.0.127.hpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/johan/project/opendlv-perception-cone-detection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Generating cluon-complete.hpp"
	/usr/bin/cmake -E create_symlink /home/johan/project/opendlv-perception-cone-detection/src/cluon-complete-v0.0.127.hpp /home/johan/project/opendlv-perception-cone-detection/build/cluon-complete.hpp

CMakeFiles/opendlv-perception-cone-detection.dir/src/opendlv-perception-cone-detection.cpp.o: CMakeFiles/opendlv-perception-cone-detection.dir/flags.make
CMakeFiles/opendlv-perception-cone-detection.dir/src/opendlv-perception-cone-detection.cpp.o: ../src/opendlv-perception-cone-detection.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/johan/project/opendlv-perception-cone-detection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/opendlv-perception-cone-detection.dir/src/opendlv-perception-cone-detection.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opendlv-perception-cone-detection.dir/src/opendlv-perception-cone-detection.cpp.o -c /home/johan/project/opendlv-perception-cone-detection/src/opendlv-perception-cone-detection.cpp

CMakeFiles/opendlv-perception-cone-detection.dir/src/opendlv-perception-cone-detection.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opendlv-perception-cone-detection.dir/src/opendlv-perception-cone-detection.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/johan/project/opendlv-perception-cone-detection/src/opendlv-perception-cone-detection.cpp > CMakeFiles/opendlv-perception-cone-detection.dir/src/opendlv-perception-cone-detection.cpp.i

CMakeFiles/opendlv-perception-cone-detection.dir/src/opendlv-perception-cone-detection.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opendlv-perception-cone-detection.dir/src/opendlv-perception-cone-detection.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/johan/project/opendlv-perception-cone-detection/src/opendlv-perception-cone-detection.cpp -o CMakeFiles/opendlv-perception-cone-detection.dir/src/opendlv-perception-cone-detection.cpp.s

# Object files for target opendlv-perception-cone-detection
opendlv__perception__cone__detection_OBJECTS = \
"CMakeFiles/opendlv-perception-cone-detection.dir/src/opendlv-perception-cone-detection.cpp.o"

# External object files for target opendlv-perception-cone-detection
opendlv__perception__cone__detection_EXTERNAL_OBJECTS =

opendlv-perception-cone-detection: CMakeFiles/opendlv-perception-cone-detection.dir/src/opendlv-perception-cone-detection.cpp.o
opendlv-perception-cone-detection: CMakeFiles/opendlv-perception-cone-detection.dir/build.make
opendlv-perception-cone-detection: /usr/lib/x86_64-linux-gnu/librt.a
opendlv-perception-cone-detection: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.2.0
opendlv-perception-cone-detection: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.2.0
opendlv-perception-cone-detection: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.2.0
opendlv-perception-cone-detection: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.2.0
opendlv-perception-cone-detection: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.2.0
opendlv-perception-cone-detection: CMakeFiles/opendlv-perception-cone-detection.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/johan/project/opendlv-perception-cone-detection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable opendlv-perception-cone-detection"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opendlv-perception-cone-detection.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/opendlv-perception-cone-detection.dir/build: opendlv-perception-cone-detection

.PHONY : CMakeFiles/opendlv-perception-cone-detection.dir/build

CMakeFiles/opendlv-perception-cone-detection.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/opendlv-perception-cone-detection.dir/cmake_clean.cmake
.PHONY : CMakeFiles/opendlv-perception-cone-detection.dir/clean

CMakeFiles/opendlv-perception-cone-detection.dir/depend: opendlv-standard-message-set.hpp
CMakeFiles/opendlv-perception-cone-detection.dir/depend: cluon-complete.hpp
	cd /home/johan/project/opendlv-perception-cone-detection/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/johan/project/opendlv-perception-cone-detection /home/johan/project/opendlv-perception-cone-detection /home/johan/project/opendlv-perception-cone-detection/build /home/johan/project/opendlv-perception-cone-detection/build /home/johan/project/opendlv-perception-cone-detection/build/CMakeFiles/opendlv-perception-cone-detection.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/opendlv-perception-cone-detection.dir/depend

