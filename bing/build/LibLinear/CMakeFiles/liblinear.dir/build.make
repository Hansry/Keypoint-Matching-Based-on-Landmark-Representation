# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home/hansry/landmark_keypoint_matching/bing

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hansry/landmark_keypoint_matching/bing/build

# Include any dependencies generated for this target.
include LibLinear/CMakeFiles/liblinear.dir/depend.make

# Include the progress variables for this target.
include LibLinear/CMakeFiles/liblinear.dir/progress.make

# Include the compile flags for this target's objects.
include LibLinear/CMakeFiles/liblinear.dir/flags.make

LibLinear/CMakeFiles/liblinear.dir/linear.cpp.o: LibLinear/CMakeFiles/liblinear.dir/flags.make
LibLinear/CMakeFiles/liblinear.dir/linear.cpp.o: ../LibLinear/linear.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hansry/landmark_keypoint_matching/bing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object LibLinear/CMakeFiles/liblinear.dir/linear.cpp.o"
	cd /home/hansry/landmark_keypoint_matching/bing/build/LibLinear && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/liblinear.dir/linear.cpp.o -c /home/hansry/landmark_keypoint_matching/bing/LibLinear/linear.cpp

LibLinear/CMakeFiles/liblinear.dir/linear.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/liblinear.dir/linear.cpp.i"
	cd /home/hansry/landmark_keypoint_matching/bing/build/LibLinear && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hansry/landmark_keypoint_matching/bing/LibLinear/linear.cpp > CMakeFiles/liblinear.dir/linear.cpp.i

LibLinear/CMakeFiles/liblinear.dir/linear.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/liblinear.dir/linear.cpp.s"
	cd /home/hansry/landmark_keypoint_matching/bing/build/LibLinear && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hansry/landmark_keypoint_matching/bing/LibLinear/linear.cpp -o CMakeFiles/liblinear.dir/linear.cpp.s

LibLinear/CMakeFiles/liblinear.dir/linear.cpp.o.requires:

.PHONY : LibLinear/CMakeFiles/liblinear.dir/linear.cpp.o.requires

LibLinear/CMakeFiles/liblinear.dir/linear.cpp.o.provides: LibLinear/CMakeFiles/liblinear.dir/linear.cpp.o.requires
	$(MAKE) -f LibLinear/CMakeFiles/liblinear.dir/build.make LibLinear/CMakeFiles/liblinear.dir/linear.cpp.o.provides.build
.PHONY : LibLinear/CMakeFiles/liblinear.dir/linear.cpp.o.provides

LibLinear/CMakeFiles/liblinear.dir/linear.cpp.o.provides.build: LibLinear/CMakeFiles/liblinear.dir/linear.cpp.o


LibLinear/CMakeFiles/liblinear.dir/tron.cpp.o: LibLinear/CMakeFiles/liblinear.dir/flags.make
LibLinear/CMakeFiles/liblinear.dir/tron.cpp.o: ../LibLinear/tron.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hansry/landmark_keypoint_matching/bing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object LibLinear/CMakeFiles/liblinear.dir/tron.cpp.o"
	cd /home/hansry/landmark_keypoint_matching/bing/build/LibLinear && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/liblinear.dir/tron.cpp.o -c /home/hansry/landmark_keypoint_matching/bing/LibLinear/tron.cpp

LibLinear/CMakeFiles/liblinear.dir/tron.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/liblinear.dir/tron.cpp.i"
	cd /home/hansry/landmark_keypoint_matching/bing/build/LibLinear && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hansry/landmark_keypoint_matching/bing/LibLinear/tron.cpp > CMakeFiles/liblinear.dir/tron.cpp.i

LibLinear/CMakeFiles/liblinear.dir/tron.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/liblinear.dir/tron.cpp.s"
	cd /home/hansry/landmark_keypoint_matching/bing/build/LibLinear && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hansry/landmark_keypoint_matching/bing/LibLinear/tron.cpp -o CMakeFiles/liblinear.dir/tron.cpp.s

LibLinear/CMakeFiles/liblinear.dir/tron.cpp.o.requires:

.PHONY : LibLinear/CMakeFiles/liblinear.dir/tron.cpp.o.requires

LibLinear/CMakeFiles/liblinear.dir/tron.cpp.o.provides: LibLinear/CMakeFiles/liblinear.dir/tron.cpp.o.requires
	$(MAKE) -f LibLinear/CMakeFiles/liblinear.dir/build.make LibLinear/CMakeFiles/liblinear.dir/tron.cpp.o.provides.build
.PHONY : LibLinear/CMakeFiles/liblinear.dir/tron.cpp.o.provides

LibLinear/CMakeFiles/liblinear.dir/tron.cpp.o.provides.build: LibLinear/CMakeFiles/liblinear.dir/tron.cpp.o


LibLinear/CMakeFiles/liblinear.dir/blas/daxpy.c.o: LibLinear/CMakeFiles/liblinear.dir/flags.make
LibLinear/CMakeFiles/liblinear.dir/blas/daxpy.c.o: ../LibLinear/blas/daxpy.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hansry/landmark_keypoint_matching/bing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object LibLinear/CMakeFiles/liblinear.dir/blas/daxpy.c.o"
	cd /home/hansry/landmark_keypoint_matching/bing/build/LibLinear && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/liblinear.dir/blas/daxpy.c.o   -c /home/hansry/landmark_keypoint_matching/bing/LibLinear/blas/daxpy.c

LibLinear/CMakeFiles/liblinear.dir/blas/daxpy.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/liblinear.dir/blas/daxpy.c.i"
	cd /home/hansry/landmark_keypoint_matching/bing/build/LibLinear && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hansry/landmark_keypoint_matching/bing/LibLinear/blas/daxpy.c > CMakeFiles/liblinear.dir/blas/daxpy.c.i

LibLinear/CMakeFiles/liblinear.dir/blas/daxpy.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/liblinear.dir/blas/daxpy.c.s"
	cd /home/hansry/landmark_keypoint_matching/bing/build/LibLinear && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hansry/landmark_keypoint_matching/bing/LibLinear/blas/daxpy.c -o CMakeFiles/liblinear.dir/blas/daxpy.c.s

LibLinear/CMakeFiles/liblinear.dir/blas/daxpy.c.o.requires:

.PHONY : LibLinear/CMakeFiles/liblinear.dir/blas/daxpy.c.o.requires

LibLinear/CMakeFiles/liblinear.dir/blas/daxpy.c.o.provides: LibLinear/CMakeFiles/liblinear.dir/blas/daxpy.c.o.requires
	$(MAKE) -f LibLinear/CMakeFiles/liblinear.dir/build.make LibLinear/CMakeFiles/liblinear.dir/blas/daxpy.c.o.provides.build
.PHONY : LibLinear/CMakeFiles/liblinear.dir/blas/daxpy.c.o.provides

LibLinear/CMakeFiles/liblinear.dir/blas/daxpy.c.o.provides.build: LibLinear/CMakeFiles/liblinear.dir/blas/daxpy.c.o


LibLinear/CMakeFiles/liblinear.dir/blas/ddot.c.o: LibLinear/CMakeFiles/liblinear.dir/flags.make
LibLinear/CMakeFiles/liblinear.dir/blas/ddot.c.o: ../LibLinear/blas/ddot.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hansry/landmark_keypoint_matching/bing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object LibLinear/CMakeFiles/liblinear.dir/blas/ddot.c.o"
	cd /home/hansry/landmark_keypoint_matching/bing/build/LibLinear && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/liblinear.dir/blas/ddot.c.o   -c /home/hansry/landmark_keypoint_matching/bing/LibLinear/blas/ddot.c

LibLinear/CMakeFiles/liblinear.dir/blas/ddot.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/liblinear.dir/blas/ddot.c.i"
	cd /home/hansry/landmark_keypoint_matching/bing/build/LibLinear && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hansry/landmark_keypoint_matching/bing/LibLinear/blas/ddot.c > CMakeFiles/liblinear.dir/blas/ddot.c.i

LibLinear/CMakeFiles/liblinear.dir/blas/ddot.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/liblinear.dir/blas/ddot.c.s"
	cd /home/hansry/landmark_keypoint_matching/bing/build/LibLinear && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hansry/landmark_keypoint_matching/bing/LibLinear/blas/ddot.c -o CMakeFiles/liblinear.dir/blas/ddot.c.s

LibLinear/CMakeFiles/liblinear.dir/blas/ddot.c.o.requires:

.PHONY : LibLinear/CMakeFiles/liblinear.dir/blas/ddot.c.o.requires

LibLinear/CMakeFiles/liblinear.dir/blas/ddot.c.o.provides: LibLinear/CMakeFiles/liblinear.dir/blas/ddot.c.o.requires
	$(MAKE) -f LibLinear/CMakeFiles/liblinear.dir/build.make LibLinear/CMakeFiles/liblinear.dir/blas/ddot.c.o.provides.build
.PHONY : LibLinear/CMakeFiles/liblinear.dir/blas/ddot.c.o.provides

LibLinear/CMakeFiles/liblinear.dir/blas/ddot.c.o.provides.build: LibLinear/CMakeFiles/liblinear.dir/blas/ddot.c.o


LibLinear/CMakeFiles/liblinear.dir/blas/dnrm2.c.o: LibLinear/CMakeFiles/liblinear.dir/flags.make
LibLinear/CMakeFiles/liblinear.dir/blas/dnrm2.c.o: ../LibLinear/blas/dnrm2.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hansry/landmark_keypoint_matching/bing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object LibLinear/CMakeFiles/liblinear.dir/blas/dnrm2.c.o"
	cd /home/hansry/landmark_keypoint_matching/bing/build/LibLinear && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/liblinear.dir/blas/dnrm2.c.o   -c /home/hansry/landmark_keypoint_matching/bing/LibLinear/blas/dnrm2.c

LibLinear/CMakeFiles/liblinear.dir/blas/dnrm2.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/liblinear.dir/blas/dnrm2.c.i"
	cd /home/hansry/landmark_keypoint_matching/bing/build/LibLinear && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hansry/landmark_keypoint_matching/bing/LibLinear/blas/dnrm2.c > CMakeFiles/liblinear.dir/blas/dnrm2.c.i

LibLinear/CMakeFiles/liblinear.dir/blas/dnrm2.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/liblinear.dir/blas/dnrm2.c.s"
	cd /home/hansry/landmark_keypoint_matching/bing/build/LibLinear && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hansry/landmark_keypoint_matching/bing/LibLinear/blas/dnrm2.c -o CMakeFiles/liblinear.dir/blas/dnrm2.c.s

LibLinear/CMakeFiles/liblinear.dir/blas/dnrm2.c.o.requires:

.PHONY : LibLinear/CMakeFiles/liblinear.dir/blas/dnrm2.c.o.requires

LibLinear/CMakeFiles/liblinear.dir/blas/dnrm2.c.o.provides: LibLinear/CMakeFiles/liblinear.dir/blas/dnrm2.c.o.requires
	$(MAKE) -f LibLinear/CMakeFiles/liblinear.dir/build.make LibLinear/CMakeFiles/liblinear.dir/blas/dnrm2.c.o.provides.build
.PHONY : LibLinear/CMakeFiles/liblinear.dir/blas/dnrm2.c.o.provides

LibLinear/CMakeFiles/liblinear.dir/blas/dnrm2.c.o.provides.build: LibLinear/CMakeFiles/liblinear.dir/blas/dnrm2.c.o


LibLinear/CMakeFiles/liblinear.dir/blas/dscal.c.o: LibLinear/CMakeFiles/liblinear.dir/flags.make
LibLinear/CMakeFiles/liblinear.dir/blas/dscal.c.o: ../LibLinear/blas/dscal.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hansry/landmark_keypoint_matching/bing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object LibLinear/CMakeFiles/liblinear.dir/blas/dscal.c.o"
	cd /home/hansry/landmark_keypoint_matching/bing/build/LibLinear && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/liblinear.dir/blas/dscal.c.o   -c /home/hansry/landmark_keypoint_matching/bing/LibLinear/blas/dscal.c

LibLinear/CMakeFiles/liblinear.dir/blas/dscal.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/liblinear.dir/blas/dscal.c.i"
	cd /home/hansry/landmark_keypoint_matching/bing/build/LibLinear && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hansry/landmark_keypoint_matching/bing/LibLinear/blas/dscal.c > CMakeFiles/liblinear.dir/blas/dscal.c.i

LibLinear/CMakeFiles/liblinear.dir/blas/dscal.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/liblinear.dir/blas/dscal.c.s"
	cd /home/hansry/landmark_keypoint_matching/bing/build/LibLinear && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hansry/landmark_keypoint_matching/bing/LibLinear/blas/dscal.c -o CMakeFiles/liblinear.dir/blas/dscal.c.s

LibLinear/CMakeFiles/liblinear.dir/blas/dscal.c.o.requires:

.PHONY : LibLinear/CMakeFiles/liblinear.dir/blas/dscal.c.o.requires

LibLinear/CMakeFiles/liblinear.dir/blas/dscal.c.o.provides: LibLinear/CMakeFiles/liblinear.dir/blas/dscal.c.o.requires
	$(MAKE) -f LibLinear/CMakeFiles/liblinear.dir/build.make LibLinear/CMakeFiles/liblinear.dir/blas/dscal.c.o.provides.build
.PHONY : LibLinear/CMakeFiles/liblinear.dir/blas/dscal.c.o.provides

LibLinear/CMakeFiles/liblinear.dir/blas/dscal.c.o.provides.build: LibLinear/CMakeFiles/liblinear.dir/blas/dscal.c.o


# Object files for target liblinear
liblinear_OBJECTS = \
"CMakeFiles/liblinear.dir/linear.cpp.o" \
"CMakeFiles/liblinear.dir/tron.cpp.o" \
"CMakeFiles/liblinear.dir/blas/daxpy.c.o" \
"CMakeFiles/liblinear.dir/blas/ddot.c.o" \
"CMakeFiles/liblinear.dir/blas/dnrm2.c.o" \
"CMakeFiles/liblinear.dir/blas/dscal.c.o"

# External object files for target liblinear
liblinear_EXTERNAL_OBJECTS =

LibLinear/libliblinear.a: LibLinear/CMakeFiles/liblinear.dir/linear.cpp.o
LibLinear/libliblinear.a: LibLinear/CMakeFiles/liblinear.dir/tron.cpp.o
LibLinear/libliblinear.a: LibLinear/CMakeFiles/liblinear.dir/blas/daxpy.c.o
LibLinear/libliblinear.a: LibLinear/CMakeFiles/liblinear.dir/blas/ddot.c.o
LibLinear/libliblinear.a: LibLinear/CMakeFiles/liblinear.dir/blas/dnrm2.c.o
LibLinear/libliblinear.a: LibLinear/CMakeFiles/liblinear.dir/blas/dscal.c.o
LibLinear/libliblinear.a: LibLinear/CMakeFiles/liblinear.dir/build.make
LibLinear/libliblinear.a: LibLinear/CMakeFiles/liblinear.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hansry/landmark_keypoint_matching/bing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX static library libliblinear.a"
	cd /home/hansry/landmark_keypoint_matching/bing/build/LibLinear && $(CMAKE_COMMAND) -P CMakeFiles/liblinear.dir/cmake_clean_target.cmake
	cd /home/hansry/landmark_keypoint_matching/bing/build/LibLinear && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/liblinear.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
LibLinear/CMakeFiles/liblinear.dir/build: LibLinear/libliblinear.a

.PHONY : LibLinear/CMakeFiles/liblinear.dir/build

LibLinear/CMakeFiles/liblinear.dir/requires: LibLinear/CMakeFiles/liblinear.dir/linear.cpp.o.requires
LibLinear/CMakeFiles/liblinear.dir/requires: LibLinear/CMakeFiles/liblinear.dir/tron.cpp.o.requires
LibLinear/CMakeFiles/liblinear.dir/requires: LibLinear/CMakeFiles/liblinear.dir/blas/daxpy.c.o.requires
LibLinear/CMakeFiles/liblinear.dir/requires: LibLinear/CMakeFiles/liblinear.dir/blas/ddot.c.o.requires
LibLinear/CMakeFiles/liblinear.dir/requires: LibLinear/CMakeFiles/liblinear.dir/blas/dnrm2.c.o.requires
LibLinear/CMakeFiles/liblinear.dir/requires: LibLinear/CMakeFiles/liblinear.dir/blas/dscal.c.o.requires

.PHONY : LibLinear/CMakeFiles/liblinear.dir/requires

LibLinear/CMakeFiles/liblinear.dir/clean:
	cd /home/hansry/landmark_keypoint_matching/bing/build/LibLinear && $(CMAKE_COMMAND) -P CMakeFiles/liblinear.dir/cmake_clean.cmake
.PHONY : LibLinear/CMakeFiles/liblinear.dir/clean

LibLinear/CMakeFiles/liblinear.dir/depend:
	cd /home/hansry/landmark_keypoint_matching/bing/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hansry/landmark_keypoint_matching/bing /home/hansry/landmark_keypoint_matching/bing/LibLinear /home/hansry/landmark_keypoint_matching/bing/build /home/hansry/landmark_keypoint_matching/bing/build/LibLinear /home/hansry/landmark_keypoint_matching/bing/build/LibLinear/CMakeFiles/liblinear.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : LibLinear/CMakeFiles/liblinear.dir/depend
