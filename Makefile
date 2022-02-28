################################################################################
#
# Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:
#
# This source code is subject to NVIDIA ownership rights under U.S. and
# international Copyright laws.
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
# OR PERFORMANCE OF THIS SOURCE CODE.
#
# U.S. Government End Users.  This source code is a "commercial item" as
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
# "commercial computer software" and "commercial computer software
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
# and is provided to the U.S. Government only as a commercial end item.
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
# source code with only those rights set forth herein.
#
################################################################################
#
# Makefile project only supported on Mac OS X and Linux Platforms)
#
################################################################################
# Location of source files relative to Makefile
SRC_DIR=./src

# Location of the CUDA Toolkit
CUDA=$(shell echo ${PATH} | sed 's/.*\(cuda-[0-9]\+.[0-9]\).*/\1/')
CUDA_PATH ?= /usr/local/$(CUDA)

##############################
# start deprecated interface #
##############################
ifeq ($(x86_64),1)
	$(info WARNING - x86_64 variable has been deprecated)
	$(info WARNING - please use TARGET_ARCH=x86_64 instead)
	TARGET_ARCH ?= x86_64
endif
ifeq ($(ARMv7),1)
	$(info WARNING - ARMv7 variable has been deprecated)
	$(info WARNING - please use TARGET_ARCH=armv7l instead)
	TARGET_ARCH ?= armv7l
endif
ifeq ($(aarch64),1)
	$(info WARNING - aarch64 variable has been deprecated)
	$(info WARNING - please use TARGET_ARCH=aarch64 instead)
	TARGET_ARCH ?= aarch64
endif
ifeq ($(ppc64le),1)
	$(info WARNING - ppc64le variable has been deprecated)
	$(info WARNING - please use TARGET_ARCH=ppc64le instead)
	TARGET_ARCH ?= ppc64le
endif
ifneq ($(GCC),)
	$(info WARNING - GCC variable has been deprecated)
	$(info WARNING - please use HOST_COMPILER=$(GCC) instead)
	HOST_COMPILER ?= $(GCC)
endif
ifneq ($(abi),)
	$(error ERROR - abi variable has been removed)
endif
############################
# end deprecated interface #
############################

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le armv7l))
	ifneq ($(TARGET_ARCH),$(HOST_ARCH))
		ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le))
			TARGET_SIZE := 64
		else ifneq (,$(filter $(TARGET_ARCH),armv7l))
			TARGET_SIZE := 32
		endif
	else
		TARGET_SIZE := $(shell getconf LONG_BIT)
	endif
else
	$(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif
ifneq ($(TARGET_ARCH),$(HOST_ARCH))
	ifeq (,$(filter $(HOST_ARCH)-$(TARGET_ARCH),aarch64-armv7l x86_64-armv7l x86_64-aarch64 x86_64-ppc64le))
		$(error ERROR - cross compiling from $(HOST_ARCH) to $(TARGET_ARCH) is not supported!)
	endif
endif

# When on native aarch64 system with userspace of 32-bit, change TARGET_ARCH to armv7l
ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_SIZE),aarch64-aarch64-32)
	TARGET_ARCH = armv7l
endif

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),linux darwin qnx android))
	$(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif

# host compiler
ifeq ($(TARGET_OS),darwin)
	ifeq ($(shell expr `xcodebuild -version | grep -i xcode | awk '{print $$2}' | cut -d'.' -f1` \>= 5),1)
		HOST_COMPILER ?= clang++
	endif
else ifneq ($(TARGET_ARCH),$(HOST_ARCH))
	ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
		ifeq ($(TARGET_OS),linux)
			HOST_COMPILER ?= arm-linux-gnueabihf-g++
		else ifeq ($(TARGET_OS),qnx)
			ifeq ($(QNX_HOST),)
				$(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
			endif
			ifeq ($(QNX_TARGET),)
				$(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
			endif
			export QNX_HOST
			export QNX_TARGET
			HOST_COMPILER ?= $(QNX_HOST)/usr/bin/arm-unknown-nto-qnx6.6.0eabi-g++
		else ifeq ($(TARGET_OS),android)
			HOST_COMPILER ?= arm-linux-androideabi-g++
		endif
	else ifeq ($(TARGET_ARCH),aarch64)
		ifeq ($(TARGET_OS), linux)
			HOST_COMPILER ?= aarch64-linux-gnu-g++
		else ifeq ($(TARGET_OS),qnx)
			ifeq ($(QNX_HOST),)
				$(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
			endif
			ifeq ($(QNX_TARGET),)
				$(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
			endif
			export QNX_HOST
			export QNX_TARGET
			HOST_COMPILER ?= $(QNX_HOST)/usr/bin/aarch64-unknown-nto-qnx7.0.0-g++
		else ifeq ($(TARGET_OS), android)
			HOST_COMPILER ?= aarch64-linux-android-g++
		endif
	else ifeq ($(TARGET_ARCH),ppc64le)
		HOST_COMPILER ?= powerpc64le-linux-gnu-g++
	endif
endif
# all compilers required for build
CC            := gcc
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     := -O2 -Wall
LDFLAGS     :=

# build flags
ifeq ($(TARGET_OS),darwin)
	LDFLAGS += -rpath $(CUDA_PATH)/lib
	CCFLAGS += -arch $(HOST_ARCH)
else ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-armv7l-linux)
	LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
	CCFLAGS += -mfloat-abi=hard
else ifeq ($(TARGET_OS),android)
	LDFLAGS += -pie
	CCFLAGS += -fpie -fpic -fexceptions
endif

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
	ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
		ifneq ($(TARGET_FS),)
			GCCVERSIONLTEQ46 := $(shell expr `$(HOST_COMPILER) -dumpversion` \<= 4.6)
			ifeq ($(GCCVERSIONLTEQ46),1)
				CCFLAGS += --sysroot=$(TARGET_FS)
			endif
			LDFLAGS += --sysroot=$(TARGET_FS)
			LDFLAGS += -rpath-link=$(TARGET_FS)/lib
			LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
			LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-gnueabihf
		endif
	endif
endif

# Debug build flags
ifeq ($(dbg),1)
	  NVCCFLAGS += -g -G
	  BUILD_TYPE := debug
else
	  BUILD_TYPE := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

BUILD_ENABLED := 1

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I$(CUDA_PATH)/include
INCLUDES  += -I$(CUDA_PATH)/samples/common/inc
INCLUDES  += -I$(SRC_DIR)
LIBRARIES :=

################################################################################

# Makefile include to help find GL Libraries
include $(SRC_DIR)/findgllib.mk

# OpenGL specific libraries
ifeq ($(TARGET_OS),darwin)
 # Mac OSX specific libraries and paths to include
 LIBRARIES += -L/System/Library/Frameworks/OpenGL.framework/Libraries
 LIBRARIES += -lGL -lGLU
 ALL_LDFLAGS += -Xlinker -framework -Xlinker GLUT
else
 LIBRARIES += $(GLLINK)
 LIBRARIES += -lGL -lGLU -lX11 -lglut
endif

# Eigen Linear Algebra for optimizations
# INCLUDES  += -I$(EIGEN_ROOT_DIR)

# Gencode arguments
ifeq ($(TARGET_ARCH),$(filter $(TARGET_ARCH),armv7l aarch64))
SMS ?= 53
else
SMS ?= 30 35 37 50 52 60
endif

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
BUILD_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

ifeq ($(BUILD_ENABLED),0)
EXEC ?= @echo "[@]"
endif

################################################################################

# Target rules
all: build

build: quadTreeGPU quadTreeFilterGPU quadTreeFilterDevGPU profileFilterGPU profileFilterDevAsyncGPU

check.incs:
	@echo $(INCLUDES)

check.libs:
	@echo $(LIBRARIES)

check.cflags:
	@echo $(ALL_CCFLAGS)
	@echo $(CCFLAGS)

check.arch:
	@echo "Host Arch: " $(TARGET_ARCH)
	@echo "GPU code Arch(s): "$(GENCODE_FLAGS)

check.deps:
ifeq ($(BUILD_ENABLED),0)
	@echo "Build waived due to the missing openGL dependencies"
else
	@echo "Build is ready - all openGL dependencies have been met"
endif

# Quad Tree CUDA Kernels
quadTreeKernels.o: $(SRC_DIR)/quadTreeKernels.cu $(SRC_DIR)/quadTreeKernels.h
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

# Quad Tree Builder
QuadTreeBuilder.o: $(SRC_DIR)/QuadTreeBuilder.cu $(SRC_DIR)/QuadTreeBuilder.h
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -std=c++11 $(GENCODE_FLAGS) -c $<

# main application object
mainBuild.o:$(SRC_DIR)/mainBuild.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -std=c++11 $(GENCODE_FLAGS) -c $<

# main filter object
mainFilter.o:$(SRC_DIR)/mainFilter.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -std=c++11 $(GENCODE_FLAGS) -c $<

# main filter dev object
mainFilterDev.o:$(SRC_DIR)/mainFilterDev.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -std=c++11 $(GENCODE_FLAGS) -c $<

# main profile filter object
mainProfileFilter.o:$(SRC_DIR)/mainProfileFilter.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -std=c++11 $(GENCODE_FLAGS) -c $<

# main profile filter dev async object
mainProfileFilterDevAsync.o:$(SRC_DIR)/mainProfileFilterDevAsync.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -std=c++11 $(GENCODE_FLAGS) -c $<

# main build application
quadTreeGPU: mainBuild.o QuadTreeBuilder.o quadTreeKernels.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mv $@ ./bin

# main Filter application
quadTreeFilterGPU: mainFilter.o QuadTreeBuilder.o quadTreeKernels.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mv $@ ./bin

# main profile app of Filter application
profileFilterGPU: mainProfileFilter.o QuadTreeBuilder.o quadTreeKernels.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mv $@ ./bin

# main profile app of FilterDev Async application
profileFilterDevAsyncGPU: mainProfileFilterDevAsync.o QuadTreeBuilder.o quadTreeKernels.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mv $@ ./bin

# main Filter application
quadTreeFilterDevGPU: mainFilterDev.o QuadTreeBuilder.o quadTreeKernels.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mv $@ ./bin

.PHONY: clean
clean:
	rm -f ./bin/* ./*.o

clobber: clean
