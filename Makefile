# Universal Makefile for sceptic3D

# Set shell to bash (default is sh)
SHELL := /bin/bash

# Set SCEPTIC3D version number
VERSION := 0.9

# Set number of threads to use for HDF make
NUMPROC := 8

# Set compilers
G77 := mpiifort
G77nonmpi := ifort
G90 := mpiifort
G90nonmpi := ifort

CC :=icc
CPP :=icpc


NVCC		:= /usr/local/cuda/bin/nvcc
MKLROOT := /opt/intel/Compiler/11.1/073/mkl
MKL_LAPACK := -L$(MKLROOT)/lib/em64t -lmkl_solver_lp64_sequential -Wl,--start-group  -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -Wl,--end-group -lpthread
CUDA_INCLUDE_PATH	:= -I./cutil -I/opt/intel/Compiler/11.1/073/include -I/opt/intel/Compiler/11.1/073/mkl/include
CUDAFORTRAN_FLAGS := -L$(LD_LIBRARY_PATH) -L/usr/local/cuda/lib64 -lcudart -lcuda -L./cutil -lcutil_x86_64 $(MKL_LAPACK) -I$(CUDA_INCLUDE_PATH)
PGPLOT_FLAGS := -L/usr/local/pgplot -lcpgplot -lpgplot -lX11 -lgcc -lm
PGPLOT_DIR = /usr/local/pgplot/
NVCCFLAGS	:=  -m64 -O3 -Xptxas -O3 -use_fast_math -ftz=true -prec-div=false -prec-sqrt=false --maxrregcount=64  -gencode arch=compute_20,code=sm_20 --ptxas-options=-v -ccbin $(CC) -Xcompiler -fast $(CUDA_INCLUDE_PATH) 
#NVCCFLAGS	:=  -m64 -O3 -Xptxas -O3  -Xptxas -maxrregcount=40 -gencode arch=compute_20,code=sm_20 --ptxas-options=-v -ccbin /opt/intel/Compiler/11.1/073/bin/intel64/icc -Xcompiler -fast $(CUDA_INCLUDE_PATH) 

NVCC +=  $(NVCCFLAGS) #-L/home/josh/lib -lcutil_x86_64


FLIBROOT := /opt/intel/Compiler/11.1/073
INTEL_LIBS := -L$(FLIBROOT)/lib/intel64 -lifcore -lifport
# HDF root directory
#HDFDIR := $(realpath hdf5-1.8.4)
# realpath not available on loki, so use hack
DIRHDF := $(PWD)/hdf5-1.8.4

# Set Xlib location
DIRXLIB := $(shell ./setxlib)
# Note that this may not work properly, and is not needed
#   if X11 and Xt are in /usr/lib(64)

# Accis lib location unless already set
DIRACCIS ?= ./accis


# Libraries and options to pass to linker
LIB := -L$(DIRXLIB) -L$(DIRACCIS) -laccisX -lXt -lX11 $(CUDAFORTRAN_FLAGS) $(INTEL_LIBS)
# Show time and memory usage (debugging)
LIB += -Wl,-stats

# Libraries and options to pass to linker for HDF version
LIBHDF := $(LIB)
# HDF libraries and options (from h5fc -show in DIRHDF/bin)
LIBHDF += -L$(DIRHDF)/lib -lhdf5hl_fortran -lhdf5_hl \
          -lhdf5_fortran -lhdf5 -lz -lm -Wl,-rpath -Wl,$(DIRHDF)/lib
# Note that the -Wl,-rpath... options are needed since LD_LIBRARY_PATH
#   does not contain the location of the hdf shared libraries at runtime 


# Options to pass to compiler
OPTCOMP := -I. -I/opt/intel/Compiler/11.1/073/include -fPIC -assume no2underscores -fp-model precise
# Show all warnings exept unused variables
#OPTCOMP += -Wall -Wno-unused-variable
# Enable optimization
OPTCOMP += -xHOST -O3 -no-prec-div -xSSE4.2
# Save debugging info
#OPTCOMP += -g -pg
# Do bounds check (debugging)
#OPTCOMP += -ffortran-bounds-check
# Save profiling information (debugging)
#OPTCOMP += -pg

# Options to pass to compiler for HDF version
OPTCOMPHDF := $(OPTCOMP)
# Include directory with HDF modules
OPTCOMPHDF += -I$(DIRHDF)/include
# Enable HDF by defining 'HDF' for pre-compiler
OPTCOMPHDF += -DHDF

# Options to pass to compiler for MPI version
OPTCOMPMPI := $(OPTCOMP)
# Enable MPI by defining 'MPI' for pre-compiler
OPTCOMPMPI += -DMPI

# Options to pass to compiler for MPI & HDF version
OPTCOMPMPIHDF := $(OPTCOMPHDF)
# Enable MPI by defining 'MPI' for pre-compiler
OPTCOMPMPIHDF += -DMPI


# Objects common to all versions of sceptic3D
OBJ := initiate.o \
       advancing_encapsulated.o \
       advancing_interface.o \
       gpu_init.o \
       initialize_gpu.o \
       XPlist_host_functions.o \
       chargefield_gpu.o \
       randc.o \
       randf.o \
       diags.o \
       outputs.o \
       chargefield.o \
       stringsnames.o \
       rhoinfcalc.o \
       shielding3D.o \
       timing.f \
       cg3D_gpu.o \
       shielding3D_gpu.o \
       gpu_psolve_tests.o \
       stupid_sort.o
# Reinjection related objects
OBJ += orbitinject.o \
       extint.o \
       maxreinject.o
       
# Objects for GPU testing
OBJ += gpu_tests.o

# Objects for HDF version of sceptic3D
OBJHDF := $(OBJ) \
          outputhdf.o

# Objects for MPI version of sceptic3D
OBJMPI := $(OBJ) \
          cg3dmpi.o \
          mpibbdy.o \
          shielding3D_par.o

# Objects for MPI & HDF version of sceptic3D
OBJMPIHDF := $(OBJMPI) \
          outputhdf.o

# Default target is serial sceptic3D without HDF support
sceptic3D : sceptic3D.F piccom.f errcom.f $(OBJ) ./accis/libaccisX.a
	$(G77) $(OPTCOMP) -o sceptic3D sceptic3D.F $(OBJ) $(LIB)

# sceptic3D with HDF
sceptic3Dhdf : sceptic3D.F piccom.f errcom.f $(OBJHDF) ./accis/libaccisX.a
	$(G77) $(OPTCOMPHDF) -o sceptic3Dhdf sceptic3D.F $(OBJHDF) $(LIBHDF)

# sceptic3D with MPI
sceptic3Dmpi : sceptic3D.F piccom.f errcom.f piccomcg.f $(OBJMPI) ./accis/libaccisX.a
	$(G77) $(OPTCOMPMPI) -o sceptic3Dmpi sceptic3D.F $(OBJMPI) $(LIB)

# sceptic3D with MPI & HDF
sceptic3Dmpihdf : sceptic3D.F piccom.f errcom.f piccomcg.f $(OBJMPIHDF) ./accis/libaccisX.a
	$(G77) $(OPTCOMPMPIHDF) -o sceptic3Dmpihdf sceptic3D.F $(OBJMPIHDF) $(LIBHDF)


# HDF related rules
outputhdf.o : outputhdf.f piccom.f errcom.f colncom.f $(DIRHDF)/lib/libhdf5.a
	$(G90) -c $(OPTCOMPHDF) outputhdf.f

# Though more than one hdf library used, choose one as trigger
$(DIRHDF)/lib/libhdf5.a :
	cd $(DIRHDF) &&	\
	./configure --prefix=$(DIRHDF) --enable-fortran \
	FC=$(G90nonmpi) && \
	make -j$(NUMPROC) && \
	make install
# Note that providing an mpi compiler to hdf will cause it to build
#   the MPI version, which is not needed (and didn't work on sceptic)


# Other rules
./accis/libaccisX.a : ./accis/*.f
	make -C accis

orbitint : orbitint.f coulflux.o $(OBJ) ./accis/libaccisX.a
	$(G77) $(OPTCOMP) -o orbitint orbitint.f $(OBJ) coulflux.o $(LIB)

coulflux.o : tools/coulflux.f
	$(G77) -c $(OPTCOMP) tools/coulflux.f

fvinjecttest : fvinjecttest.F fvinject.o reinject.o initiate.o advancing.o chargefield.o randf.o fvcom.f
	$(G77)  -o fvinjecttest $(OPTCMOP) fvinjecttest.F fvinject.o reinject.o initiate.o advancing.o chargefield.o randf.o  $(LIB)

fvinject.o : fvinject.f fvcom.f piccom.f errcom.f
	$(G77) -c $(OPTCOMP) fvinject.f


# Pattern rules
%.o : %.f piccom.f errcom.f fvcom.f;
	$(G77) -c $(OPTCOMP) $*.f

%.o : %.F piccom.f errcom.f;
	$(G77) -c $(OPTCOMP) $*.F
	
%.o : %.cu
	$(NVCC) -c $*.cu

% : %.f
	$(G77) -o $* $(OPTCOMP) $*.f $(LIB)

% : %.F
	$(G77) -o $* $(OPTCOMP) $*.F $(LIB)


# Distributable archive (explicitly make .PHONY to force rebuild) 
sceptic3D.tar.gz : ./tar-1.26/src/tar
	make -C accis mproper
	make cleanall
	./copyattach.sh $(VERSION)
	./tar-1.26/src/tar -chzf sceptic3D.tar.gz \
	  -C .. sceptic3D \
	  --exclude-vcs --exclude="hdf5-1.8.4" \
	  --exclude="hdf5-1.8.4.tar.gz" \
	  --exclude="tar-1.26" \
	  --exclude="sceptic3D.tar.gz"
	./copyremove.sh

# Distributable HDF archive (explicitly make .PHONY to force rebuild)
hdf5-1.8.4.tar.gz : ./tar-1.26/src/tar
	make cleanhdf
	./tar-1.26/src/tar -chzf hdf5-1.8.4.tar.gz hdf5-1.8.4 \
	  --exclude-vcs

# Need tar version with --exclude-vcs support, so build from source
./tar-1.26/src/tar :
	cd tar-1.26 && ./configure
	make -C tar-1.26


# The following targets will never actually exist
.PHONY: all distro clean cleandata cleanaccis cleanhdf cleanall ftnchek sceptic3D.tar.gz hdf5-1.8.4.tar.gz

all : sceptic3D sceptic3Dhdf sceptic3Dmpi sceptic3Dmpihdf

distro : sceptic3D.tar.gz hdf5-1.8.4.tar.gz

clean :
	-rm *.o
	-rm *.ps
	-rm *.orb
	-rm *.html
	-rm Orbits.txt
	-rm *~
	-rm .*~
	-rm \#*\#
	-rm sceptic3D sceptic3Dmpi sceptic3Dhdf sceptic3Dmpihdf

cleandata :
	-rm *.dat
	-rm *.frc
	-rm *.h5

cleanaccis :
	make -C accis clean
	-rm ./accis/libaccisX.a

cleanhdf :
	make -C $(DIRHDF) clean
	-rm $(DIRHDF)/lib/libhdf5.a

cleanall :
	make clean
	make cleandata
	make cleanaccis
	make cleanhdf

ftnchek :
	ftnchek -nocheck -nof77 -calltree=text,no-sort -mkhtml -quiet -brief sceptic3D.F *.f
