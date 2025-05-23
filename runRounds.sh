#!/bin/bash
INPDIR=inputs

nvc++ -O3 -std=c++14 -acc=multicore  parMDS.cpp -o parMDS.out

EXENAME=parMDS

OUTDIR="results/openmp"
mkdir -p $OUTDIR

touch $OUTDIR/time.txt

# Num OMP Threads is default to #CPU CORES ! 
nTHREADS=`nproc --all`

# Else pick from args!
if [ $# -gt 0 ]; then
  nTHREADS=$1
fi

for file in `ls -Sr $INPDIR/*.vrp` 
do
  # GET the filename with extension removing foldername prefix.
  fileName=$(echo $file | awk -F[./] '{print $(NF-1)}')
  
  #~ ROUNDING is enabled for all instances except Golden+CMT 
  isROUND=1  
  
  # ROUNDING is disabled if Golden or CMT
  if [[ $fileName = Golden* ]] || [[ $fileName = CMT* ]]; then
    isROUND=0
  fi
  
  # EXECUTION
  ./$EXENAME.out $file -nthreads $nTHREADS -round $isROUND > $OUTDIR/$fileName.sol 2>> $OUTDIR/time.txt
  
  echo $file - Done $isROUND
done

sort $OUTDIR/time.txt


# ----

nvc++ -O3 -std=c++14 -acc=multicore  opti/parMDS.cpp -o parMDS.out

EXENAME=parMDS

OUTDIR="results/optimized"
mkdir -p $OUTDIR

touch $OUTDIR/time.txt

# Num OMP Threads is default to #CPU CORES ! 
nTHREADS=`nproc --all`

# Else pick from args!
if [ $# -gt 0 ]; then
  nTHREADS=$1
fi

for file in `ls -Sr $INPDIR/*.vrp` 
do
  # GET the filename with extension removing foldername prefix.
  fileName=$(echo $file | awk -F[./] '{print $(NF-1)}')
  
  #~ ROUNDING is enabled for all instances except Golden+CMT 
  isROUND=1  
  
  # ROUNDING is disabled if Golden or CMT
  if [[ $fileName = Golden* ]] || [[ $fileName = CMT* ]]; then
    isROUND=0
  fi
  
  # EXECUTION
  ./$EXENAME.out $file -nthreads $nTHREADS -round $isROUND > $OUTDIR/$fileName.sol 2>> $OUTDIR/time.txt
  
  echo $file - Done $isROUND
done

sort $OUTDIR/time.txt


# ----

nvcc -O3 --use_fast_math -arch=sm_75 -std=c++14 opti/parMDS.cu -o parMDS.out


OUTDIR="results/cuda"
mkdir -p $OUTDIR

touch $OUTDIR/time.txt

# Num OMP Threads is default to #CPU CORES ! 
nTHREADS=`nproc --all`

# Else pick from args!
if [ $# -gt 0 ]; then
  nTHREADS=$1
fi

for file in `ls -Sr $INPDIR/*.vrp` 
do
  # GET the filename with extension removing foldername prefix.
  fileName=$(echo $file | awk -F[./] '{print $(NF-1)}')
  
  #~ ROUNDING is enabled for all instances except Golden+CMT 
  isROUND=1  
  
  # ROUNDING is disabled if Golden or CMT
  if [[ $fileName = Golden* ]] || [[ $fileName = CMT* ]]; then
    isROUND=0
  fi
  
  # EXECUTION
  ./$EXENAME.out $file -nthreads $nTHREADS -round $isROUND > $OUTDIR/$fileName.sol 2>> $OUTDIR/time.txt
  
  echo $file - Done $isROUND
done

sort $OUTDIR/time.txt


cd results
python3 viz.py