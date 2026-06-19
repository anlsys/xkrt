#!/bin/bash

# Requires
#   LLVM  >= 20.x
#   cmake >= 3.17
#   hwloc

##########################################################################
# Configuration (modify as you want, see CMakeLists.txt for all options) #
##########################################################################

export CC=clang
export CXX=clang++

WORK_DIRECTORY="$(pwd)"

XKRT_BRANCH=gh200-unified-memory
XKBLAS_BRANCH=v2.0

XKRT_CMAKE_BUILD_TYPE="Release"
XKBLAS_CMAKE_BUILD_TYPE="Release"

CMAKE_XKRT_OPTS="-DUSE_CUDA=on -DSTRICT=off"
CMAKE_XKBLAS_OPTS="-DUSE_CUDA=off -DUSE_CUBLAS=on -DSTRICT=off "
CMAKE_XKBLAS_OPTS+="-DUSE_TESTS=on -DUSE_OPENBLAS=on "
#CMAKE_XKBLAS_OPTS+=" -DUSE_MKL=on -DUSE_CBLAS=on -DUSE_TESTS=on"

export CMAKE_PREFIX_PATH=$CUDA_PATH:$CMAKE_PREFIX_PATH

INSTALL_SUFFIX="cuda"

###########
# Install #
###########

INSTALL_DIRECTORY="$(pwd)/install"
REPO_DIRECTORY="$(pwd)/repo"
MODULES_DIRECTORY="$(pwd)/modules"

INSTALL_NAME=$XKRT_CMAKE_BUILD_TYPE-$INSTALL_SUFFIX

mkdir -p $INSTALL_DIRECTORY
mkdir -p $REPO_DIRECTORY
mkdir -p $MODULES_DIRECTORY

##################
# Install XKRT #
##################

git clone https://gitlab.inria.fr/xkrt/dev-v2.git $REPO_DIRECTORY/xkrt
cd $REPO_DIRECTORY/xkrt
git pull
git checkout $XKRT_BRANCH
XKRT_HASH=$(git rev-parse HEAD | cut -c 1-12)
buildir=$REPO_DIRECTORY/xkrt/build/$XKRT_HASH/$INSTALL_NAME
XKRT_INSTALL_DIR=$INSTALL_DIRECTORY/xkrt/$XKRT_HASH/$INSTALL_NAME
rm -rf $buildir
mkdir -p $buildir
cd $buildir
CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH cmake $CMAKE_XKRT_OPTS -DCMAKE_BUILD_TYPE="$XKRT_CMAKE_BUILD_TYPE" -DCMAKE_INSTALL_PREFIX="$XKRT_INSTALL_DIR" $REPO_DIRECTORY/xkrt
make install -j

##################
# Install XKBlas #
##################

git clone https://gitlab.inria.fr/xkblas/dev $REPO_DIRECTORY/xkblas
cd $REPO_DIRECTORY/xkblas
git pull
git checkout $XKBLAS_BRANCH
XKBLAS_HASH=$(git rev-parse HEAD | cut -c 1-12)
buildir=$REPO_DIRECTORY/xkblas/build/$XKBLAS_HASH/$INSTALL_NAME
rm -rf $buildir
XKBLAS_INSTALL_DIR=$INSTALL_DIRECTORY/xkblas/$XKBLAS_HASH/$INSTALL_NAME
mkdir -p $buildir
cd $buildir
CMAKE_PREFIX_PATH=$XKRT_INSTALL_DIR:$CMAKE_PREFIX_PATH cmake $CMAKE_XKBLAS_OPTS -DCMAKE_BUILD_TYPE="$XKBLAS_CMAKE_BUILD_TYPE" -DCMAKE_INSTALL_PREFIX="$XKBLAS_INSTALL_DIR" $REPO_DIRECTORY/xkblas
make install -j

#######################
# Create module files #
#######################

XKRT_MODULE_DIRECTORY=$MODULES_DIRECTORY/xkrt/$XKRT_HASH/
XKBLAS_MODULE_DIRECTORY=$MODULES_DIRECTORY/xkblas/$XKBLAS_HASH/

mkdir -p $XKRT_MODULE_DIRECTORY
mkdir -p $XKBLAS_MODULE_DIRECTORY

XKRT_MODULE_FILE=$XKRT_MODULE_DIRECTORY/$INSTALL_NAME
XKBLAS_MODULE_FILE=$XKBLAS_MODULE_DIRECTORY/$INSTALL_NAME

cp $REPO_DIRECTORY/xkrt/modulefile $XKRT_MODULE_FILE
cp $REPO_DIRECTORY/xkrt/modulefile $XKBLAS_MODULE_FILE

sed -i "s,MY_MODULE_HOME,XKRT_HOME,g"             $XKRT_MODULE_FILE
sed -i "s,MY_WHATIS,xkrt,g"                       $XKRT_MODULE_FILE
sed -i "s,MY_PREFIX_PATH,$XKRT_INSTALL_DIR,g"     $XKRT_MODULE_FILE

sed -i "s,MY_MODULE_HOME,XKBLAS_HOME,g"             $XKBLAS_MODULE_FILE
sed -i "s,MY_WHATIS,xkblas,g"                       $XKBLAS_MODULE_FILE
sed -i "s,MY_PREFIX_PATH,$XKBLAS_INSTALL_DIR,g"     $XKBLAS_MODULE_FILE

echo "-------------------------------------------------"
echo "Success. You may type
\`\`\`
module use $MODULES_DIRECTORY
module load xkrt/$XKRT_HASH/$INSTALL_NAME
module load xkblas/$XKBLAS_HASH/$INSTALL_NAME
\`\`\`
Then build the following program with
    clang++ main.cc -lxkblas
"
echo "-------------------------------------------------"

echo "
# include <xkblas/xkblas.h>

int main(void)
{
    xkblas_init();
    xkblas_deinit();
    return 0;
}
$"
