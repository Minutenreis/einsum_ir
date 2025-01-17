echo "**********************************"
echo "*** Installing System Packages ***"
echo "**********************************"

sudo dnf install -y vim htop gfortran clang wget git cmake environment-modules g++-13 python3.10 scons tmux

echo "**************************"
echo "*** Installing Pytorch ***"
echo "**************************"

python3.10 -m venv venv_pytorch
source venv_pytorch/bin/activate
pip3 install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__config__.show()); print(torch.__config__.parallel_info());"
ln -s /home/fedora/venv_pytorch/lib/python3.10/site-packages/torch.libs/* /home/fedora/venv_pytorch/lib/python3.10/site-packages/torch/lib/

echo "*****************************************"
echo "*** Installing Einsum IR Dependencies ***"
echo "*****************************************"

git clone https://github.com/libxsmm/libxsmm.git
cd libxsmm
git log | head -n 25
make BLAS=0 -j
cd ..

wget https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.28.tar.gz
tar -xvf v0.3.28.tar.gz
cd OpenBLAS-0.3.28
make -j
make PREFIX=$(pwd)/../openblas install
cd ..

git clone https://github.com/devinamatthews/tblis.git tblis_src
cd tblis_src
git checkout 2cbdd21
git log | head -n 25
./configure --prefix=$(pwd)/../tblis --enable-thread-model=openmp
sed -i '971d' Makefile
make -j
make install
cd ..
rm -rf tblis_src

for dir_type in indexed_dpd indexed dpd fwd
do
mkdir -p tblis/include/tblis/external/marray/marray/${dir_type}
mv tblis/include/tblis/external/marray/marray/*${dir_type}*.hpp tblis/include/tblis/external/marray/marray/${dir_type}
done

mkdir -p tblis/include/tblis/external/marray/marray/detail
mv tblis/include/tblis/external/marray/marray/utility* tblis/include/tblis/external/marray/marray/detail

echo "****************************"
echo "*** Installing Einsum IR ***"
echo "****************************"

# install mpich
wget https://www.mpich.org/static/downloads/4.2.3/mpich-4.2.3.tar.gz
tar xfz mpich-4.2.3.tar.gz 
mkdir mpich
mkdir tmp_mpich
cd tmp_mpich
../mpich-4.2.3/configure -prefix=/home/fedora/mpich |& tee c.txt
make -j 16
make install

cd ..

git clone https://github.com/Minutenreis/einsum_ir.git
cd einsum_ir
git log | head -n 25

wget https://github.com/catchorg/Catch2/releases/download/v2.13.10/catch.hpp



CXX=g++-13 CC=g++-13 scons libtorch=../venv_pytorch/lib/python3.10/site-packages/torch blas=$(pwd)/../openblas tblis=$(pwd)/../tblis libxsmm=$(pwd)/../libxsmm -j8 --sconstruct=SConstruct_g8c
mv build build_gcc

CXX=clang++ CC=clang scons libtorch=../venv_pytorch/lib/python3.10/site-packages/torch blas=$(pwd)/../openblas tblis=$(pwd)/../tblis libxsmm=$(pwd)/../libxsmm -j8 --sconstruct=SConstruct_g8c
mv build build_llvm
cd ..