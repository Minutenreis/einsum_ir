scons libtorch=../venv_pytorch/lib/python3.9/site-packages/torch blas=yes libxsmm=$(pwd)/../libxsmm -j8 --sconstruct=SConstruct_grace
