# TF1.2
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
# g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-9.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0



CUDA_ROOT=/usr/local/cuda
TF_ROOT=/home/biolab/virtualenv/hcfs3d/lib/python2.7/site-packages/tensorflow

# tf1.2 my
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I ${TF_ROOT}/include -I ${CUDA_ROOT}/include -lcudart -L ${CUDA_ROOT}/lib64/ -O2 -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=0

#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I ${TF_ROOT}/include -I ${CUDA_ROOT}/include -I ${TF_ROOT}/include/external/nsync/public -lcudart -L ${CUDA_ROOT}/lib64/ -L ${TF_ROOT} -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
