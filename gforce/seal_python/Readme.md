  ```shell
  conda install -c anaconda cmake
  conda install -c conda-forge gxx_linux-64
  conda install pybind11
  cd seal_python/SEAL/native/src
  cmake -S . -B build -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZLIB=OFF
  cmake --build build
  cd ../../../src
  python3 setup.py build_ext -i
  python3 setup.py install
  ```
