## Microsoft SEAL For Python

Microsoft [**SEAL**](https://github.com/microsoft/SEAL) is an easy-to-use open-source ([MIT licensed](https://github.com/microsoft/SEAL/blob/master/LICENSE)) homomorphic encryption library developed by the Cryptography Research group at Microsoft.

[**pybind11**](https://github.com/pybind/pybind11) is a lightweight header-only library that exposes C++ types in Python and vice versa, mainly to create Python bindings of existing C++ code.


## Special Notes

- This implementation contains noise-flooding to protect the circuit privacy of Homomorphic Encryption
- This repo is dedicated to [GForce: GPU-Friendly Oblivious and Rapid Neural Network Inference](https://github.com/Lucieno/gforce-public), but it is also welcome for other uses.


## Contents

* [Build](https://github.com/Huelse/pyseal#build)
* [Tests](https://github.com/Huelse/pyseal#tests)
* [About](https://github.com/Huelse/pyseal#about)
* [Contributing](https://github.com/Huelse/pyseal#contributing)


## Build

* ### Environment

  CMake (>= 3.10), GNU G++ (>= 6.0) or Clang++ (>= 5.0), Python (>=3.6.8)

  `sudo apt-get update && sudo apt-get install -y g++ make git gcc-8 g++-8`

  `git clone https://github.com/Lucieno/SEAL-Python.git`

  We highly recommend you to install SEAL-Python in a [Anaconda](https://www.anaconda.com/) virtual environment.

* ### SEAL 3.3.2

  ```shell
  cd SEAL/native/src
  cmake .
  make
  sudo make install
  ```

* ### pybind11

    ```
    conda install pybind11
    ```
  
* ### SEAL-Python

  ```
  cd src
  pip install -r requirements.txt
  
  export CC='/usr/bin/gcc-8'
  export CXX='/usr/bin/g++-8'
  python setup.py install
  ```

* ### Others

    If you clone a new SEAL lib from the [Github](https://github.com/microsoft/SEAL), do not forget add a function set_scale in `seal/ciphertext.h` line 632, like this:

    ```c++
    /**
    Set the scale.
    */
    inline void set_scale(double scale)
    {
      scale_ = scale;
    }
    ```

    The setuptools will build a dynamic Link Library in project folder, named like `seal.cpython-36m-x86_64-linux-gnu.so`.

    You can use the CMake to build it, I have already written the `CMakelists.txt`. Make sure the SEAL and the pybind11 is correctly installed.
    
    The path is very important, please check it before you do anything.



## Tests

`cd tests`

`python [example_name].py`

* The `.so` file need in the same folder, or you had `make install` it already.



## Getting Started

| C++               | Python           | Description                                                  | Progress |
| ----------------- | ---------------- | ------------------------------------------------------------ | -------- |
| 1_bfv_basics.cpp  | 1_bfv_basics.py  | Encrypted modular arithmetic using the BFV scheme            | Finished |
| 2_encoders.cpp    | 2_encoders.py    | Encoding more complex data into Microsoft SEAL plaintext objects | Finished |
| 3_levels.cpp      | 3_levels.py      | Introduces the concept of levels; prerequisite for using the CKKS scheme | Finished |
| 4_ckks_basics.cpp | 4_ckks_basics.py | Encrypted real number arithmetic using the CKKS scheme       | Finished |
| 5_rotation.cpp    | 5_rotation.py    | Performing cyclic rotations on encrypted vectors in the BFV and CKKS schemes | Finished |
| 6_performance.cpp | 6_performance.py | Performance tests for Microsoft SEAL                         | Finished |


