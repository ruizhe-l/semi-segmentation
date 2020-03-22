# **Framework developed by Ruizhe Li for Xin's students**

## **Environment installation instruction:**

### **Windows 10**

1. Install [64-bit python 3.7.5](https://www.python.org/ftp/python/3.7.5/python-3.7.5-amd64.exe) for windows or [other 64-bit versions](https://www.python.org/downloads/windows/) (select pip as an optional feature), 

1. Install NumPy, and Tensorflow 2.0 with GPU from PyPI:
    ```bash
    pip install --upgrade pip
    pip install --upgrade Numpy
    pip install --upgrade tensorflow-gpu
    ```

3. Install CUDA 10.0.

4. Download cuDNN and copy the files (bin, include, lib) from `*\cuda\` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\`.

5. Test Tensorflow 2.0:
    ```python
    import tensorflow as tf
    hello = tf.constant("hello TensorFlow!")
    print (hello)
    ```
---

### **Linux (Server)**

1. install and update Anaconda

    ```bash
    # move to local folder
    mkdir -p $HOME/usr/local/
    cd $HOME/usr/local

    # download and install Anaconda (if not installed)
    wget "https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh"
    bash Anaconda3-2019.10-Linux-x86_64.sh

    # remove install file
    rm Anaconda3-2019.10-Linux-x86_64.sh

    # update Anaconda
    conda update --all
    ```

2. Create conda environment for tensorflow
    ```
    conda create --name tf2_gpuenv
    ```

3. Activate the environment and install tensorflow-gpu
    ```bash
    # activate tensorflow environment

    conda activate tf2_gpuenv

    # install python 3.7 package (include pip)
    conda install python=3.7

    # install tensorflow-gpu use pip
    pip install --upgrade tensorflow-gpu

    # install cuda 10.0 and cudnn
    conda install cudatoolkit=10.0
    conda install cudnn
    ```

4. Choosing GPUs
    ```bash
    # check available gpus
    nvidia-smi

    # choose gpu [n]
    export CUDA_VISIBLE_DEVICES=[n]
    ```

5. Test tensorflow2 gpu version in python
    ```python
    import tensorflow as tf

    hello = tf.constant("hello TensorFlow!")
    print (hello)
    ```
---

### **Additional library needed:**
```bash
pip install --upgrade numpy, sklearn, scipy
pip install --upgrade opencv-python
pip install --upgrade Pillow
pip install --upgrade nibabel
pip install --upgrade shutil
```

For the densecrf, you need to install git first from:  https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
```bash
pip install --upgrade git+https://github.com/lucasb-eyer/pydensecrf.git
```
---