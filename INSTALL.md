---
output: html_document
---


# Install PyTorch from the *conda* console



# Install PyTorch from the *R* console



# PyTorch installation with *conda*

## Linux

### CPU, latest

```
conda install pytorch torchvision cpuonly -c pytorch
```

![image-20200830142123191](assets/INSTALL/image-20200830142123191.png)



### GPU cuda 10.2, latest

```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

![image-20200830142105228](assets/INSTALL/image-20200830142105228.png)



### CPU, nightly build

```
conda install pytorch torchvision cpuonly -c pytorch-nightly
```

![image-20200830142145949](assets/INSTALL/image-20200830142145949.png)



### GPU cuda 9.2, nightly build

```
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch-nightly
```

![image-20200830142202925](assets/INSTALL/image-20200830142202925.png)



## MacOS

### CPU, stable

```
conda install pytorch torchvision -c pytorch
```



![image-20200830142251065](assets/INSTALL/image-20200830142251065.png)



### CPU, nightly build

```
conda install pytorch torchvision -c pytorch-nightly
```

![image-20200830142225896](assets/INSTALL/image-20200830142225896.png)



### GPU, stable

```
conda install pytorch torchvision -c pytorch
# MacOS Binaries don't support CUDA, install from source if CUDA is needed
```

![image-20200830142704051](assets/INSTALL/image-20200830142704051.png)
