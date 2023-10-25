# Understanding-Transformers

## Understanding PyTorch 
The first step in understanding deep learning is to understand how the libraries of deep learning work. There are two major libraries that are used for deep learning. The most famous in the research community is PyTorch. The most famous in the industry is TensorFlow. However, the building block of these libraries is tensor. So the question we ask ourselves is what is tensor? 

### What is Tensor
You will find tensors in all deep-learning libraries. But why is tensor so important? Why don't we use regular numpy arrays? Why not lists or tuples? 
The answer is Tensors are built in a way that they facilitate deep learning by being compatible with GPUs. 

```
import torch
tensor1 = tensor([1,2,3])
```
 Here we have initialized a tensor that stores some numbers. It is as simple as this. We can also convert Numpy arrays or lists into tensors with these commands

```
import numpy as np
import torch
array1 = np.array([3,2,1])
tensor2 = torch.from_numpy(array1)
```
