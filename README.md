### A simple neural network with one hidden layer trained on MNIST's hand written digits

This implementation uses solely numpy (pytorch is used to get the data).
The number of nodes in the hidden layer is an input parameter. I found initializing the weights from a uniform distribution between -0.15 and 0.15 works best.

The branch called 'numba' is meant for a CUDA-optimized version using numba, which is being under development.

![Pic_for_MNIST_project](https://github.com/user-attachments/assets/4f558f94-9e72-464f-99af-97d3b462ded5)
