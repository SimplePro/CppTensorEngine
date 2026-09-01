# 🚀 Pure C++ Tensor Engine
A lightweight, high-performance tensor engine built from scratch in C++. <br>
This project implements core deep learning components-from autograd to advanced optimizer-without any external libraries.

## 🌟 Key Features
### 1. Autograd Engine (Automatic Differentiation)
- ***Topological Sorting***: Implements a DFS-based topological sort to ensure the correct order of operations during backpropagation.
- ***Memory Management***: Utilizes C++ ```std::shared_ptr``` to manage the lifecycle of temporary tensors and prevent memory leaks in complex computational graphs.
```cpp
void build_topo(shared_ptr<Tensor> v, vector<shared_ptr<Tensor>>& topo_list, set<shared_ptr<Tensor>>& visited) {
    if (v == nullptr || visited.count(v) > 0) return;

    visited.insert(v);

    if(v->grad_fn) {
        if(v->grad_fn->saved_tensors.size() > 0 && v->grad_fn->saved_tensors[0]->requires_grad) build_topo(v->grad_fn->saved_tensors[0], topo_list, visited);
        if(v->grad_fn->saved_tensors.size() > 1 && v->grad_fn->saved_tensors[1]->requires_grad) build_topo(v->grad_fn->saved_tensors[1], topo_list, visited);
    }

    topo_list.push_back(v);
}
```
### 2. Optimized Matrix Multiplication
- ***Cache-Friendly Design***: Employs matrix transposition for the right-hand side (RHS) operand to achieve a higher cache hit rate during dot product operations.
```cpp
class MatrixMultiplication : public Function {
    public:

    vector<int> res_strides;

    shared_ptr<Tensor> forward(vector<shared_ptr<Tensor>> inputs) {
        auto lhs = inputs[0];
        auto rhs_T = transpose_(inputs[1]); // key points!

        save_for_backward(lhs);
        save_for_backward(rhs_T);

        vector<int> res_shape = lhs->shape;
        res_shape[res_shape.size()-1] = rhs_T->shape[0];
        auto res = make_shared<Tensor>(res_shape, lhs->requires_grad||rhs_T->requires_grad);
        saved_attrs.push_back(res->total_size);
        res_strides = res->strides;

        res->fill(0.0);

        for(int i=0; i<res->total_size; i++) {
            int remaining = i;
            int lhs_idx = 0;
            int rhs_T_idx = 0;
            
            for(int j=0; j<res->shape.size()-1; j++) {
                lhs_idx += remaining/res->strides[j] * lhs->strides[j];
                remaining %= res->strides[j];
            }
            rhs_T_idx = remaining;

            for(int k=0; k<lhs->shape[lhs->shape.size()-1]; k++) {
                res->data[i] += lhs->data[lhs_idx+k] * rhs_T->data[rhs_T_idx*rhs_T->strides[0]+k];
            }
        }

        return res;
    }
    ...
};
```
### 3. Core Components
- ***Broadcasting***: Seamlessly handles operations between tensors of different shapes.
- ***Modular Layers***: Built-in layers like ```Linear(FC)```, ```LeakyReLU```, and ```Softmax```.

## ✨ Challenges & Debugging. 
### 1. The "Zero Output" Problem. (Integer Division Bug)
- **Problem**: The model consistently predicted zero, and all weights were found to be zero after initialization.
- **Root Cause**: During **He Initialization**, the standard deviation calculation used integer division (```2/n_in```), resulting in a ```sigma``` of 0.
- **Solution**: Explicitly cast literals to floating-point (```2.0 / n_in```) to ensure precise weight distribution.
```cpp
void he_initialization(shared_ptr<Tensor> params, int n_in) {
    long double mu = 0.0;
    long double sigma = sqrt(2.0/n_in);
    normal_distribution<double> dist(mu, sigma);

    for(int i=0; i<params->total_size; i++) {
        params->data[i] = dist(gen);
    }
}
```
### 2. The "Accumulation" Bug: Resolving Extreme Gradient Explosion.
- **Problem**: Adding hidden layers caused the gradients to explode exponentially (NaN/Inf).
- **Investigation**: Debugging revealed that outputs from the first layer were being **accumulated** across batches. It appeared as if the memory was "leaking" previous values into current calculations.
- **Solution**: Identified that the ```forward``` pass of ```MatrixMultiplication``` lacked a **Zero-initialization** step for the result tensor. Adding ```res->fill(0.0)``` ensured a clean slate for every operation, stabilizing the entire network.
```cpp
class MatrixMultiplication : public Function {
    public:

    shared_ptr<Tensor> forward(vector<shared_ptr<Tensor>> inputs) {
        ...

        vector<int> res_shape = lhs->shape;
        res_shape[res_shape.size()-1] = rhs_T->shape[0];
        auto res = make_shared<Tensor>(res_shape, lhs->requires_grad||rhs_T->requires_grad);

        res->fill(0.0);

        ...
    }

    ...

};
```
## 🪂 Advanced Techniques
To achieve more stability in a custom engine, the following techniques were implemented:
- **Batch Accumulation**: Simulates larger batch sizes to improve gradient estimation.
- **Gradient Clipping**: Prevents gradient explosion by capping the norm of gradients.
- **Adam Optimization**: Adaptive learning rates for faster and more stable convergence.
- **Gaussian Noise Injection**: Adds $\sigma=0.05$ noise to training data to improve generalization and prevent overfitting.

## 🎈 MNIST Benchmark Results
### Model Architecture
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
             FC1                       [128]            100,352
         LeakyReLU                     [128]                  0
             FC2                       [10]               1,280
          Softmax                      [10]                   0
================================================================
Total params: 101,632
```
### Performance
- **Training Accuracy: 97.41%**
- **Test Accuracy: 96.37%**
- **Generalization Gap: 1.04%** (Extremeley stable due to advanced techniques)

<p align="center">
<img src="figures/28*28->128->leaky->10->softmax + noise (epoch 3)_graph.png" width=90%>
</p>