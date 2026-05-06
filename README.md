# 🚀 Pure C++ Tensor Engine
This project is the tensor engine made of C++. <br>
It can be used to train some models like linear regression, classifier such as mnist classification.

## 🌟 Key Features
### 1. Auto Grad
- Build a Topologiest list
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
- Design Function Structure
- Use Smart Pointer to Keep Temporary Object.
### 2. BroadCasting
### 3. Matrix Multiplication
- Use Transpose to Get a Higher Cache Hit Rate.
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

## ✨ Challenges (with MNIST dataset)
### 1. Model gives only 0.
- To find the problem factor, i checked weights of model. and I found that weights are all zeros.
- The key factor of this problem is about type of standard deviation of weights in initialization. when initializing the weights, their s.d. is 2/n_in or 2 / (n_in + n_out). In C++, int / int => int. So I modified this to 2.0 / n_in or 2.0 / (n_in + n_out) and Solved.
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
### 2. Gradient Exploding (Very Extremeley)
- When the number of layers is bigger than 1, always Gradient Exploding Problem emerges, very extremeley.
- Through a lot of experiments, debugging ,, I finally found the key factor of this problem. when model forwarding, some of outputs of fc1 layer were being accumulated. It was memory leaking!
- Then i thought accumulating something has to do with +(add) operation. fc1 consists of matmul ... matmul has +operation. And i missed to initialize result variable in MatrixMultiplication. So i added initializing logic And Finally solved!
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
## 🪂 Techniques
### 1. Batch Accumulation
### 2. Gradient Clipping
### 3. Adam Optimization
### 4. Insert Noise

## 🎈 Result of MNIST
### Model Structure
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
             FC1                       [32]              25,120
         LeakyReLU                     [32]                   0
             FC2                       [10]                 330
          Softmax                      [10]                   0
================================================================
Total params: 25,450
```
### Train Accuracy: 92.14%, Test Accuracy: 91.01%
![alt text](<figures/28*28-\>32-\>leaky-\>10-\>softmax + noise (epoch 200)_graph.png>)
