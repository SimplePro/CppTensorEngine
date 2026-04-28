#include <iostream>
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <memory>
#include <set>
#include <algorithm>
#include <cmath>
#include <random>

using namespace std;

class Tensor;

// Function 클래스 코드 이해 부족.
class Function {
    public:

    vector<shared_ptr<Tensor>> saved_tensors; // 공통 규칙 saved_tensors[0]: left_parent, saved_tensors[1]: right_parent
    vector<int> saved_attrs;

    virtual ~Function() = default;

    // virtual shared_ptr<Tensor> forward(vector<shared_ptr<Tensor>> inputs) = 0;
    virtual void backward(shared_ptr<double[]> grad_output) = 0;

    void save_for_backward(shared_ptr<Tensor> t) {
        saved_tensors.push_back(t);
    }
};

class Tensor {
    public:

    vector<int> shape; // Tensor의 shape 저장
    vector<int> strides; // Tensor 를 flattening 해서 사용하는데, indexing 할때 필요한 변수
    int total_size = 1; // Tensor의 total_size
    shared_ptr<double[]> data; // Tensor의 값, Temporary Object 에 대한 해결방법으로 shared_ptr 활용
    // shared_ptr 은 그 자체로 포인터 객체이다.
    // shared_ptr<double> -> 문제점 1. data[i] 접근 안됨    2. ControlBlock이 delete[]가 아닌 delete를 하여 memory leaking!!
    // shared_ptr<double*> -> 이중포인터
    // shared_ptr<double[]> 를 통해 배열 전체를 관리해야함을 알려주고, data[i] 접근 가능

    // shared_ptr<Tensor> left_parent = nullptr;
    // shared_ptr<Tensor> right_parent = nullptr;

    shared_ptr<double[]> grad = nullptr; // gradient 저장 변수
    // string op = "";
    shared_ptr<Function> grad_fn = nullptr;
    
    bool requires_grad = true;

    Tensor (vector<int> s, bool requires_grad=true) : shape(s), requires_grad(requires_grad) {
        for(int dim : shape) {
            total_size *= dim;
        }

        strides.resize(shape.size());

        strides[shape.size()-1] = 1;

        for(int i=shape.size()-2; i >= 0; i--) {
            strides[i] = strides[i+1]*shape[i+1];
        }

        // data = new double[total_size];
        data = shared_ptr<double[]>(new double[total_size]);
        // data = make_shared<double[]>(total_size); // aviailable for c++20

        if (requires_grad) {
            grad = shared_ptr<double[]>(new double[total_size]);
            for(int i = 0; i < total_size; i++) {
                grad[i] = 0.0;
            }
        }
    }

    // ~Tensor() {
    //     delete[] data;
    // }

    void fill(int value) {
        for(int i = 0; i < total_size; i++) {
            data[i] = value;
        }
    }

    void print() {
        for(int i = 0; i < total_size; i++) {
            for(int j=0; j < strides.size()-1; j++) {
                if (i != 0 && i % strides[j] == 0) cout << endl;
            }
            cout << data[i] << " ";
        }
    }

    // index type: initializer_list<int> or vector<int>
    template <typename T>
    int get_flatten_idx(const T& index) {
        // 디버깅할 때만 사용
        if (index.size() != shape.size()) {
            throw out_of_range("Dimension mismatch!");
        }

        int idx = 0;
        int i = 0;
        for (int s : index) {
            if(s < 0 || s >= shape[i]) {
                throw out_of_range("Index out of bounds at dimension " + to_string(i));
            }
            idx += s * strides[i];
            i += 1;
        }

        return idx;
    }

    double& operator() (initializer_list<int> index) {
        return data[get_flatten_idx<initializer_list<int>>(index)];
    }

    double& get_gradient_element(initializer_list<int> index) {
        return grad[get_flatten_idx<initializer_list<int>>(index)];
    }


    void reshape(initializer_list<int> shape_) {
        int size = 1;
        for(int dim : shape_) {
            size *= dim;
        }

        if(total_size != size) {
            throw invalid_argument("Size mismatch!");
        }
        
        shape.resize(shape_.size());
        strides.resize(shape_.size());

        for(int i=0; i < shape_.size(); i++) {
            shape[i] = *(shape_.begin() + i);
            // cout << shape[i] << endl;
        }
        
        strides[shape_.size()-1] = 1;
        for(int i=shape_.size()-2; i >= 0; i--) {
            strides[i] = strides[i+1] * shape[i+1];
        }
    }

    void set_data(const vector<double>& values) {
        if (values.size() != this->total_size) {
            throw runtime_error("Size mismatch!");
        }

        copy(values.begin(), values.end(), this->data.get());
    }

};

vector<vector<int>> get_broadcasting_index(const vector<int>& shape1, const vector<int>& strides1,
                                            const vector<int>& shape2, const vector<int>& strides2) {
    // int max_size = shape1.size() > shape2.size() ? shape1.size() : shape2.size();
    int max_size = max(shape1.size(), shape2.size());
    
    vector<int> res_shape(max_size);
    vector<int> res_strides(max_size);

    vector<int> new_strides1(max_size);
    vector<int> new_strides2(max_size);
    
    for(int i=0; i<max_size; i++) {
        int idx1 = shape1.size() - 1 - i;
        int idx2 = shape2.size() - 1 - i;
        int res_idx = max_size - 1 - i;

        int s1 = (idx1>=0) ? shape1[idx1] : 1;
        int s2 = (idx2>=0) ? shape2[idx2] : 1;

        int st1 = (idx1>=0) ? strides1[idx1] : 0;
        int st2 = (idx2>=0) ? strides2[idx2] : 0;

        if(s1 != s2 && s1 != 1 && s2 != 1) throw runtime_error("Error: Shapes are not broadcastable!");

        res_shape[res_idx] = max(s1, s2);
        
        new_strides1[res_idx] = (s1==1) ? 0 : st1;
        new_strides2[res_idx] = (s2==1) ? 0 : st2;
    }

    int res_total_size = 1;
    for(int s : res_shape) res_total_size *= s;
    
    res_strides[max_size-1] = 1;
    for(int i=max_size-2; i >= 0; i--) {
            res_strides[i] = res_strides[i+1]*res_shape[i+1];
    }

    vector<int> index1(res_total_size), index2(res_total_size); // flattened

    for(int i=0; i<res_total_size; i++) {
        int i1=0, i2=0;
        int remaining = i;

        for(int j=0; j<max_size; j++) {
            int multi_dim_index = remaining / res_strides[j];
            remaining %= res_strides[j];

            i1 += multi_dim_index * new_strides1[j];
            i2 += multi_dim_index * new_strides2[j];
        }

        index1[i] = i1;
        index2[i] = i2;
    }

    return {index1, index2, res_shape};

}

class Add : public Function {
    public:

    vector<vector<int>> broadcasting_index;

    shared_ptr<Tensor> forward(vector<shared_ptr<Tensor>> inputs) {
        auto lhs = inputs[0];
        auto rhs = inputs[1];

        save_for_backward(lhs);
        save_for_backward(rhs);

        broadcasting_index = get_broadcasting_index(lhs->shape, lhs->strides, rhs->shape, rhs->strides);

        auto res = make_shared<Tensor>(broadcasting_index[2], lhs->requires_grad || rhs->requires_grad);
        saved_attrs.push_back(res->total_size);

        for(int i=0; i<res->total_size; i++) {
            (res->data)[i] = (lhs->data)[broadcasting_index[0][i]] + (rhs->data)[broadcasting_index[1][i]];
        }
        // auto res = make_shared<Tensor>(lhs->shape, lhs->requires_grad || rhs->requires_grad);

        // for(int i=0; i<lhs->total_size; i++) {
        //     (res->data)[i] = (lhs->data)[i] + (rhs->data)[i];
        // }

        return res;
    }

    void backward(shared_ptr<double[]> grad_output) override {

        auto lhs = saved_tensors[0];
        auto rhs = saved_tensors[1];

        for(int i = 0; i < saved_attrs[0]; i++) {
            if(lhs->requires_grad) (lhs->grad)[broadcasting_index[0][i]] += grad_output[i];
            if(rhs->requires_grad) (rhs->grad)[broadcasting_index[1][i]] += grad_output[i];
        }
    }
};

class Subtraction : public Function {
    public:

    vector<vector<int>> broadcasting_index;

    shared_ptr<Tensor> forward(vector<shared_ptr<Tensor>> inputs) {
        auto lhs = inputs[0];
        auto rhs = inputs[1];

        save_for_backward(lhs);
        save_for_backward(rhs);

        broadcasting_index = get_broadcasting_index(lhs->shape, lhs->strides, rhs->shape, rhs->strides);

        auto res = make_shared<Tensor>(broadcasting_index[2], lhs->requires_grad || rhs->requires_grad);
        saved_attrs.push_back(res->total_size);

        // auto res = make_shared<Tensor>(lhs->shape, lhs->requires_grad || rhs->requires_grad);
        for(int i=0; i < saved_attrs[0]; i++) {
            (res->data)[i] = (lhs->data)[broadcasting_index[0][i]] - (rhs->data)[broadcasting_index[1][i]];
            // (res->data)[i] = (lhs->data)[i] - (rhs->data)[i];
        }

        return res;
    }

    void backward(shared_ptr<double[]> grad_output) override {
        auto lhs = saved_tensors[0];
        auto rhs = saved_tensors[1];

        for(int i=0; i < saved_attrs[0]; i++) {
            if(lhs->requires_grad) (lhs->grad)[broadcasting_index[0][i]] += grad_output[i];
            if(rhs->requires_grad) (rhs->grad)[broadcasting_index[1][i]] -= grad_output[i];
        }
    }
};

shared_ptr<Tensor> subtraction(shared_ptr<Tensor> lhs, shared_ptr<Tensor> rhs) {
    auto grad_fn = make_shared<Subtraction>();
    auto res = grad_fn->forward(vector<shared_ptr<Tensor>>{lhs, rhs});
    res->grad_fn = grad_fn;

    return res;
}

class Multiplication : public Function {
    public:

    vector<vector<int>> broadcasting_index;

    shared_ptr<Tensor> forward(vector<shared_ptr<Tensor>> inputs) {
        auto lhs = inputs[0];
        auto rhs = inputs[1];

        save_for_backward(lhs);
        save_for_backward(rhs);

        broadcasting_index = get_broadcasting_index(lhs->shape, lhs->strides, rhs->shape, rhs->strides);

        auto res = make_shared<Tensor>(broadcasting_index[2], lhs->requires_grad || rhs->requires_grad);
        saved_attrs.push_back(res->total_size);

        for(int i = 0; i < saved_attrs[0]; i++) {
            (res->data)[i] = (lhs->data)[broadcasting_index[0][i]] * (rhs->data)[broadcasting_index[1][i]];
        }

        return res;
    }

    void backward(shared_ptr<double[]> grad_output) override {
        auto lhs = saved_tensors[0];
        auto rhs = saved_tensors[1];

        for(int i = 0; i < saved_attrs[0]; i++) {
            if(lhs->requires_grad) (lhs->grad)[broadcasting_index[0][i]] += (rhs->data)[broadcasting_index[1][i]] * grad_output[i];
            if(rhs->requires_grad) (rhs->grad)[broadcasting_index[1][i]] += (lhs->data)[broadcasting_index[0][i]] * grad_output[i];
        }
    }
};

class ReciprocalFunction : public Function {
    public:

    shared_ptr<Tensor> forward(shared_ptr<Tensor> input) {
        auto res = make_shared<Tensor>(input->shape, input->requires_grad);
        
        for(int i=0; i<res->total_size; i++) {
            // cout << i << " " << input->data[i] << endl;
            res->data[i] = 1 / input->data[i];
        }
        save_for_backward(input);
        // saved_tensors.push_back(input);

        return res;
    }

    void backward(shared_ptr<double[]> grad_output) override {
        auto input = saved_tensors[0];

        if(input->requires_grad) {
            for(int i=0; i<input->total_size; i++) {
                input->grad[i] -= pow((1/input->data[i]), 2) * grad_output[i];
            } 
        }
    }
};

shared_ptr<Tensor> reciprocal(shared_ptr<Tensor> input) {
    // auto grad_fn = shared_ptr<ReciprocalFunction>(); // 이렇게 하면 ve_for_backward 부분에서 segmentation fault가 뜸.
    auto grad_fn = make_shared<ReciprocalFunction>(); // 이렇게 해야 ReciprocalFunction 객체가 생성되고, 그 객체를 가리키는 pointer 객체가 만들어짐.
    auto res = grad_fn->forward(input);
    res->grad_fn = grad_fn;

    return res;
}

// Currently supports 2d only. To be updated later.
class Transpose : public Function {
    public:

    shared_ptr<Tensor> forward(shared_ptr<Tensor> input) {
        auto result = make_shared<Tensor>(vector<int>{input->shape[1], input->shape[0]}, input->requires_grad);

        for(int i=0; i<input->shape[0]; i++) {
            for(int j=0; j<input->shape[1]; j++) {
                result->data[result->get_flatten_idx<initializer_list<int>>({j, i})] = input->data[input->get_flatten_idx<initializer_list<int>>({i, j})];
            }
        }

        save_for_backward(input);

        return result;
    }

    void backward(shared_ptr<double[]> grad_output) override {
        auto parent = saved_tensors[0];
        if(parent->requires_grad){
            int s1 = parent->shape[0];
            int s2 = parent->shape[1];
    
            for(int k=0; k<s1*s2; k++) {
                int res_row_idx = k / s1; // res 의 행index
                int res_col_idx = k % s1; // res 의 열index
    
                parent->get_gradient_element({res_col_idx, res_row_idx}) += grad_output[k];
            }
        }
    }
};

shared_ptr<Tensor> transpose_(shared_ptr<Tensor> input) {
    auto transpose_fn = make_shared<Transpose>();
    auto res = transpose_fn->forward(input);
    res->grad_fn = transpose_fn;

    return res;
}

class MatrixMultiplication : public Function {
    public:

    vector<int> res_strides;

    shared_ptr<Tensor> forward(vector<shared_ptr<Tensor>> inputs) {
        auto lhs = inputs[0];
        // auto rhs = inputs[1];
        auto rhs_T = transpose_(inputs[1]);

        // for(int i=0; i<4; i++) cout << rhs_T->data[i] << " ";
        // cout << endl;

        save_for_backward(lhs);
        save_for_backward(rhs_T);

        vector<int> res_shape = lhs->shape;
        res_shape[res_shape.size()-1] = rhs_T->shape[0];
        auto res = make_shared<Tensor>(res_shape, lhs->requires_grad||rhs_T->requires_grad);
        saved_attrs.push_back(res->total_size);
        res_strides = res->strides;

        for(int i=0; i<res->total_size; i++) {
            // vector<int> index;
            int remaining = i;
            int lhs_idx = 0;
            int rhs_T_idx = 0;
            
            for(int j=0; j<res->shape.size()-1; j++) {
                lhs_idx += remaining/res->strides[j] * lhs->strides[j];
                // index.push_back(remaining/strides[j]);
                remaining %= res->strides[j];
            }
            rhs_T_idx = remaining;

            // cout << "i: " << i << endl;
            // cout << "lhs_idx: " << lhs_idx << endl;
            // cout << "rhs_T_idx: " << rhs_T_idx << endl << endl;
            

            for(int k=0; k<lhs->shape[lhs->shape.size()-1]; k++) {
                res->data[i] += lhs->data[lhs_idx+k] * rhs_T->data[rhs_T_idx*rhs_T->strides[0]+k];
            }
        }

        return res;
    }

    void backward(shared_ptr<double[]> grad_output) override {
        auto lhs = saved_tensors[0];
        auto rhs_T = saved_tensors[1];

        for(int i=0; i<saved_attrs[0]; i++) {
            int remaining = i;
            int lhs_idx = 0;
            int rhs_T_idx = 0;

            for(int j=0; j<lhs->shape.size()-1; j++) {
                lhs_idx += remaining/res_strides[j] * lhs->strides[j];
                remaining %= res_strides[j];
            }
            rhs_T_idx = remaining;

            for(int k=0; k<lhs->shape[lhs->shape.size()-1]; k++) {
                if(lhs->requires_grad) lhs->grad[lhs_idx+k] += rhs_T->data[rhs_T_idx*rhs_T->strides[0]+k] * grad_output[i];
                if(rhs_T->requires_grad) rhs_T->grad[rhs_T_idx*rhs_T->strides[0]+k] += lhs->data[lhs_idx+k] * grad_output[i];
            }
        }
    }
};

shared_ptr<Tensor> matmul(shared_ptr<Tensor> lhs, shared_ptr<Tensor> rhs) {
    auto grad_fn = make_shared<MatrixMultiplication>();
    auto res = grad_fn->forward(vector<shared_ptr<Tensor>>{lhs, rhs});
    res->grad_fn = grad_fn;
    return res;
}

class ExponentialFunction : public Function {
    public:

    shared_ptr<Tensor> forward(shared_ptr<Tensor> input) {
        auto result = make_shared<Tensor>(input->shape, input->requires_grad);
        save_for_backward(input);
        saved_attrs.push_back(input->total_size);

        for(int i=0; i<saved_attrs[0]; i++) {
            result->data[i] = exp(input->data[i]);
        }

        return result;
    }

    void backward(shared_ptr<double[]> grad_output) override {
        auto input = saved_tensors[0];
        if (input->requires_grad){
            for(int i=0; i<saved_attrs[0]; i++) {
                input->grad[i] += exp(input->data[i]) * grad_output[i];
            }
        }

    }
};

shared_ptr<Tensor> exp(shared_ptr<Tensor> input) {
    auto grad_fn = make_shared<ExponentialFunction>();
    auto result = grad_fn->forward(input);
    result->grad_fn = grad_fn;

    return result;
}

class LeakyReLUFunction : public Function {
    public:

    double slope;

    LeakyReLUFunction(double slope=0.2) : slope(slope) {}

    shared_ptr<Tensor> forward(shared_ptr<Tensor> input) {
        auto res = make_shared<Tensor>(input->shape, input->requires_grad);
        save_for_backward(input);

        for (int i=0; i<res->total_size; i++) {
            if(input->data[i] >= 0) {
                res->data[i] = input->data[i];
            } else {
                res->data[i] = slope * input->data[i];
            }
        }

        return res;
    }

    void backward(shared_ptr<double[]> grad_output) override {
        auto input = saved_tensors[0];

        if(input->requires_grad) {
            for(int i=0; i<input->total_size; i++) {
                if(input->data[i] >= 0) {
                    input->grad[i] += grad_output[i];
                } else {
                    input->grad[i] += slope * grad_output[i];
                }
            }
        }
    }
};

shared_ptr<Tensor> leaky_relu(double slope, shared_ptr<Tensor> input) {
    auto grad_fn = make_shared<LeakyReLUFunction>(slope);
    auto res = grad_fn->forward(input);
    res->grad_fn = grad_fn;

    return res;
}

// class ConcatFunction : public Function {
//     public:

//     int dim;
    
//     shared_ptr<Tensor> forward(vector<shared_ptr<Tensor>> inputs, int dim) : dim(dim) {
//         vector<int> res_shape;
//         bool requires_grad = false;

//         for(int i=0; i<inputs.size(); i++) {
//             if(inputs[i]->shape[dim] != 1) throw invalid_argument("shape[dim] must be 1!");
//             requires_grad = requires_grad || inputs[i].requires_grad;
//         }

//         for(int i=0; i<inputs[0]->shape.size(); i++) {
//             if(i == dim) {
//                 res_shape.push_back(inputs.size());
//             } else {
//                 res_shape.push_back(inputs[0]->shape[i]);
//             }
//         }
//         auto res = make_shared<Tensor>(res_shape, requires_grad);

//         return;
//     }
// };

// Cover only when dim=-1
class SoftmaxFunction : public Function {
    public:

    shared_ptr<Tensor> forward(shared_ptr<Tensor> input) {
        vector<int> sum_shape = input->shape;
        sum_shape[sum_shape.size()-1] = 1;

        int last_shape = input->shape[input->shape.size()-1];

        shared_ptr<Tensor> max_v = make_shared<Tensor>(vector<int>{1}, false);

        for(int i=0; i<input->total_size; i++) {
            if(input->data[i]>max_v->data[0]){
                max_v->data[0] = input->data[i];
            }
        }

        for(int i=0; i<input->total_size; i++) {
            input->data[i] -= max_v->data[0];
        }

        // auto subtracted_input = subtraction(input, max_v);

        // shared_ptr<Tensor> exp_input = exp(subtracted_input);
        shared_ptr<Tensor> exp_input = exp(input);
        shared_ptr<Tensor> sum_ = make_shared<Tensor>(sum_shape, false);
        shared_ptr<Tensor> res = make_shared<Tensor>(input->shape, input->requires_grad);

        for(int i=0; i<sum_->total_size; i++) {
            for(int j=0; j<last_shape; j++) {
                sum_->data[i] += exp_input->data[i*last_shape + j];
            }
        }

        // shared_ptr<Tensor> res = exp_input * reciprocal(sum_); // it couldn't be used for complexity of chain rule.

        for(int i=0; i<sum_->total_size; i++) {
            for(int j=0; j<last_shape; j++) {
                res->data[i*last_shape + j] = exp_input->data[i*last_shape+j]/sum_->data[i];
            }
        }
        
        save_for_backward(exp_input); // exp_input is parent role;
        save_for_backward(sum_);
        // save_for_backward(res);
        saved_attrs.push_back(last_shape);

        return res;
    }

    void backward(shared_ptr<double[]> grad_output) override {
        auto exp_input = saved_tensors[0];
        if (!exp_input->requires_grad) return;

        auto sum_ = saved_tensors[1];
        int last_shape = saved_attrs[0];

        for(int i=0; i<sum_->total_size; i++) {
            for(int j=0; j<last_shape; j++) {
                exp_input->grad[i*last_shape+j] += (sum_->data[i] - exp_input->data[i*last_shape+j])/pow(sum_->data[i], 2.0) * grad_output[i*last_shape + j];

                for(int k=0; k<last_shape; k++) {
                    if(k!=j) {
                        exp_input->grad[i*last_shape+j] -= exp_input->data[i*last_shape+k]/pow(sum_->data[i], 2.0) * grad_output[i*last_shape+k];
                    }
                }
                if(isnan(exp_input->grad[i*last_shape+j])) {
                    for(int k=0; k<last_shape; k++) {
                        auto a = exp_input->data[i*last_shape+k]/pow(sum_->data[i], 2.0) * grad_output[i*last_shape + k];
                        if (isnan(a)) {
                            cout << endl << "exp_input->data[i*last_shape+k] " << exp_input->data[i*last_shape+k] << endl << pow(sum_->data[i], 2.0) << endl << grad_output[i*last_shape + k] << endl;
                        }
                    }
                    cout << endl;
                }
            }
        }
    }
};

// class SoftmaxFunction : public Function {
//     public:

//     shared_ptr<Tensor> forward(shared_ptr<Tensor> input) {
//         vector<int> sum_shape = input->shape;
//         sum_shape[sum_shape.size()-1] = 1;

//         int last_shape = input->shape[input->shape.size()-1];


//     }
// }

shared_ptr<Tensor> softmax(shared_ptr<Tensor> input) {
    auto grad_fn = make_shared<SoftmaxFunction>();
    auto res = grad_fn->forward(input);
    res->grad_fn = grad_fn;

    return res;
}


shared_ptr<Tensor> create_constant(double value) {
    auto c = make_shared<Tensor>(vector<int>{1}, false);
    c->data[0] = value;

    return c;
}

void build_topo(shared_ptr<Tensor> v, vector<shared_ptr<Tensor>>& topo_list, set<shared_ptr<Tensor>>& visited) {
    if (v == nullptr || visited.count(v) > 0) return;

    visited.insert(v);

    if(v->grad_fn) {
        if(v->grad_fn->saved_tensors.size() > 0 && v->grad_fn->saved_tensors[0]->requires_grad) build_topo(v->grad_fn->saved_tensors[0], topo_list, visited);
        if(v->grad_fn->saved_tensors.size() > 1 && v->grad_fn->saved_tensors[1]->requires_grad) build_topo(v->grad_fn->saved_tensors[1], topo_list, visited);
    }

    topo_list.push_back(v);
}

void backward(shared_ptr<Tensor> t) {
    
    for(int i = 0; i < t->total_size; i++) {
        t->grad[i] = 1.0;
    }
    // for(int i=0; i<12; i++) {
    //     if(i%4==0) {t->grad[i] = 1.0;}
    //     else {t->grad[i] = 0.0;}
    // }

    vector<shared_ptr<Tensor>> topo_list;
    set<shared_ptr<Tensor>> visited;
    
    build_topo(t, topo_list, visited);
    
    for(int i = topo_list.size()-1; i >= 0; i--) {
        if(topo_list[i]->grad_fn == nullptr) continue;
        // cout << topo_list[i]->grad_fn->saved_tensors.size() << endl;
        topo_list[i]->grad_fn->backward(topo_list[i]->grad);
    }

    for (auto& t: topo_list) {
        if(t->grad_fn) {
            t->grad_fn->saved_tensors.clear();
        }
    }
    
}


void is_same_size(shared_ptr<Tensor> lhs, shared_ptr<Tensor> rhs) {
    if((lhs->shape).size() != (rhs->shape).size()) {
        throw invalid_argument("Dimension mismatch!!");
    }

    for(int i=0; i<(lhs->shape).size(); i++) {
        if ((lhs->shape)[i] != (rhs->shape)[i]) {
            throw invalid_argument("size mismatch at dimension " + to_string(i));
        }
    }
}

shared_ptr<Tensor> operator+(shared_ptr<Tensor> lhs, shared_ptr<Tensor> rhs) {
    auto grad_fn = make_shared<Add>();
    auto res = grad_fn->forward(vector<shared_ptr<Tensor>>{lhs, rhs});
    res->grad_fn = grad_fn;
    
    return res;
}

shared_ptr<Tensor> operator-(shared_ptr<Tensor> lhs, shared_ptr<Tensor> rhs) {
    auto grad_fn = make_shared<Subtraction>();
    auto res = grad_fn->forward(vector<shared_ptr<Tensor>>{lhs, rhs});
    res->grad_fn = grad_fn;
    
    return res;
}

shared_ptr<Tensor> operator*(shared_ptr<Tensor> lhs, shared_ptr<Tensor> rhs) {
    auto grad_fn = make_shared<Multiplication>();
    auto res = grad_fn->forward(vector<shared_ptr<Tensor>>{lhs, rhs});
    res->grad_fn = grad_fn;
    
    return res;
}

shared_ptr<Tensor> C1 = create_constant(1);
shared_ptr<Tensor> C2 = create_constant(-1);
shared_ptr<Tensor> EPS = create_constant(1e-7);

shared_ptr<Tensor> sigmoid(shared_ptr<Tensor> input) {
    return reciprocal(C1 + exp(C2 * input));
}

// shared_ptr<Tensor> softmax(shared_ptr<Tensor> input) {
//     shared_ptr<Tensor> res = make_shared<Tensor>(input->shape, input->requires_grad);
//     int size = (res->shape).size();

//     int sum_shape = vector<int>{};
//     for(int i=0; i<size-1; i++) {
//         sum_shape.push_back(shape[i]);
//     }
//     sum_shape.push_back(1);

//     shared_ptr<Tensor> sum_ = make_shared<Tensor>(sum_shape, input->requires_grad);
    
    
// }

class Layer {
    protected:
    bool requires_grad = true;
    
    public:
    vector<shared_ptr<Tensor>> parameters;
    virtual ~Layer() = default;
};

class FCLayer : public Layer {
    public:

    int in_features, out_features;
    bool bias = true;

    FCLayer(int in_features, int out_features, bool bias=true, bool requires_grad_=true) : in_features(in_features), out_features(out_features), bias(bias) {
        requires_grad = requires_grad_; 
        shared_ptr<Tensor> weight = make_shared<Tensor>(vector<int>{in_features, out_features}, requires_grad);
        parameters.push_back(weight);

        if (bias) {
            shared_ptr<Tensor> b = make_shared<Tensor>(vector<int>{out_features}, requires_grad);
            parameters.push_back(b);
        }
    }

    shared_ptr<Tensor> forward(shared_ptr<Tensor> input) {
        shared_ptr<Tensor> res = matmul(input, parameters[0]);
        if (bias) res = res + parameters[1];

        return res;
    }
};

class LeakyReLULayer : public Layer {
    public:

    double slope;

    LeakyReLULayer(double slope=0.2) : slope(slope) {}

    shared_ptr<Tensor> forward(shared_ptr<Tensor> input) {
        return leaky_relu(slope, input);
    }
};

class SoftmaxLayer : public Layer {
    public:

    shared_ptr<Tensor> forward(shared_ptr<Tensor> input) {
        return softmax(input);
    }
};

random_device rd;
mt19937 gen(rd());

void xavier_initialization(shared_ptr<Tensor> params, int n_in, int n_out) {
    double mu = 0.0;
    double sigma = sqrt(2.0 / (n_in + n_out));
    normal_distribution<double> dist(mu, sigma);

    for(int i=0; i<params->total_size; i++) {
        params->data[i] = dist(gen);
    }
}

void he_initialization(shared_ptr<Tensor> params, int n_in) {
    long double mu = 0.0;
    long double sigma = sqrt(2.0/n_in);
    normal_distribution<double> dist(mu, sigma);

    for(int i=0; i<params->total_size; i++) {
        params->data[i] = dist(gen);
    }
}

class Optimization {
    public:

    vector<shared_ptr<Tensor>> params;
    vector<shared_ptr<Tensor>> momentums;

    void add_parameters(shared_ptr<Tensor> parameters) {
        this->params.push_back(parameters);
        auto temp = make_shared<Tensor>(parameters->shape, false);
        temp->fill(0.0);
        momentums.push_back(temp);
    }
};

class SGDOptimization : public Optimization {
    public:

    double lr;
    double m;

    SGDOptimization(double lr=0.001, double m=0.9) {
        this->lr = lr;
        this->m = m;
    }

    void step() {

        // double total_norm = 0.0;
        // int total_size_p = 0;
        // for(auto& p : params) {
        //     for(int i=0; i<p->total_size; i++) {
        //         if(p->requires_grad) total_norm += p->grad[i] * p->grad[i];
        //     }
        //     total_size_p += p->total_size;
        // }

        // total_norm /= total_size_p;
        // total_norm = sqrt(total_norm);

        // double clip_threshold = 1.0;
        // // cout << endl << total_norm << endl; 
        // if (isnan(total_norm)) {
        //     cout << endl << "total_norm is -nan!!" << endl;
        // }

        // if (total_norm >= clip_threshold) {
        //     // cout << endl << true << endl;
        //     double scale = clip_threshold / (total_norm + 1e-6);
        //     for(auto& p : params) {
        //         for(int i=0; i<p->total_size; i++) {
        //             if(p->requires_grad) p->grad[i] *= scale;
        //         }
        //     }
        // }

        for(int i=0; i<params.size(); i++) {
            if (params[i]->requires_grad == false) continue;

            for(int j=0; j < params[i]->total_size; j++) {

                momentums[i]->data[j] = m * momentums[i]->data[j] + lr * params[i]->grad[j];
                params[i]->data[j] -= momentums[i]->data[j];
            }
        }
    }

    void zero_grad() {

        for(int i=0; i < params.size(); i++) {
            if(params[i]->requires_grad == false) continue;

            for(int j=0; j < params[i]->total_size; j++) {
                params[i]->grad[j] = 0.0;
            }
        }
    }
};

class MSELoss {
    public:
}; 

class CrossEntropyLoss : public Function {
    public:


    shared_ptr<Tensor> forward(shared_ptr<Tensor> inputs, shared_ptr<Tensor> labels) {
        vector<int> res_shape;
        int class_n = inputs->shape[inputs->shape.size()-1];
        saved_attrs.push_back(class_n);

        save_for_backward(inputs);
        save_for_backward(labels);

        for(int i=0; i<inputs->shape.size()-1; i++) {
            res_shape.push_back(inputs->shape[i]);
        }

        if (res_shape.size() == 0) {
            res_shape.push_back(1);
        }

        auto res = make_shared<Tensor>(res_shape, inputs->requires_grad);
        saved_attrs.push_back(res->total_size);

        for(int i=0; i<res->total_size; i++) {
            res->data[i] = 0.0;
            for(int j=0; j<class_n; j++) {
                res->data[i] -= labels->data[i*class_n+j] * log(inputs->data[i*class_n+j] + 1e-5);
            }
        }

        return res;
    }

    void backward(shared_ptr<double []> grad_output) {
        int class_n = saved_attrs[0];
        int total_size = saved_attrs[1];
        
        auto inputs = saved_tensors[0];
        auto labels = saved_tensors[1];

        for(int i=0; i<total_size; i++) {
            for(int j=0; j<class_n; j++) {
                inputs->grad[i*class_n+j] -= labels->data[i*class_n+j]/(inputs->data[i*class_n+j]+1e-7) * grad_output[i];
            }
        }
    }
};

shared_ptr<Tensor> cross_entropy_loss(shared_ptr<Tensor> inputs, shared_ptr<Tensor> labels) {
    auto loss_fn = make_shared<CrossEntropyLoss>();
    auto res = loss_fn->forward(inputs, labels);
    res->grad_fn = loss_fn;

    return res;
}

#include "mnist/include/mnist/mnist_reader.hpp"

int main() {
    int batch_size = 1; // restricted
    int img_size = 28*28;    
    int epoch = 1000;
    int iters = 60000;
    double *train_acc = new double[epoch+5];
    vector<double> train_loss;
    double *test_acc = new double[epoch+5];

    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

    // model structure
    // fc1 - FC Layer: 28*28 -> 28
    // act1 - LeakyReLU(0.2)
    // fc2 - FC Layer: 28 -> 10
    // act2 - LeakyReLU(0.2)
    // act3 - Softmax()

    auto fc1 = FCLayer(28*28, 28, true, true);
    he_initialization(fc1.parameters[0], 28*28); // weight initialization by he
    fc1.parameters[1]->fill(0.0); // bias initialization to zero

    auto act1 = LeakyReLULayer(0.2);

    auto fc2 = FCLayer(28, 10, true, true);
    he_initialization(fc2.parameters[0], 28); // weight initialization by he
    fc2.parameters[1]->fill(0.0); // bias initialization to zero

    auto act2 = LeakyReLULayer(0.2);

    // auto fc3 = FCLayer(28, 10, true, true);
    // he_initialization(fc3.parameters[0], 28);
    // fc3.parameters[1]->fill(0.0);

    // auto act3 = LeakyReLULayer(0.2);

    auto act3 = SoftmaxLayer();

    auto optim = SGDOptimization(0.000001, 0.90);
    optim.add_parameters(fc1.parameters[0]);
    optim.add_parameters(fc1.parameters[1]);
    optim.add_parameters(fc2.parameters[0]);
    optim.add_parameters(fc2.parameters[1]);
    // optim.add_parameters(fc3.parameters[0]);
    // optim.add_parameters(fc3.parameters[1]);

    random_device rd2;
    mt19937 gen2(rd2());
    uniform_int_distribution<> distr(0, 60000-1);

    for(int i=0; i < epoch; i++) {
        for(int j=0; j < iters; j++) {
            auto x = make_shared<Tensor>(vector<int>{img_size}, false);
            auto y = make_shared<Tensor>(vector<int>{10}, false);
            y->fill(0.0);

            int rand_idx = distr(gen2);

            auto x_ = dataset.training_images[rand_idx];
            auto y_label = dataset.training_labels[rand_idx];
            
            for(int k=0; k<img_size; k++) {
                x->data[k] = x_[k]/255.0;
            }

            y->data[y_label] = 1.0;

            // x->print();
            // cout << endl << endl;
            // y->print();
            // cout << endl << endl;

            auto x2 = fc1.forward(x);
            auto x3 = act1.forward(x2);
            auto x4 = fc2.forward(x3);
            auto x5 = act2.forward(x4);
            // auto x6 = fc3.forward(x5);
            // auto x7 = act3.forward(x6);
            auto pred = act3.forward(x5);

            // fc1.parameters[0]->print();
            // x2->print();
            // cout << endl << endl;
            // x3->print();
            // cout << endl;
            // x4->print();
            // cout << endl;
            // x5->print();
            // cout << endl;
            // x6->print();
            // cout << endl;
            // x7->print();
            // cout << endl;
            // pred->print();
            // cout << endl;

            auto loss = cross_entropy_loss(pred, y);
            // loss->print();
            // cout << endl;
            
            optim.zero_grad();
            backward(loss);

            for (auto &p : optim.params) {
                for (int i=0; i < p->total_size; i++) {
                    if (isnan(p->grad[i])) {
                        break;
                    }
                }
            }
            
            optim.step();


            train_loss.push_back(loss->data[0]);
            pred->print();
            cout << endl << i << " " << j << " " << loss->data[0] << endl;
        }
    }

    return 0;
}