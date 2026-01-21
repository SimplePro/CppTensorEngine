#include <iostream>
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <memory>
#include <set>

using namespace std;

class Tensor;

class Function {
    public:

    vector<shared_ptr<Tensor>> saved_tensors; // 공통 규칙 saved_tensors[0]: left_parent, saved_tensors[1]: right_parent
    vector<int> saved_attrs;

    virtual ~Function() = default;

    virtual shared_ptr<Tensor> forward(vector<shared_ptr<Tensor>> inputs) = 0;
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

    shared_ptr<double[]> grad = nullptr;
    // string op = "";
    shared_ptr<Function> grad_fn = nullptr;
    
    bool requires_grad = true;

    Tensor (vector<int> s) : shape(s) {
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

        if (requires_grad) {
            grad = shared_ptr<double[]>(new double[total_size]);
            for(int i = 0; i < total_size; i++) {
                grad[i] = 0.0;
            }
        }
        // data = make_shared<double[]>(total_size);
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

    double& operator() (initializer_list<int> index) {

        // 디버깅할 때만 사용
        if (index.size() != shape.size()) {
            throw out_of_range("Dimension mismatch!");
        }

        int idx = 0;
        for (int i=0; i<shape.size(); i++) {
            // 디버깅할 때만 사용
            if (*(index.begin() + i) < 0 || *(index.begin() + i) >= shape[i]) {
                throw out_of_range("Index out of bounds at dimension " + to_string(i));
            }
            idx += *(index.begin() + i) * strides[i];

        }

        // cout << idx << endl;

        return data[idx];
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

};

class Add : public Function {
    public:
    shared_ptr<Tensor> forward(vector<shared_ptr<Tensor>> inputs) override {
        auto lhs = inputs[0];
        auto rhs = inputs[1];

        save_for_backward(lhs);
        save_for_backward(rhs);

        auto res = make_shared<Tensor>(lhs->shape);
        for(int i=0; i<lhs->total_size; i++) {
            (res->data)[i] = (lhs->data)[i] + (rhs->data)[i];
        }

        return res;
    }

    void backward(shared_ptr<double[]> grad_output) override {
        auto lhs = saved_tensors[0];
        auto rhs = saved_tensors[1];

        for(int i = 0; i < lhs->total_size; i++) {
            (lhs->grad)[i] += grad_output[i];
            (rhs->grad)[i] += grad_output[i];
        }
    }
};

class Subtraction : public Function {
    public:
    shared_ptr<Tensor> forward(vector<shared_ptr<Tensor>> inputs) override {
        auto lhs = inputs[0];
        auto rhs = inputs[1];

        save_for_backward(lhs);
        save_for_backward(rhs);

        auto res = make_shared<Tensor>(lhs->shape);
        for(int i=0; i < lhs->total_size; i++) {
            (res->data)[i] = (lhs->data)[i] - (rhs->data)[i];
        }

        return res;
    }

    void backward(shared_ptr<double[]> grad_output) override {
        auto lhs = saved_tensors[0];
        auto rhs = saved_tensors[1];

        for(int i=0; i < lhs->total_size; i++) {
            (lhs->grad)[i] += grad_output[i];
            (rhs->grad)[i] -= grad_output[i];
        }
    }
};

class Multiplication : public Function {
    public:
    shared_ptr<Tensor> forward(vector<shared_ptr<Tensor>> inputs) override {
        auto lhs = inputs[0];
        auto rhs = inputs[1];

        save_for_backward(lhs);
        save_for_backward(rhs);

        auto res = make_shared<Tensor>(lhs->shape);
        for(int i = 0; i < lhs->total_size; i++) {
            (res->data)[i] = (lhs->data)[i] * (rhs->data)[i];
        }

        return res;
    }

    void backward(shared_ptr<double[]> grad_output) override {
        auto lhs = saved_tensors[0];
        auto rhs = saved_tensors[1];

        for(int i = 0; i < lhs->total_size; i++) {
            (lhs->grad)[i] += (rhs->data)[i] * grad_output[i];
            (rhs->grad)[i] += (lhs->data)[i] * grad_output[i];
        }
    }
};

void build_topo(shared_ptr<Tensor> v, vector<shared_ptr<Tensor>>& topo_list, set<shared_ptr<Tensor>>& visited) {
    if (v == nullptr || visited.count(v) > 0) return;

    visited.insert(v);

    if(v->grad_fn) {
        build_topo(v->grad_fn->saved_tensors[0], topo_list, visited);
        build_topo(v->grad_fn->saved_tensors[1], topo_list, visited);
    }

    topo_list.push_back(v);
}

void backward(shared_ptr<Tensor> t) {
    
    for(int i = 0; i < t->total_size; i++) {
        t->grad[i] = 1.0;
    }
    
    vector<shared_ptr<Tensor>> topo_list;
    set<shared_ptr<Tensor>> visited;
    
    build_topo(t, topo_list, visited);
    
    for(int i = topo_list.size()-1; i >= 0; i--) {
        if(topo_list[i]->grad_fn == nullptr) continue;
        topo_list[i]->grad_fn->backward(topo_list[i]->grad);
        // if(topo_list[i]->left_parent == nullptr || topo_list[i]->right_parent == nullptr) continue;
        // if(topo_list[i]->op == "add") {
        //     for(int j = 0; j < topo_list[i]->total_size; j++) {
        //         (topo_list[i]->left_parent->grad)[j] += (topo_list[i]->grad)[j];
        //         (topo_list[i]->right_parent->grad)[j] += (topo_list[i]->grad)[j];
        //     }
        // }
        // else if(topo_list[i]->op == "subtraction") {
        //     for(int j = 0; j < topo_list[i]->total_size; j++) {
        //         (topo_list[i]->left_parent->grad)[j] += (topo_list[i]->grad)[j];
        //         (topo_list[i]->right_parent->grad)[j] -= (topo_list[i]->grad)[j];
        //     }
        // }
        // else if(topo_list[i]->op == "multiplication") {
        //     for(int j = 0; j < topo_list[i]->total_size; j++) {
        //         (topo_list[i]->left_parent->grad)[j] += (topo_list[i]->grad)[j] * (topo_list[i]->right_parent->data)[j];
        //         (topo_list[i]->right_parent->grad)[j] += (topo_list[i]->grad)[j] * (topo_list[i]->left_parent->data)[j];
        //     }
        // }
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
    // auto res = make_shared<Tensor>(lhs->shape);
    // res->grad_fn = make_shared<Add>();
    auto res = grad_fn->forward(vector<shared_ptr<Tensor>>{lhs, rhs});
    res->grad_fn = grad_fn;
    // res->data = res->grad_fn->forward(vector<shared_ptr<Tensor>>{lhs, rhs})->data;

    return res;
}

shared_ptr<Tensor> operator-(shared_ptr<Tensor> lhs, shared_ptr<Tensor> rhs) {
    // auto res = make_shared<Tensor>(lhs->shape);
    // res->grad_fn = make_shared<Subtraction>();
    // res->data = res->grad_fn->forward(vector<shared_ptr<Tensor>>{lhs, rhs})->data;
    auto grad_fn = make_shared<Subtraction>();
    auto res = grad_fn->forward(vector<shared_ptr<Tensor>>{lhs, rhs});
    res->grad_fn = grad_fn;
    
    return res;
}

shared_ptr<Tensor> operator*(shared_ptr<Tensor> lhs, shared_ptr<Tensor> rhs) {
    // auto res = make_shared<Tensor>(lhs->shape);
    // res->grad_fn = make_shared<Multiplication>();
    // res->data = res->grad_fn->forward(vector<shared_ptr<Tensor>>{lhs, rhs})->data;
    auto grad_fn = make_shared<Multiplication>();
    auto res = grad_fn->forward(vector<shared_ptr<Tensor>>{lhs, rhs});
    res->grad_fn = grad_fn;

    return res;
}

// shared_ptr<Tensor> operator+(shared_ptr<Tensor> lhs, shared_ptr<Tensor> rhs) {
    
//     is_same_size(lhs, rhs);

//     auto res = make_shared<Tensor>(lhs->shape); // (*lhs).shape 과 동일

//     for(int i=0; i<lhs->total_size; i++) {
//         (res->data)[i] = (lhs->data)[i] + (rhs->data)[i];
//     }

//     res->requires_grad = (lhs->requires_grad || rhs->requires_grad);

//     if(res->requires_grad) {
//         res->left_parent = lhs;
//         res->right_parent = rhs;

//         res->op = "add";
//     }    

//     return res;

// }

// shared_ptr<Tensor> operator-(shared_ptr<Tensor> lhs, shared_ptr<Tensor> rhs) {
//     is_same_size(lhs, rhs);

//     auto res = make_shared<Tensor>(lhs->shape);

//     for(int i=0; i<(lhs->total_size); i++) {
//         (res->data)[i] = (lhs->data)[i] - (rhs->data)[i];
//     }

//     res->requires_grad = (lhs->requires_grad || rhs->requires_grad);

//     if(res->requires_grad) {
//         res->left_parent = lhs;
//         res->right_parent = rhs;
        
//         res->op = "subtraction";
//     }    
    
//     return res;
// }

// shared_ptr<Tensor> operator*(shared_ptr<Tensor> lhs, shared_ptr<Tensor> rhs) {

//     is_same_size(lhs, rhs);

//     auto res = make_shared<Tensor>(lhs->shape);

//     for(int i=0; i<lhs->total_size; i++) {
//         (res->data)[i] = (lhs->data)[i] * (rhs->data)[i];
//     }

//     res->requires_grad = (lhs->requires_grad || rhs->requires_grad); // 만약 부모중에 학습이 필요한 부모가 있다면 grad가 흘러야함.

//     if(res->requires_grad) {
//         res->left_parent = lhs;
//         res->right_parent = rhs;

//         res->op = "multiplication";
//     }    

//     return res;
// }


int main() {
    shared_ptr<Tensor> a = make_shared<Tensor>(vector<int>({3, 3}));
    shared_ptr<Tensor> b = make_shared<Tensor>(vector<int>({3, 3}));
    shared_ptr<Tensor> c = make_shared<Tensor>(vector<int>({3, 3}));
    shared_ptr<Tensor> d = make_shared<Tensor>(vector<int>({3, 3}));

    a->fill(2);
    b->fill(3);
    c->fill(-2);
    d->fill(-1);

    a->print();
    cout << endl << endl;
    b->print();
    cout << endl << endl;
    c->print();
    cout << endl << endl;
    d->print();
    cout << endl << endl;

    shared_ptr<Tensor> e = a*b - c*d + a*b*c*d;

    e->print();
    cout << endl;
    // cout << endl << e-> << endl;

    backward(e);

    cout << "a grad: " << a->grad[2] << endl;
    cout << "b grad: " << b->grad[0] << endl;
    cout << "c grad: " << c->grad[0] << endl;
    cout << "d grad: " << d->grad[0] << endl;
    cout << "e grad: " << e->grad[0] << endl;

    return 0;
}