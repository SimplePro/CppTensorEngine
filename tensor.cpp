#include <iostream>
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <memory>
#include <set>
#include <algorithm>

using namespace std;

class Tensor;

// Function 클래스 코드 이해 부족.
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
    int total_size;

    shared_ptr<Tensor> forward(vector<shared_ptr<Tensor>> inputs) override {
        auto lhs = inputs[0];
        auto rhs = inputs[1];

        save_for_backward(lhs);
        save_for_backward(rhs);

        broadcasting_index = get_broadcasting_index(lhs->shape, lhs->strides, rhs->shape, rhs->strides);

        auto res = make_shared<Tensor>(broadcasting_index[2], lhs->requires_grad || rhs->requires_grad);
        total_size = res->total_size;

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

        for(int i = 0; i < total_size; i++) {
            if(lhs->requires_grad) (lhs->grad)[broadcasting_index[0][i]] += grad_output[i];
            if(rhs->requires_grad) (rhs->grad)[broadcasting_index[1][i]] += grad_output[i];
        }
    }
};

class Subtraction : public Function {
    public:

    vector<vector<int>> broadcasting_index;
    int total_size;

    shared_ptr<Tensor> forward(vector<shared_ptr<Tensor>> inputs) override {
        auto lhs = inputs[0];
        auto rhs = inputs[1];

        save_for_backward(lhs);
        save_for_backward(rhs);

        broadcasting_index = get_broadcasting_index(lhs->shape, lhs->strides, rhs->shape, rhs->strides);

        auto res = make_shared<Tensor>(broadcasting_index[2], lhs->requires_grad || rhs->requires_grad);
        total_size = res->total_size;

        // auto res = make_shared<Tensor>(lhs->shape, lhs->requires_grad || rhs->requires_grad);
        for(int i=0; i < total_size; i++) {
            (res->data)[i] = (lhs->data)[broadcasting_index[0][i]] - (rhs->data)[broadcasting_index[1][i]];
            // (res->data)[i] = (lhs->data)[i] - (rhs->data)[i];
        }

        return res;
    }

    void backward(shared_ptr<double[]> grad_output) override {
        auto lhs = saved_tensors[0];
        auto rhs = saved_tensors[1];

        for(int i=0; i < total_size; i++) {
            if(lhs->requires_grad) (lhs->grad)[broadcasting_index[0][i]] += grad_output[i];
            if(rhs->requires_grad) (rhs->grad)[broadcasting_index[1][i]] -= grad_output[i];
        }
    }
};

class Multiplication : public Function {
    public:

    vector<vector<int>> broadcasting_index;
    int total_size;

    shared_ptr<Tensor> forward(vector<shared_ptr<Tensor>> inputs) override {
        auto lhs = inputs[0];
        auto rhs = inputs[1];

        save_for_backward(lhs);
        save_for_backward(rhs);

        broadcasting_index = get_broadcasting_index(lhs->shape, lhs->strides, rhs->shape, rhs->strides);

        auto res = make_shared<Tensor>(broadcasting_index[2], lhs->requires_grad || rhs->requires_grad);
        total_size = res->total_size;

        for(int i = 0; i < total_size; i++) {
            (res->data)[i] = (lhs->data)[broadcasting_index[0][i]] * (rhs->data)[broadcasting_index[1][i]];
        }

        return res;
    }

    void backward(shared_ptr<double[]> grad_output) override {
        auto lhs = saved_tensors[0];
        auto rhs = saved_tensors[1];

        for(int i = 0; i < total_size; i++) {
            if(lhs->requires_grad) (lhs->grad)[broadcasting_index[0][i]] += (rhs->data)[broadcasting_index[1][i]] * grad_output[i];
            if(rhs->requires_grad) (rhs->grad)[broadcasting_index[1][i]] += (lhs->data)[broadcasting_index[0][i]] * grad_output[i];
        }
    }
};

class MatrixMultiplication : public Function {
    public:

    shared_ptr<Tensor> forward(vector<shared_ptr<Tensor>> inputs) override {
        auto lhs = inputs[0];
        auto rhs = inputs[1];

        save_for_backwards(lhs);
        save_for_backwards(rhs);

        
    }
}

void build_topo(shared_ptr<Tensor> v, vector<shared_ptr<Tensor>>& topo_list, set<shared_ptr<Tensor>>& visited) {
    if (v == nullptr || visited.count(v) > 0) return;

    visited.insert(v);

    if(v->grad_fn) {
        if(v->grad_fn->saved_tensors.size() > 0) build_topo(v->grad_fn->saved_tensors[0], topo_list, visited);
        if(v->grad_fn->saved_tensors.size() > 1) build_topo(v->grad_fn->saved_tensors[1], topo_list, visited);
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


int main() {
    // auto broadcasting_index = get_broadcasting_index(vector<int>{2, 3, 1, 2}, vector<int> {6, 2, 2, 1}, vector<int>{1, 4, 2}, vector<int>{8, 2, 1});

    shared_ptr<Tensor> a = make_shared<Tensor>(vector<int>({3, 3}));
    for(int i=0; i<9; i++) a->data[i] = i-4;
    shared_ptr<Tensor> b = make_shared<Tensor>(vector<int>({1, 3}));
    shared_ptr<Tensor> c = make_shared<Tensor>(vector<int>({3, 3}));
    shared_ptr<Tensor> d = make_shared<Tensor>(vector<int>({3, 1}));

    // a->fill(2);
    b->fill(3);
    b->data[2] = 2;
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

    shared_ptr<Tensor> e = a*b + a*b*c*d;
    shared_ptr<Tensor> f = e*e + c*d;

    f->print();
    cout << endl;
    // cout << endl << e-> << endl;

    backward(f);

    cout << "a grad: " << a->grad[0] << endl;
    cout << "b grad: " << b->grad[0] << endl;
    cout << "c grad: " << c->grad[0] << endl;
    cout << "d grad: " << d->grad[0] << endl;
    cout << "e grad: " << e->grad[0] << endl;
    cout << "f grad: " << f->grad[0] << endl;

    return 0;
}