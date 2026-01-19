#include <iostream>
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <memory>

using namespace std;

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

    shared_ptr<Tensor> left_parent = nullptr;
    shared_ptr<Tensor> right_parent = nullptr;

    shared_ptr<double[]> grad = nullptr;
    string op = "";
    
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

        is_same_size(lhs, rhs);

        auto res = make_shared<Tensor>(lhs->shape); // (*lhs).shape 과 동일
        
        for(int i=0; i<lhs->total_size; i++) {
            (res->data)[i] = (lhs->data)[i] + (rhs->data)[i];
        }

        res->left_parent = lhs;
        res->right_parent = rhs;

        res->op = "add";

        return res;

    }

    shared_ptr<Tensor> operator*(shared_ptr<Tensor> lhs, shared_ptr<Tensor> rhs) {

        is_same_size(lhs, rhs);

        auto res = make_shared<Tensor>(lhs->shape);

        for(int i=0; i<lhs->total_size; i++) {
            (res->data)[i] = (lhs->data)[i] * (rhs->data)[i];
        }

        res->left_parent = lhs;
        res->right_parent = rhs;

        res->op = "multiply";

        return res;
    }

};

int main() {
    Tensor tensor({5, 5, 3});
    tensor.fill(5);
    // tensor.print();

    cout << tensor({4, 1, 2}) << endl;
    tensor({4, 4, 2}) = 1;
    // tensor.print();

    tensor.reshape({5*5, 3});
    // tensor.print();

    return 0;
}