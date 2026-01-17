#include <iostream>
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <string>

using namespace std;

class Tensor {
    public:

    vector<int> shape; // Tensor의 shape 저장
    vector<int> strides; // Tensor 를 flattening 해서 사용하는데, indexing 할때 필요한 변수
    int total_size = 1; // Tensor의 total_size
    double* data; // Tensor의 값

    Tensor (vector<int> s) : shape(s) {
        for(int dim : shape) {
            total_size *= dim;
        }

        strides.resize(shape.size());

        strides[shape.size()-1] = 1;

        for(int i=shape.size()-2; i >= 0; i--) { 
            strides[i] = strides[i+1]*shape[i+1];
        }

        data = new double[total_size];
    }

    ~Tensor() {
        delete[] data;
    }

    void fill(int value) {
        for(int i = 0; i < total_size; i++) {
            data[i] = value;
        }
    }

    void print() {
        for(int i = 0; i < total_size; i++) {
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

int main() {
    Tensor tensor({512, 512, 3});
    tensor.fill(5);
    // tensor.print();

    // cout << tensor({5, 1, 2}) << endl;
    tensor({511, 511, 2}) = 1;
    // tensor.print();

    tensor.reshape({512*512, 3});

    return 0;
}