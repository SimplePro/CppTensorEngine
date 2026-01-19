2026/01/16
1. 프로젝트의 기본적 틀 설계 (NOTES.md)
2. tensor 자료형 설계
- 데이터를 flattening 해서 관리 -> 사용자가 입력할 shape에 따라 데이터의 크기가 달라지는데, 이를 flattening해서 관리하면 더욱 쉬움. (indexing을 stride를 활용해서 연산하면 쉬움.) reshape도 stride만 바꿔주면 되어서 연산크기가 거의 0. 
- 동적할당을 통해서 관리해야 함. 동적할당을 사용하면 변수 크기만큼의 배열을 다룰 수 있음. (int arr[total_size]; 가 에러가 나는 이유: 컴파일러가 total_size는 컴파일을 다 완료해야만 알 수 있는 값인데, 예약해둬야 할 메모리 크기를 compile time에 알 수 없어서 에러 발생). 동적 할당을 사용하면 크기가 작은 stack이 아니라 거의 RAM 전체 영역을 쓸 수 있는 heap 영역에 데이터가 저장됨. stack은 크기가 보통 몇MB 정도임. 정적 할당의 경우 stack에 저장됨. 또한 동적할당을 copy할때는 deepcopy를 하는 것이 안전하다. 단순히 대입연산자를 사용하면 두 변수가 같은 주소를 참조하여 둘 중 하나의 값이 변하면 다른 값이 변하거나 둘 중 한 변수가 deallocation되면 다른 변수도 쓰지 못하게 되는 상황이 발생할 수 있음. (참고: new 와 delete를 짝을 맞춰줘야 함. new int[] -> delete[] , new -> delete . 만약 new int[] -> delete 로 하면 대참사가 일어남. 메모리 누수 발생 위험)
-> double* data = new double[total_size]; : double* data (data가 포인터 변수임) , new double[total_size]; :(sizeof(double) * total_size 만큼의 메모리를 요청하고 해당 메모리 주소의 첫번째 값을 반환).
(역참조연산자 * 는 포인터변수의 그 장소로 접근하는 것 자체를 의미하는 것에 가깝다.)
(동적 변수는 memory leaking을 고려해야 한다. 따라서 ~Tensor { delete[] data; }; 를 해줘야 함. (참고: 클래스 내부에서 멤버 변수에 접근할 때 this-> 를 생략해도 됨. 보통은 변수 이름이 겹칠 때만 명시적으로 this-> 를 사용함)) 
(flattening 된 변수의 indexing에 대하여 shape = (A_1, A_2, .. , A_N) 이고, data[a_1, a_2, .. , a_N] 을 할 때는
int index = 0;
for (int i = 0; i < N-1; i++) {
    index = (index + a[i]) * shape[i+1];
}
index += a[N-1];
로 하는 것이 optimal solution이다.

2026/01/17
1. Tensor 클래스 코드 작성
- 참조 변수: int& a = b; 이런식으로 자료형 옆에 &을 붙이면 참조 변수가 된다. 참조 변수는 값을 복사하지 않고, 메모리 주소만 참조하기 때문에 수행시간이 거의 0이다. 또한 참조 변수 a를 수정하면 b도 수정되고, b를 수정해도 a가 수정된다. 그냥 참조 변수는 그 자체로 b 변수와 같아진다고 이해하면 된다. 실제로 컴파일러가 작동할 때, 참조 변수 a는 특정한 메모리 공간을 차지하지 않고, a가 사용될 때마다 그냥 컴파일러가 b로 치환해서 해석하기도 한다. 참조 변수를 사용할 때는 dangling reference 를 주의해야 한다. 원본 변수의 lifetime이 참조 변수보다 짧다면, 심각한 에러를 발생시킬 것이다.
- operator는 c++에서 예약된 이름이다. double& operator() (index) {return data[index];} 이런식으로 작성하면 객체의 연산자를 함수로 만든것이다. 여기서는 () 연산자를 조작한 것이다.
- vector는 동적할당을 해주고, 데이터들이 메모리상에서 한줄로 위치한다. (속도 매우 빠름.), initialize_list는 데이터 수정이 불가한 대신에 함수 인자로 넘겨줄 때 자주 사용됨. vector는 자주 생성/소멸될 때 비효율적이지만, initialize_list는 그렇지 않음.
- 예외처리에 대하여: 디버깅할 때만 하고, release했을 때는 안 하는 경우 존재.
- reshape(): Tensor Class 의 shape과 strides 만 바꿔주면 됨.

2026/01/18
1. 역전파에 대하여
- autograd 방식을 위해 연산 graph를 만들어야 함. 각 tensor는 부모 tensor를 기록해둬야 함. 또한 Temporary Object 에 대하여는 연산 그래프를 어떻게 구현할지 고려해야함. -> 복합 연산에서는 temporary object가 발생하고 이를 그냥 관리하면 연산이 끝나고 temporary object 가 날아가버리기 때문에 나중에 역전파할 때 안됨.

(1) Temporary Object에 대하여: shared_ptr 을 사용한다. shared_ptr은 변수 이름이 중요한 것이 아니라 본인을 어떤것이 참조하고 있다면 계속 메모리 공간에서 남아있는 타입이다. new가 하는 역할은 메모리를 요청하는 역할을 한다. 그리고 delete가 되기 전까지 메모리 공간을 아무도 못쓰게 한다. shared_ptr 역시 메모리를 요청한다는 점에서 new와 비슷하지만 Control Block이 참조횟수를 체크하며 참조횟수가 0보다 클때까지 메모리 공간을 차지하도록 한다. shared_ptr<double[]> 은 double type의 배열 전체를 가리키도록 명시한다. shared_ptr<double> 로 하게 되면 단일 원소만 가리키는 것으로 착각하여 내부적으로 소멸될 때 delete[]가 아닌 delete를 하며 memory leaking 위험이 높다. shared_ptr<T>에서 T는 스마트 포인터가 관리해야할 알맹이 데이터의 type을 가리킨다. shared_ptr<double*> 은 이중포인터가 되어서 위험하다. 이 또한 control block 이 delete[]가 아닌 delete를 하여 메모리 에러가 발생한다. shared_ptr 자체가 포인터를 뜻하고, <double[]> 은 control block이 관리해야할 데이터가 배열이라는 것을 알려주는 것이다. 또한 shared_ptr<double> 로 하면 data[i] 등의 접근을 막아두는데 shared_ptr<double[]> 로 하면 data[i] 접근 가능. shared_ptr로 생성하는 것과 make_ptr로 생성하는 것의 차이. shared_ptr<double[]>(new double[total_size]); 로 하면 new 부분에서 배열에 대한 할당 요청 한번과 shared_ptr 부분에서 제어 블록에 관한 할당 요청이 한 번, 총 두 번 할당 요청이 들어가게 되어 속도가 느리다. make_shared 로 하면 할당 요청을 한번에 하기 때문에 속도가 빠름.
- d = a*b + c 이고, a, b, c의 참조횟수는 각각 1이라고 하면, a*b에서 a와 b가 operator* 의 인자로 넘어갈 때 참조횟수가 한번씩 증가해서 2가 되고, left_parent=lhs,right_parent=rhs 부분에서 참조횟수가 한번씩 증가해서 3이 되고, res는 make_shared를 해서 참조횟수가 1이었다가, res가 operator*에서 반환되면, 함수 scope에서 인자로 넘어올 때 참조횟수가 한번씩 증가했던게 사라지면서 a와 b의 참조횟수는 2가 되고, res였던 a*b는 참조회수가 여전히 1이었다가, a*b + c를 할 때 위의 과정을 또같이 거치면 a*b 객체는 참조횟수가 2, c도 참조횟수가 2가 되고, (a*b+c) 객체는 참조횟수가 1이었다가, d에 대입될 때 참조횟수가 1 증가해서 2가 되고, 그 다음줄로 넘어가면 temporary object 였던 a*b와 a*b+c가 사라져서 참조횟수가 1씩 감소한다. 즉, a, b, c의 참조횟수는 2, a*b와 a*b+c, d의 참조횟수는 1이 된다. shared_ptr의 참조횟수가 올라가는 유일한 방법은 shared_ptr 그 자체가 누군가에게 대입되거나 복사될 때이다. ex) shared_ptr<int> b = make_shared<int>(10); shared_ptr<int> c = make_shared<int>(20); int a = *b + *c; // 여전히 b와 c의 참조횟수는 1이다.

2026/01/19
1. 역전파 순서에 대하여
- 역전파 순서에서 가장 중요한 것은 부모 노드에 grad가 전달되기 전에 자식 노드의 grad가 완성되어야 한다는 점이다. 따라서 부모 노드가 꼭 자식 노드보다 나중에 grad를 전달받아야 한다. dfs 알고리즘을 이용해서 부모노드를 모두 방문한 노드를 리스트에 추가하면, 모든 자식 노드가 부모 노드보다 뒤에 오도록 정렬된다. 이 리스트를 거꾸로 뒤집으면 모든 부모 노드가 자식 노드보다 나중에 위치하는 정렬 순서를 얻게 된다.
