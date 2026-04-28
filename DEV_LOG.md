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

(1) Temporary Object에 대하여: shared_ptr 을 사용한다. shared_ptr은 변수 이름이 중요한 것이 아니라 본인을 어떤것이 참조하고 있다면 계속 메모리 공간에서 남아있는 타입이다. new가 하는 역할은 메모리를 요청하는 역할을 한다. 그리고 delete가 되기 전까지 메모리 공간을 아무도 못쓰게 한다. shared_ptr 역시 메모리를 요청한다는 점에서 new와 비슷하지만 Control Block이 참조횟수를 체크하며 참조횟수가 0보다 클때까지 메모리 공간을 차지하도록 한다. shared_ptr<double[]> 은 double type의 배열 전체를 가리키도록 명시한다. `shared_ptr<double>` 로 하게 되면 단일 원소만 가리키는 것으로 착각하여 내부적으로 소멸될 때 delete[]가 아닌 delete를 하며 memory leaking 위험이 높다. `shared_ptr<T>`에서 T는 스마트 포인터가 관리해야할 알맹이 데이터의 type을 가리킨다. shared_ptr<double*> 은 이중포인터가 되어서 위험하다. 이 또한 control block 이 delete[]가 아닌 delete를 하여 메모리 에러가 발생한다. shared_ptr 자체가 포인터를 뜻하고, <double[]> 은 control block이 관리해야할 데이터가 배열이라는 것을 알려주는 것이다. 또한 `shared_ptr<double>` 로 하면 data[i] 등의 접근을 막아두는데 shared_ptr<double[]> 로 하면 data[i] 접근 가능. shared_ptr로 생성하는 것과 make_ptr로 생성하는 것의 차이. shared_ptr<double[]>(new double[total_size]); 로 하면 new 부분에서 배열에 대한 할당 요청 한번과 shared_ptr 부분에서 제어 블록에 관한 할당 요청이 한 번, 총 두 번 할당 요청이 들어가게 되어 속도가 느리다. make_shared 로 하면 할당 요청을 한번에 하기 때문에 속도가 빠름.
- d = a*b + c 이고, a, b, c의 참조횟수는 각각 1이라고 하면, a*b에서 a와 b가 operator* 의 인자로 넘어갈 때 참조횟수가 한번씩 증가해서 2가 되고, left_parent=lhs,right_parent=rhs 부분에서 참조횟수가 한번씩 증가해서 3이 되고, res는 make_shared를 해서 참조횟수가 1이었다가, res가 operator*에서 반환되면, 함수 scope에서 인자로 넘어올 때 참조횟수가 한번씩 증가했던게 사라지면서 a와 b의 참조횟수는 2가 되고, res였던 a*b는 참조회수가 여전히 1이었다가, a*b + c를 할 때 위의 과정을 또같이 거치면 a*b 객체는 참조횟수가 2, c도 참조횟수가 2가 되고, (a*b+c) 객체는 참조횟수가 1이었다가, d에 대입될 때 참조횟수가 1 증가해서 2가 되고, 그 다음줄로 넘어가면 temporary object 였던 a*b와 a*b+c가 사라져서 참조횟수가 1씩 감소한다. 즉, a, b, c의 참조횟수는 2, a*b와 a*b+c, d의 참조횟수는 1이 된다. shared_ptr의 참조횟수가 올라가는 유일한 방법은 shared_ptr 그 자체가 누군가에게 대입되거나 복사될 때이다. ex) `shared_ptr<int>` b = `make_shared<int>`(10); `shared_ptr<int>` c = `make_shared<int>`(20); int a = *b + *c; // 여전히 b와 c의 참조횟수는 1이다.

2026/01/19
1. 역전파 순서에 대하여
- 역전파 순서에서 가장 중요한 것은 부모 노드에 grad가 전달되기 전에 자식 노드의 grad가 완성되어야 한다는 점이다. 따라서 부모 노드가 꼭 자식 노드보다 나중에 grad를 전달받아야 한다. dfs 알고리즘을 이용해서 부모노드를 모두 방문한 노드를 리스트에 추가하면, 모든 자식 노드가 부모 노드보다 뒤에 오도록 정렬된다. 이 리스트를 거꾸로 뒤집으면 모든 부모 노드가 자식 노드보다 나중에 위치하는 정렬 순서를 얻게 된다. 

2. 역전파 기능 추가함.
- backward() 안에서 build_topo() 의 첫번째 인자로 넣어줄 때, this는 Tensor* 타입인데 인자로 받는 타입은 shared_ptr<Tensor> 라서 고민이 되었는데, shared_from_this()라는 메소드가 있어서 enable_shared_from_this를 상속받아서 사용하면 해결된다. shared_from_this는 shared_ptr<Tensor>를 새로 만드는 대신에 이미 this를 가리키고 있는 shared_ptr<Tensor>를 찾아서 그걸 넣어주는 것이다. 근데, 테스트 도중에 bad_weak_ptr 에러를 마주쳤고, 해결 방법을 못 떠올려서 그냥 backward()를 전역함수로 빼주고 shared_ptr<Tensor>를 인자로 받는 함수로 바꾸어주었다. 공부를 더 해봐야겠다.

2026/01/20
1. operation에 대한 다형성 확보
- 연산이 매우 많기 때문에 다형성이 확보되어야 하고, 기존에 op를 String으로 관리하던 것을 객체형태로 관리하는 것으로 변경. 

2026/01/21
1. operation에 대한 다형성 확보
- 코드 작성

2026/01/23
1. broadcasting 구현 및 설계
- shape은 오른쪽부터 비교하며, 만약 shape이 다르다면 둘 중 하나는 1이어야 한다. (broadcastable 조건) broadcasting을 할 때 shape이 1인 dimension은 stretch 되는데, 이를 쉽게 구현하기 위해서 해당 dimension의 stride를 0으로 지정한다. 그렇게 입력 데이터의 shape과 stride를 새롭게 정의했다면, 출력값의 각 index에 대응하는 입력 데이터들의 index를 쉽게 계산할 수 있다. ex) v1의 shape: [2, 3, 1, 2], strides: [6, 2, 2, 1] 이고, v2의 shape: [1, 4, 2], strides: [8, 2, 1]이라면 다음과 같이 shape과 stride가 새롭게 정의된다. v1은 shape: [2, 3, 1, 2], strides: [6, 2, 0, 1], v2는 shape: [1, 1, 4, 2], strides: [0, 0, 2, 1], 그리고 출력 shape은 [2, 3, 4, 2]이다. 

참고 
- shared_ptr<double[]> a에서 a의 배열 길이는 구할 수 없다. 따라서 size를 따로 저장해두어야 한다. 근데 나중에 delete[]를 할 때 결국 지정된 범위의 메모리만큼을 해지하려면 배열 크기를 알고 있어야 하지 않나? 하는 의문이 있었는데, 그 배열 크기는 메모리 어딘가에 저장되어 있기는 하지만, shared_ptr 내부가 아니라 cookie라는 숨겨진 공간에 저장된다고 한다. 그 공간에는 c++ 컴파일러와 운영체제가 담당하여 접근하기가 어려울 뿐더러 운영체제마다 컴파일러가 저장하는 방식이 다르며, 배열이 아닌 단순한 자료형일 때는 크기를 아예 저장하지 않기도 하는 등의 문제로 c++ 표준 함수가 없어서 배열 길이를 구할 수 없다.
- 설계에 있어서 문제를 정확히 정의하고, 로직 과정을 구체적으로 정리하는 것이 중요하다. 코드를 깔끔하고 효율적으로 작성하는데도 도움이 되고 문제해결이 더 수월함.

2026/02/08
1. backpropagation의 순서를 결정할 때 dfs 알고리즘으로 위상정렬하고 뒤집는 것이 직관적으로 이해가 되지 않아서 좀 더 공부해보았다. 비순환 방향 그래프에서의 Dependency Resolution(어떤 일을 하기  미리 할 일들을 찾아가는 과정) 에서 DFS Post-order 방식이 효과적이다. DFS Post-order는 특정 노드가 의존하는 노드들을 뿌리부터 뽑는 느낌이다. 위상적 관점에서는 모든 부모 노드가 자식 노드보다 앞에 있도록 한다. 역전파 과정에서는 모든 자식노드가 부모노드보다 앞에 있어야 하는 문제를 해결해야 했고, 이를 뒤집어서 모든 부모노드가 자식노드보다 앞에 오도록 문제를 변형해서 생각한 것이다. 그리고 그 순서는 DFS post-order로 해결이 가능했다. DFS post-order는 다중상속된 클래스를 compile할 때도 마찬가지로 사용된다. Dependency Resolution에서 DFS post-order가 효과적인 것 같다.

2. 메모리 영역은 크게는 stack, heap, data 영역으로 나누어진다. stack 영역은 잠깐 쓰고 사라지는 변수들이 위치하고, heap 영역은 필요할 때 필요하 만큼 빌려쓰는 공간, data 영역은 static 변수와 전역 변수가 위치하며 프로그램이 끝날 때까지 존재한다. 그리고 cache도 존재한다. CPU와 물리적 거리가 멀수록 속도가 매우 많이 느려진다고 한다. 그래서 컴퓨터는 자주 쓰는 값들을 cache에 옮겨둔다. cache에 값을 옮겨둘 때는 1byte씩 옮겨두는게 아니고 64 byte와 같은 형식으로 묶어서 옮겨둔다고 한다. 그래서 속도 측면에서 메모리 주소상 거리가 먼 값들을 교차하면서 접근하면 속도가 느려진다. 너무 큰 보폭으로 건너뛰며 메모리에 접근하면 Cache가 계속계속 바뀌는 Cache Thrashing 현상이 발생한다. 따라서 연속적으로 사용할 값들을 가깝게 배치해두는 것이 메모리 효율에 유리하다. 이 점을 다차원 배열 상에 적용해보면 접근할 때 가장 보폭이 좁게 움직이는 것이 속도 측면에서 빠르다.

2026/02/09
1. Matrix Multiplication
- 딥러닝에서는 N차원과 2차원의 matrix multiplication 을 구현해야 한다. 그리고 연산 속도를 고려하여 A@B=C 일 때, B를 transpose해둔 채로 계산하는 것이 cache hit rate가 더 높기 떄문에 속도가 빠르다. C의 ( ..., m, n) 위치의 값은 A( ..., m) 과 B^T(n) 의 내적이다. 참고로 A(..., m)과 B^T(n)은 row vector이다. 일반적으로 계산하면 B를 column vector로 나누어서 연산하게 되는데 이보다는 row vector로 다루어서 연속적인 메모리 레이아웃을 사용하는 것이 앞서 말한 까닭으로 인해 속도측면에서 유리하다.

2026/02/10
- 텐서별로 데이터를 다루는 것이 아니라 성분별로 객체로 보고 다루면 정의해줘야 하는 연산도 줄어들고 구현도 쉬워지지 않을까 하는 생각을 하였다. 찾아보니 이미 이 개념은 micrograd라고 불리우고 있었고, 구현면에서 훨씬 간단한 것은 맞지만 성분들을 다 객체로 다루면 메모리도 엄청 많이 먹을 뿐더러(성분별로 parents, attrs 등을 저장해야함.) 캐시 친화적 구현이 어렵다. 따라서 그냥 Tensor 단위로 데이터를 다루는 것을 현재 라이브러리들은 택하고 있다고 한다.
- 함수 인자에는 auto 타입이 들어갈 수 있다. (단, c++ 버전이 20 이상이어야 함.)  컴파일러는 이를 내부적으로 template으로 처리한다고 한다. template은 타입 자체를 변수화해서 처리할 수 있다. template <typename T1, typename T2 .. > 이렇게 쓰고 밑에 함수나 클래스를 정의하면, 파라미터의 타입을 T1, T2로 쓸 수 있다. 함수를 호출할 때는 함수<int, float> 과 같은 형식으로 typename을 넣어줄 수 있다. template은 typename뿐만 아니라, class도 쓸 수 있고, int 등등 여러가지 타입을 받을 수 있다. template은 문법 요소를 변수처럼 다룰 수 있게 해주는 기능을 한다.
- Function 내부에 forward함수가 모든 자식클래스의 forward 입력 형태를 강제해서 불편함. 따라서 그냥 forward를 지워버림. backward는 별문제가 없어서 남겨둠.
- 나중에 학습을 돌릴 때, 연산 graph가 누적되어서 생성되고 없어져야 하는 shared_ptr이 쓸데없는 참조때문에 살아있는 그런 문제들이 우려되었음. shared_ptr의 참조를 어떻게 추적할까 고민이 되었는데, shared_ptr의 참조횟수는 최초 생성될 때, 대입될 때, 함수 인자로 넘어갈 떄 (단 참조타입으로 넘어갈 때는 카운트 안 늘어남.) 만 늘어난다. 모든 대입과 함수 인자로 넘어가는 부분을 확인해보려고 했으나 대입된 변수와 함수 인자로 넘어가는 건 scope가 끝나면 자동적으로 메모리 할당이 해지되기 때문에 참조횟수가 줄어들어서, 결국 고려해야 하는 것은 최종 계산이 되었을 때의 shared_ptr의 참조를 관찰하면 되는 것이다. (임시 참조와 영구 참조의 차이) 참조는 이때만 관찰해주면 되고, 연산 graph 누적 문제는 backward가 수행되고 나서 직접 각 shared_ptr<Tensor>들의 parents들을 비워주는 작업을 하면서 아래서부터 가지를 하나씩 끊으면, 결국에는 다 풀리게 되어서 초기화가 된다.

1. Matrix Multiplication 구현
- Transpose를 먼저 구현해야 함. 구현은 크게 어렵지 않고, backward 부분만 실수하지 않도록 조심해서 구현해주면 된다.
- Matrix Multiplication의 A@B에서 B는 transpose를 취하여 연산한다. 행렬미분에 대한 지식이 충분하지는 않지만, 결국 행렬연산을 분해해보면 성분별 곱과 합으로 이루어져 있기 때문에 구현은 할 수 있다. (이후 행렬미분에 대한 공부는 필요)

2026/02/14
- exp, reciprocal, sigmoid, leakyrelu를 추가하였다. sigmoid는 따로 클래스를 정의할 필요 없이 이미 정의된 모듈들을 통해서 함수 하나만으로 표현이 가능했다.
- 전역 scope에서는 변수의 선언과 초기화만 허용된다.
- Layer class 를 추가하여, 이 class를 상속받는 여러 Layer Class를 만들어야 함. 이 Layer들은 각각의 parameters 멤버변수를 갖고 있고 나중에 optimization은 이 parameters에 접근해서 update를 하는 것이다.
- Layer들을 그룹으로 관리할 수 있도록 Sequential class 를 만들어야 한다.

2026/03/02
- reciprocal 함수를 호출할 때 오류가 발생. saved_tensors vector에 push_back할 때 segmentation fault 에러가 뜸. 이유는 reciprocal 함수에서 auto grad_fn = shared_ptr<ReciprocalFunction>(); 으로 써서 났던 것임. shared_ptr 은 type의 한 형태로 이해하면 된다. 더 정확히는 특정 객체를 가리키는 스마트 포인터인데, shared_ptr<ReciprocalFunction>() 은 ReciprocalFunction 타입 객체를 가리키는 스마트 포인터라는 의미일 뿐, ReciprocalFunction 객체를 생성하지는 않는다. make_shared<ReciprocalFunction>() 을 해야 ReciprocalFunction 객체를 생성하고 그 객체를 가리키는 스마트포인터 객체도 생성해서 반환하는 것이다.

2026/03/11
- reciprocal, exp, sigmoid, leaky_relu 함수를 검증을 했는데, exp 함수에서 backward 부분에서 grad_output[i]를 곱해주지 않는 오류가 있어 수정하였다.
- C++에서 가상 메모리 주소와 실제 메모리 주소가 어떤 식으로 다른지 좀 더 찾아보았다. C++에서 변수의 메모리 주소를 출력했을 때 나오는 값은 가상 메모리 주소이다. 실제 메모리 주소와는 다르다. 실제로 OS가 메모리를 관리할 때는 덩어리 단위로 관리를 하는데, 만약 a라는 배열의 크기가 1MB이고, OS가 4KB씩 나누어서 메모리를 관리한다면 실제 메모리 주소에서는 값들이 일렬로 배열되어 있지 않을 확률이 있다. 하지만 C++에서 compile 할 때는 가상 메모리 주소상 일렬로 배열되어 있어서 그냥 메모리 주소 연산이 가능하다. ex) a + 3 은 4번째 값의 가상 메모리 주소로 나오고 그 가상 메모리 주소를 역참조연산자로 접근하면 4번째 값이 나옴. 

2026/03/17
- softmax 함수를 구현할 때 처음에는 backward 부분에서 실제 값과 다른 값을 내는 오류가 있었다. 역전파를 할 때, 입력인 Y_i의 grad는 D_i 에서만 온다고 생각한 것이다. 그러나 실제 미분을 할 때는 D_i에 영향을 주는 모든 변수에 D_i의 grad를 전파해야 하므로, Y_i의 grad는 D_i에서만 오는 것이 아니라, summation한 다른 index에서도 전부 오는 것이다. 역전파를 할 때는 입력 입장에서 보는 것보다, 출력 입장에서 grad를 뿌리는걸로 생각하는게 더 정확하다. 사실 행렬, 벡터 미분으로 제대로 표현하면 실수할 일이 없을 것 같긴 하다. softmax function에서는 exp_input을 parent로 둬서 계산을 진행하고, exp부분의 backward는 기존의 autograd를 따라갈 수 있도록 하였다.

2026/04/26
- DEV_LOG를 너무 오랫동안 안 써서 코드는 수정이 되었는데, 기록이 안 된 부분이 좀 많은 것 같다.
- Tensor class에서 set_data method를 추가해주었다. std::copy라는 함수를 이용하면 값 여러개를 한번에 넣어줄 수 있다. vector.begin(), vector.end() 는 반복자라고 불리고, .begin()은 0번쨰 index, .end()는 마지막 index+1 번째 주소를 가리킨다. 수학적으로 반개구간이라고 하는 [a, b)의 개념이다. 그리고 this->data.get()을 해주었는데, 그냥 data만 하면 shared_ptr<double[]> 이라서 객체를 가리키게 되어, data.get()을 하여 그 안에 배열을 가리키도록 한다. 만약에 data의 크기가 vector 크기보다 크다면 data의 0번째 index부터 부분복사가 된다. 만약 data의 크기가 vector 크기보다 작다면 buffer overflow가 발생하는 대참사가 일어난다. 이때 copy는 에러를 안 띄워서 반드시 예외처리를 해주어야 한다. vector 말고 array를 copy하고 싶다면, copy(array, array+n, a); 혹은 copy(std::begin(array), std::end(array), a); 처럼 사용하면 된다.
- make_shared<T[]>(size); 와 같은 문법은 c++20 이후부터 가능하다. 그 이하 버전에서는 shared_ptr<T[]>(new T[size]); 처럼 사용해야 함.
- CrossEntropyLoss를 추가함.

2026/04/28
- mnist 데이터를 불러오고 학습을 돌리려고 하는데, 테스트 도중 prediction이 0으로만 나오는 것을 확인. -> 가중치를 확인해보니 전부 0임. 초기화쪽에 문제있음. -> sigma쪽에 2/n_in가 있었는데, 정수/정수 는 정수 타입으로 나와서 생긴 오류였다. 2를 2.0으로 수정하여 해결.
- auto x = make_shared<double>(3.0); x = make_shared<double>(5.0); 이렇게 하면 기존의 3.0은 사라진다고 함.
- batch_size를 고려하지 않고 코드를 먼저 짰어서, 이미지를 하나씩 학습시키는데 불안정도가 매우 높음. learning_rate를 아주 작게 해야 안튀는거 같음.
- 층이 두개 이상이 되면 자꾸 튀어서 softmax에서 max값을 빼주는 계산을 추가해줌. 그리고 sgd에서 momentum을 이용할 수 있도록 추가해줌. -> 훨씬 덜 튀게 되었음.
