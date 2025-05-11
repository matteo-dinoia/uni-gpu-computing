//Implement a simple GEMM algorithm that multiplies two matrices of size NxN.
//• The algorithm takes the dimension of the matrices that is a power of 2.
//• For example, ./gemm 10 multiplies two matrices of 2 10 x 2 10 .
//• Measure the FLOPS of your implementation by using -00 –O1 –O2 –O3 options.

void gemm(int** m1, int** m2, int** res, int N){
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			int sum = 0;
			for (int c = 0; c < N; c++){
				m1
			}
		}
	}
}

void gen_random_matrix(int **m, int M, int N){
	for (int i = 0; i < M; i++){
		for (int j = 0; j < N; j++){
			m[]
		}
	}
}

int main(){
	int exp = 0;
	scanf("%d", &exp);
	int N = pow(2, exp);

	int m1[N][N];
	int m2[N][N];
	int res[N][N];



	gemm(m1, m2, res, N);
}