__kernel void matrixMultiply(__global float *A, __global float *B, __global float *C, int A_Rows, int A_Cols, int B_Rows, int B_Cols, int C_Rows, int C_Cols){
	
	int row = get_global_id(0);
	int col = get_global_id(1);

	if((row < A_Rows) && (col < B_Cols)){
	
		float CValue = 0.0f;
		for(int k = 0; k < A_Cols; k++)
				CValue += A[row * A_Cols + k] * B[k * B_Cols + col];	

		C[row * C_Cols + col] = CValue;
	}
}
	