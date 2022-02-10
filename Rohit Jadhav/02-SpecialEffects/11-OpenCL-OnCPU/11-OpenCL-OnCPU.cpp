#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<CL/opencl.h>

#include"helper_timer.h"

#define BLOCK_WIDTH 2048

cl_int ret_ocl;
cl_platform_id *oclPlatformID;
cl_device_id oclComputeDeviceID;
cl_context oclContext;
cl_command_queue oclCommandQueue;
cl_program oclProgram;
cl_kernel oclKernel;

char *oclKernelSouceCode = NULL;
size_t sizeKernelCode;

size_t localWorkSize[] = {8192};
size_t globalWorkSize[] = {BLOCK_WIDTH * BLOCK_WIDTH};

float *hostA = NULL;
float *hostB = NULL;
float *hostC = NULL;
float *CHost = NULL;	//CPU;

cl_mem deviceA = NULL;
cl_mem deviceB = NULL;
cl_mem deviceC = NULL;

float timeOnCPU;
float timeOnGPU;

int main(void) {


	void fillFloatArrayWithRandomNumbers(float*, int);
	size_t roundGlobalSizeToNearestMultipleOfLocalSize(int, unsigned int);
	void matMulHost(float*, float*, float*, int, int, int);
	char *loadOclProgramSource(const char*, const char*, size_t*);
	void cleanup(void);

	int numA_Rows;
	int numA_Cols;
	int numB_Rows;
	int numB_Cols;
	int numC_Rows;
	int numC_Cols;
	int numCHost_Rows;
	int numCHost_Cols;

	numA_Rows = BLOCK_WIDTH;
	numA_Cols = BLOCK_WIDTH;
	numB_Rows = BLOCK_WIDTH;
	numB_Cols = BLOCK_WIDTH;

	numC_Rows = numA_Rows;
	numC_Cols = numB_Cols;

	numCHost_Rows = numA_Rows;
	numCHost_Cols = numB_Cols;

	int sizeA = numA_Rows * numA_Cols * sizeof(float);
	int sizeB = numB_Rows * numB_Cols * sizeof(float);
	int sizeC = numC_Rows * numC_Cols * sizeof(float);
	int sizeCHost = numCHost_Rows * numCHost_Cols * sizeof(float);

	printf("SIZE: %d\n", sizeCHost);

	hostA = (float*)malloc(sizeA);
	if (hostA == NULL) {
		printf("CPU Memory Fatal Error: Can not Allocate Enough Memory For hostA\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	hostB = (float*)malloc(sizeB);
	if (hostB == NULL) {
		printf("CPU Memory Fatal Error: Can not Allocate Enough Memory For hostB\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	hostC = (float*)malloc(sizeC);
	if (hostC == NULL) {
		printf("CPU Memory Fatal Error: Can not Allocate Enough Memory For hostC\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	CHost = (float*)malloc(sizeCHost);
	if (CHost == NULL) {
		printf("CPU Memory Fatal Error: Can not Allocate Enough Memory For CHost\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	fillFloatArrayWithRandomNumbers(hostA, numA_Rows * numA_Cols);
	fillFloatArrayWithRandomNumbers(hostB, numB_Rows * numB_Cols);

	//1. Platform ID
	cl_uint platformNo;
	clGetPlatformIDs(0, NULL, &platformNo);
	oclPlatformID = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformNo);
	printf("SUCCESS: No Of Platform: %d\n", platformNo);

	ret_ocl = clGetPlatformIDs(platformNo, oclPlatformID, NULL);
	if (ret_ocl != CL_SUCCESS) {
		printf("clGetPlatformIDs() Failed: %d\n", ret_ocl);
		cleanup();
		exit(EXIT_FAILURE);
	}



	/********** OPENCL CPU DEVICE !!!!!*********/
	//2. Device ID
	ret_ocl = clGetDeviceIDs(oclPlatformID[1], CL_DEVICE_TYPE_CPU, 1,
		&oclComputeDeviceID, NULL);
	if (ret_ocl != CL_SUCCESS) {
		printf("clGetDeviceIDs() Failed: %d\n", ret_ocl);
		cleanup();
		exit(EXIT_FAILURE);
	}

	char gpu_name[255];
	clGetDeviceInfo(oclComputeDeviceID, CL_DEVICE_NAME, sizeof(gpu_name), &gpu_name, NULL);
	printf("\nYour GPU: %s\n", gpu_name);
	
	//3. Creating Context
	oclContext = clCreateContext(NULL, 1, &oclComputeDeviceID, NULL, NULL, &ret_ocl);
	if (ret_ocl != CL_SUCCESS) {
		printf("clCreateContext() Failed: %d\n", ret_ocl);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//4. Creating Command Queue
	//0 - default command queue propertys
	oclCommandQueue = clCreateCommandQueue(oclContext, oclComputeDeviceID, 0, &ret_ocl);
	if (ret_ocl != CL_SUCCESS) {
		printf("clCreateCommandQueue() Failed: %d\n", ret_ocl);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//5. Program Object for .cl File
	oclKernelSouceCode = loadOclProgramSource("MatMul.cl", "", &sizeKernelCode);
	oclProgram = clCreateProgramWithSource(oclContext, 1,(const char**)&oclKernelSouceCode, NULL, &ret_ocl);
	if (ret_ocl != CL_SUCCESS) {
		printf("clCreateProgramWithSource() Failed: %d\n", ret_ocl);
		cleanup();
		exit(EXIT_FAILURE);
	}


	//6. Build Program object
	//4th NULL for Build Property which is default
	ret_ocl = clBuildProgram(oclProgram, 0, NULL, NULL, NULL, NULL);
	if (ret_ocl != CL_SUCCESS) {
		
		printf("clBuildProgram() Failed: %d\n", ret_ocl);
		size_t len;
		char buffer[2048];
		clGetProgramBuildInfo(oclProgram, oclComputeDeviceID, CL_PROGRAM_BUILD_LOG,
			sizeof(buffer), buffer, &len);
		printf("Program Build Log: %s\n", buffer);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//7. Create Kernel From Program Object
	oclKernel = clCreateKernel(oclProgram, "matrixMultiply", &ret_ocl);
	if (ret_ocl != CL_SUCCESS) {
		printf("clCreateKernel() Failed: %d\n", ret_ocl);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//8. Allocate Memory for Device Array
	int size = sizeof(float) * numC_Rows * numC_Cols;
	deviceA = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &ret_ocl);
	if (ret_ocl != CL_SUCCESS) {
		printf("deviceA clCreateBuffer() Failed: %d\n", ret_ocl);
		cleanup();
		exit(EXIT_FAILURE);
	}

	deviceB = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &ret_ocl);
	if (ret_ocl != CL_SUCCESS) {
		printf("deviceB clCreateBuffer() Failed: %d\n", ret_ocl);
		cleanup();
		exit(EXIT_FAILURE);
	}

	deviceC = clCreateBuffer(oclContext, CL_MEM_WRITE_ONLY, size, NULL, &ret_ocl);
	if (ret_ocl != CL_SUCCESS) {
		printf("deviceC clCreateBuffer() Failed: %d\n", ret_ocl);
		cleanup();
		exit(EXIT_FAILURE);
	}
	else if(ret_ocl == CL_SUCCESS){
		printf("DeviceC\n");
	}

	//9. Setting Kernel Arg
		//1
	ret_ocl = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), (void*)&deviceA);
	if (ret_ocl != CL_SUCCESS) {
		printf("1 clSetKernelArg() Failed: %d\n", ret_ocl);
		cleanup();
		exit(EXIT_FAILURE);
	}
		//2
	ret_ocl = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), (void*)&deviceB);
	if (ret_ocl != CL_SUCCESS) {
		printf("2 clSetKernelArg() Failed: %d\n", ret_ocl);
		cleanup();
		exit(EXIT_FAILURE);
	}
		//3
	ret_ocl = clSetKernelArg(oclKernel, 2, sizeof(cl_mem), (void*)&deviceC);
	if (ret_ocl != CL_SUCCESS) {
		printf("3 clSetKernelArg() Failed: %d\n", ret_ocl);
		cleanup();
		exit(EXIT_FAILURE);
	}
		//4
	ret_ocl = clSetKernelArg(oclKernel, 3, sizeof(cl_int), (void*)&numA_Rows);
	if (ret_ocl != CL_SUCCESS) {
		printf("4 clSetKernelArg() Failed: %d\n", ret_ocl);
		cleanup();
		exit(EXIT_FAILURE);
	}
		//5
	ret_ocl = clSetKernelArg(oclKernel, 4, sizeof(cl_int), (void*)&numA_Cols);
	if (ret_ocl != CL_SUCCESS) {
		printf("5 clSetKernelArg() Failed: %d\n", ret_ocl);
		cleanup();
		exit(EXIT_FAILURE);
	}
		//6
	ret_ocl = clSetKernelArg(oclKernel, 5, sizeof(cl_int), (void*)&numB_Rows);
	if (ret_ocl != CL_SUCCESS) {
		printf("6 clSetKernelArg() Failed: %d\n", ret_ocl);
		cleanup();
		exit(EXIT_FAILURE);
	}
		//7
	ret_ocl = clSetKernelArg(oclKernel, 6, sizeof(cl_int), (void*)&numB_Cols);
	if (ret_ocl != CL_SUCCESS) {
		printf("7 clSetKernelArg() Failed: %d\n", ret_ocl);
		cleanup();
		exit(EXIT_FAILURE);
	}
		//8
	ret_ocl = clSetKernelArg(oclKernel, 7, sizeof(cl_int), (void*)&numC_Rows);
	if (ret_ocl != CL_SUCCESS) {
		printf("8 clSetKernelArg() Failed: %d\n", ret_ocl);
		cleanup();
		exit(EXIT_FAILURE);
	}
		//9
	ret_ocl = clSetKernelArg(oclKernel, 8, sizeof(cl_int), (void*)&numC_Cols);
	if (ret_ocl != CL_SUCCESS) {
		printf("9 clSetKernelArg() Failed: %d\n", ret_ocl);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//10. Host To Device
	ret_ocl = clEnqueueWriteBuffer(oclCommandQueue, deviceA, CL_FALSE, 0, size, hostA,
		0, NULL, NULL);
	if (ret_ocl != CL_SUCCESS) {
		printf("clEnqueueWriteBuffer() Failed: %d\n", ret_ocl);
		cleanup();
		exit(EXIT_FAILURE);
	}

	ret_ocl = clEnqueueWriteBuffer(oclCommandQueue, deviceB, CL_FALSE, 0, size, hostB,
		0, NULL, NULL);
	if (ret_ocl != CL_SUCCESS) {
		printf("clEnqueueWriteBuffer() Failed: %d\n", ret_ocl);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//11. Run The Kernel

	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	ret_ocl = clEnqueueNDRangeKernel(oclCommandQueue, oclKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

	if (ret_ocl != CL_SUCCESS) {
		printf("clEnqueuNDRangeKernel() Failed: %d\n", ret_ocl);
		cleanup();
		exit(EXIT_FAILURE);
	}
	else if(ret_ocl == CL_SUCCESS){
		printf("Kernel Called\n");
	}

	//finish Command Queue
	clFinish(oclCommandQueue);

	sdkStopTimer(&timer);
	timeOnGPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);

	//12. Device To Host
	ret_ocl = clEnqueueReadBuffer(oclCommandQueue, deviceC, CL_TRUE, 0, size, hostC, 0, NULL, NULL);
	if (ret_ocl != CL_SUCCESS) {
		printf("clEnqueueReadBuffer() Failed: %d\n", ret_ocl);

		switch(ret_ocl){
			case CL_INVALID_PROGRAM_EXECUTABLE:	
				printf("\t1\n");
				break;

			case CL_INVALID_COMMAND_QUEUE:
				printf("\t2\n");
				break;

			case CL_INVALID_CONTEXT:
				printf("\t3\n");
				break;

			case CL_INVALID_WORK_DIMENSION:
				printf("\t4\n");
				break;

			case CL_INVALID_WORK_GROUP_SIZE:
				printf("\t5\n");
				break;

			case CL_INVALID_WORK_ITEM_SIZE:
				printf("\t6\n");
				break;   



			case CL_INVALID_MEM_OBJECT:
				printf("\t6*\n");
				break;


			case CL_INVALID_VALUE:
				printf("\t6-\n");
				break;



			case CL_MEM_OBJECT_ALLOCATION_FAILURE:
				printf("\t8\n");
				break;

			case CL_OUT_OF_HOST_MEMORY:
				printf("\t9\n");
				break;


			case CL_INVALID_EVENT_WAIT_LIST:
				printf("\t10\n");
				break;

			case CL_INVALID_GLOBAL_OFFSET:
				printf("\t11\n");
				break;

			case CL_INVALID_KERNEL_ARGS:
				printf("\t12\n");
				break;

			case CL_INVALID_KERNEL:
				printf("\t13\n");
				break;  


			case CL_OUT_OF_RESOURCES:
				printf("\tipou\n");
				break;     

		}


		cleanup();
		exit(EXIT_FAILURE);
	}

	//13. CPU 
	matMulHost(hostA, hostB, CHost, numA_Cols, numC_Rows, numC_Cols);

	//14. Compare
	const float epsilon = 0.0001f;
	bool bAccuracy = true;
	int breakValue = 0;
	int i;
	float diff;
	for (i = 0; i < numA_Rows * numA_Cols; i++){
		float val1 = CHost[i];
		float val2 = hostC[i];

		//printf("Host: %f \t Device: %f\n", val1, val2);
		diff = fabs(val1 - val2);
		//printf("Diff: %f\n", diff);

		if ( diff > epsilon) {
			bAccuracy = false;
			breakValue = i;
			break;
		}
	}

	if (bAccuracy == false)
		printf("\n\nBreak Value: %d\n", breakValue);

	//15. Answer
	char str[125];
	if (bAccuracy == true)
		sprintf(str, "%s", "Comparison of Output Are Accurate Within The Limit of 0.000001f\n");
	else
		sprintf(str, "%s", "Not All Comparison of Output Are Accurate Within The Limit of 0.000001f\n");


	printf("\n");
	printf("1st Matrix 0th Element: %f and %dth Element: %f\n", hostA[0], (numA_Rows * numA_Cols) - 1, hostA[(numA_Rows * numA_Cols) - 1]);
	printf("2nd Matrix 0th Element: %f and %dth Element: %f\n", hostB[0], (numB_Rows * numB_Cols) - 1, hostB[(numB_Rows * numB_Cols) - 1]);
	printf("\n");

	printf("Matrix Multiplicatio of 1st and 2nd Matrix: \n");
	printf("3rd Matrix 0th Element: %f and %dth Element: %f\n", hostC[0], (numC_Rows * numC_Cols) - 1, hostC[(numC_Rows * numC_Cols) - 1]);
	printf("\n");

	printf("Time on CPU: %f\n", timeOnCPU);
	printf("Time on GPU: %f\n", timeOnGPU);

	printf("\n");
	printf("%s\n", str);

	cleanup();
	
	return(0);
}

void cleanup(void) {

	if(oclPlatformID){
		free((void*)oclPlatformID);
		oclPlatformID = NULL;
	}

	
	if (deviceC) {
		clReleaseMemObject(deviceC);
		deviceC = NULL;
	}

	if (deviceB) {
		clReleaseMemObject(deviceB);
		deviceB = NULL;
	}

	if (deviceA) {
		clReleaseMemObject(deviceA);
		deviceA = NULL;
	}

	if (oclKernel) {
		clReleaseKernel(oclKernel);
		oclKernel = NULL;
	}

	if (oclProgram) {
		clReleaseProgram(oclProgram);
		oclProgram = NULL;
	}

	if (oclKernelSouceCode) {
		free(oclKernelSouceCode);
		oclKernelSouceCode = NULL;
	}

	if (oclCommandQueue) {
		clReleaseCommandQueue(oclCommandQueue);
		oclCommandQueue = NULL;
	}

	if (CHost) {
		free(CHost);
		CHost = NULL;
	}

	if (hostC) {
		free(hostC);
		hostC = NULL;
	}

	if (hostB) {
		free(hostB);
		hostB = NULL;
	}
	
	if (hostA) {
		free(hostA);
		hostA = NULL;
	}
}

void fillFloatArrayWithRandomNumbers(float *pFloatArray, int iSize) {
	int i;
	const float fScale = 1.0f / (float)RAND_MAX;
	for (i = 0; i < iSize; i++)
		pFloatArray[i] = fScale * rand();
}

size_t roundGlobalSizeToNearestMultipleOfLocalSize(int local_size, unsigned int global_size) {

	unsigned int r = global_size % local_size;
	if (r == 0)
		return(global_size);
	else
		return(global_size + local_size - r);
}

void matMulHost(float *A, float *B, float *C, int iA_Cols, int iC_Rows, int iC_Cols) {

	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	for (int i = 0; i < iC_Rows; i++) {
		for (int j = 0; j < iC_Cols; j++) {
			float sum = 0.0f;
			for (int k = 0; k < iA_Cols; k++) {
				float a = A[i * iA_Cols + k];
				float b = B[j + k * iA_Cols];
				
				sum += a * b;
			}
			C[i * iC_Cols + j] = sum;
		}	
	}

	sdkStopTimer(&timer);
	timeOnCPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
}

char *loadOclProgramSource(const char *filename, const char *preamble, size_t *sizeFinalLength) {
		
	FILE *pFile = NULL;
	size_t sizeSourceLength;

	pFile = fopen(filename, "rb");
	if (pFile == NULL)
		return(NULL);

	size_t sizePreambleLength = (size_t)strlen(preamble);

	fseek(pFile, 0, SEEK_END);
	sizeSourceLength = ftell(pFile);
	fseek(pFile, 0, SEEK_SET);

	char *sourceString = (char*)malloc(sizeSourceLength + sizePreambleLength + 1);
	memcpy(sourceString, preamble, sizePreambleLength);

	if (fread((sourceString)+sizePreambleLength, sizeSourceLength, 1, pFile) != 1) {
		fclose(pFile);
		free(sourceString);
		return(0);
	}

	fclose(pFile);
	if (sizeFinalLength != 0)
		*sizeFinalLength = sizeSourceLength + sizePreambleLength;

	sourceString[sizeSourceLength + sizePreambleLength] = '\0';
	return(sourceString);
}
