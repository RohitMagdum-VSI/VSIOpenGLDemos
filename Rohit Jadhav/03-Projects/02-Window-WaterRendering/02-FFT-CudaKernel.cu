#include<cufft.h>


int cuda_iDivUp(int x, int y){
	
	return(   (x + (y - 1)) / y   );
}


__device__  float2 conjugate(float2 arg){
	return(make_float2(arg.x, -arg.y));
}

__device__ float2 complex_exp(float arg){
	return(make_float2(cosf(arg), sinf(arg)));
}


__device__ float2 complex_add(float2 ab, float2 cd){
	
	return(make_float2(ab.x + cd.x, ab.y + cd.y));
}

__device__ float2 complex_mult(float2 ab, float2 cd){
	
	float2 ans = make_float2(ab.x * cd.x - ab.y * cd.y, ab.x * cd.y + ab.y * cd.x);
	return(ans);
}


__global__ void generateSpectrumKernel(
				float2 *h0, 
				float2 *ht,
				unsigned int in_width,
				unsigned int out_width,
				unsigned int out_height,
				float animTime,
				float patchSize)
{

	#define RRJ_PI 3.1415926535f

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int in_index = y * in_width + x;

	// Mirror Index
	unsigned int in_mindex = (out_height - y) * in_width + (out_width - x);

	unsigned int out_index = y * out_width + x;


	float2 k;
	k.x = (-(int)out_width / 2.0f + x) * (2.0f * RRJ_PI / patchSize);
	k.y = (-(int)out_width / 2.0f + y) * (2.0f * RRJ_PI / patchSize);

	float k_len = sqrt(k.x * k.x + k.y * k.y);
	float w = sqrt(9.81f * k_len);


	if((x < out_width) && (y < out_height)){

		float2 h0_k = h0[in_index];
		float2 h0_mk = h0[in_mindex];

		ht[out_index] = complex_add(
						complex_mult(h0_k, complex_exp(w * animTime)), 
						complex_mult(conjugate(h0_mk), complex_exp(-w * animTime)));
	}
}




 void 
 cudaGenerateSpectrumKernel(
 	float2 *d_h0,				//In
 	float2 *d_ht,				//Out
 	unsigned int in_width,
 	unsigned int out_width,
 	unsigned int out_height,
 	float animTime,
 	float patch)
 {

 	dim3 block(32, 32, 1);
 	dim3 grid(cuda_iDivUp(out_width, block.x), cuda_iDivUp(out_height, block.y), 1);
 	generateSpectrumKernel<<<grid, block>>>(d_h0, d_ht, in_width, out_width, out_height, animTime, patch);
 }








__global__ void updateHeightMapKernel(float *heightMap, float2 *ht, unsigned int width, unsigned int height){
	
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index = y * width + x;

	float signCorrection = ((x + y) & 0x01) ? -1.0f : 1.0f;

	heightMap[index] = ht[index].x * signCorrection;
}



 void 
 cudaUpdateHeightMapKernel(
 	float *d_heightMap,			//Out
 	float2 *d_ht,				//In
 	unsigned int width,
 	unsigned int height)

{
	
	dim3 block(32, 32, 1);
 	dim3 grid(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);

	updateHeightMapKernel<<<grid, block>>>(d_heightMap, d_ht, width, height);

}







__global__ void calculateSlopeKernel(float *heightMap, float2 *slopeOut, unsigned int width, unsigned int height){
	
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index = y * width + x;

	float2 slope = make_float2(0.0f, 0.0f);

	if((x > 0) && (y > 0) && (x < width - 1) && (y < height - 1)){
		slope.x = heightMap[index + 1] - heightMap[index - 1];
		slope.y = heightMap[index + width] - heightMap[index - width];
	}

	slopeOut[index] = slope;
}



 void 
 cudaCalculateSlopeKernel(
 	float *d_heightMap,			//In
 	float2 *d_slope,				//Out
 	unsigned int width,
 	unsigned int height)

{
	dim3 block(32, 32, 1);
 	dim3 grid(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);

	calculateSlopeKernel<<<grid, block>>>(d_heightMap, d_slope, width, height);
}

