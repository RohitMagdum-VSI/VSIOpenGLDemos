
__device__ float fSoftenSq;


__device__ float3 
bodyBodyInteraction(float3 ai, float4 bi, float4 bj){
	

	float3 r;

	r.x = bi.x - bj.x;
	r.y = bi.y - bj.y;	
	r.z = bi.z - bj.z;

	float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
	distSqr = distSqr + fSoftenSq;

	float distSixth = distSqr * distSqr * distSqr;
	float invDistCube = 1.0f / sqrtf(distSixth);

	float s = bj.w * invDistCube;

	ai.x = ai.x + r.x * s;
	ai.y = ai.y + r.y * s;
	ai.z = ai.z + r.z * s;

	return(ai);
}




__device__ float3
gravitation(float4 pos, float3 acc){
	

	extern __shared__ float4 sharedPos[];

	for(int i = 0; i < blockDim.x; ){

		acc = bodyBodyInteraction(acc, sharedPos[i + blockDim.x * threadIdx.y], pos);
		i++;

		acc = bodyBodyInteraction(acc, sharedPos[i + blockDim.x * threadIdx.y], pos);
		i++;

		acc = bodyBodyInteraction(acc, sharedPos[i + blockDim.x * threadIdx.y], pos);
		i++;

		acc = bodyBodyInteraction(acc, sharedPos[i + blockDim.x * threadIdx.y], pos);
		i++;

	}

	return(acc);

}


#define WARP(x, m) (((x) < m) ? x : (x - m))



__device__ float3 
calculateBodyForce(float4 pos, float4 *pOldPos, unsigned int numParticales)
{
	
	extern __shared__ float4 sharedPos[];

	float3 acc = {0.0f, 0.0f, 0.0f};

	int p = blockDim.x;
	int q = blockDim.y;
	int n = numParticales;


	int start = n / q * threadIdx.y;
	int tile0 = start / (n / q);
	int tile = tile0;
	int finish = start + n / q;


	for(int i = start; i < finish; i+=p, tile++){

		sharedPos[threadIdx.x + blockDim.x * threadIdx.y] = pOldPos[WARP(blockIdx.x + tile, gridDim.x) * blockDim.x + threadIdx.x];

		__syncthreads();


		acc = gravitation(pos, acc);

		__syncthreads();
	}

	return(acc);
}





__global__ void 
AnimateNBodyKernel(
	float4 *pNewPos, 
	float4 *pNewVel,
	float4 *pOldPos,
	float4 *pOldVelo,
	float fSofting,
	float fDumping,
	float fAnimTime,
	unsigned int numParticales)
{
	
	int index1 = blockIdx.x * blockDim.x + threadIdx.x;

	if(index1 > numParticales)
		return;
	

	fSoftenSq = fSofting * fSofting;

	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	float4 pos = pOldPos[index];
	
	float3 force = calculateBodyForce(pos, pOldPos, numParticales);

	float forceMag = sqrtf(force.x * force.x + force.y * force.y + force.z * force.z);

	float4 velo = pOldVelo[index];

	velo.x = velo.x + force.x * fAnimTime;
	velo.y = velo.y + force.y * fAnimTime;
	velo.z = velo.z + force.z * fAnimTime;

	velo.x *= fDumping;
	velo.y *= fDumping;
	velo.z *= fDumping;
	

	pos.x = pos.x + velo.x * fAnimTime;
	pos.y = pos.y + velo.y * fAnimTime;
	pos.z = pos.z + velo.z * fAnimTime;


	//pNewPos[index] = make_float4(pos.x, pos.y, pos.z, 1.0f);
	
	pNewPos[index] = pos;
	pNewVel[index] = velo;
}






void 
AnimateNBody(
	float *pNewPos,
	float *pNewVel,
	float *pOldPos,
	float *pOldVelo,
	float fSofting,
	float fDumping,
	float fAnimTime,
	unsigned int numParticales)
{
	
	int sharedMemSize = 256 * sizeof(float4);

	dim3 block(256, 1, 1);
	dim3 grid(numParticales / block.x, 1, 1);


	AnimateNBodyKernel<<<grid, block, sharedMemSize>>>(
									(float4*)pNewPos, (float4*)pNewVel,
									(float4*)pOldPos, (float4*)pOldVelo,
									fSofting, fDumping, fAnimTime, numParticales);	

}