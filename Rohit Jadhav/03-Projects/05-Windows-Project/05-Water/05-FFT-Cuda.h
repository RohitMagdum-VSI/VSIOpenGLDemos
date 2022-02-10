 

void Generate_H0(float2 *d_h0);

 void 
 cudaGenerateSpectrumKernel(
 	float2 *d_h0,				//In
 	float2 *d_ht,				//Out
 	unsigned int in_width,
 	unsigned int out_width,
 	unsigned int out_height,
 	float animTime,
 	float patch);


 void 
 cudaUpdateHeightMapKernel(
 	float *d_heightMap,			//Out
 	float2 *d_ht,				//In
 	unsigned int width,
 	unsigned int height);


 void 
 cudaCalculateSlopeKernel(
 	float *d_heightMap,			//In
 	float2 *d_slope,				//Out
 	unsigned int width,
 	unsigned int height);


