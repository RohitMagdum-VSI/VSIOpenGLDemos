
__kernel void interop(__global float4 *arr, int width, int height, float animTime){

	int x = get_global_id(0);
	int y = get_global_id(1);

	float u, v, w;

	u = x / (float)width;
	v = y / (float)height;


	float freq = 4.0f;
	u = u * 2.0f - 1.0f;
	v = v * 2.0f - 1.0f;

	w = sin(freq * u + animTime) * cos(freq * v + animTime) * 0.5f;

	arr[x * width + y] = (float4)(u, w, v, 1.0);

}
