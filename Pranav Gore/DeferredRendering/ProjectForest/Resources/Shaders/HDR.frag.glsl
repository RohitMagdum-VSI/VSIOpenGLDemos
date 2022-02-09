#version 450 core

out vec4 FragColor;

in vec2 out_texcoord;

uniform sampler2D HDRTexture;
uniform float exposure;

const float gamma = 2.2;

// for Uncharted 2 tonemap
const float A = 0.15;
const float B = 0.50;
const float C = 0.10;
const float D = 0.20;
const float E = 0.02;
const float F = 0.30;
const float W = 11.2;
//const float W = 1000.0;

vec3 ReinhardTonemap(vec3 TexColor);
vec3 ExposureBasedTonemap(vec3 TexColor);
vec3 Uncharted2Tonemap(vec3 TexColor);

void main(void)
{
	vec3 HDRColor = texture(HDRTexture, out_texcoord).rgb;

	// If using Reinhard or Exposure based tonemap then use following two lines only
	vec3 FinalColor = ExposureBasedTonemap(HDRColor);
	FragColor = vec4(FinalColor, 1.0);
	//FragColor = vec4(HDRColor, 1.0);

	// Uncharted 2 Tonemap algorithm
	// Hardcoded Exposure adjustment
	//HDRColor *= 16;

	//float ExposureBias = 2.0;
	//vec3 Color = Uncharted2Tonemap(ExposureBias * HDRColor);
	
	//vec3 WhiteScale = 1.0 / Uncharted2Tonemap(vec3(W));

	//vec3 TonemappedColor = Color * WhiteScale;

	// Gamma Correction
	//vec3 FinalColor = pow(TonemappedColor, vec3(1.0 / gamma));
	//FragColor = vec4(FinalColor, 1.0);

	//FragColor = vec4(TonemappedColor, 1.0);
}



vec3 ReinhardTonemap(vec3 TexColor)
{
	// Reinhard tone mapping
	vec3 result = TexColor / (TexColor + vec3(1.0));

	// gamma correction
	result = pow(result, vec3(1.0 / gamma));
	
	return(result);
}


vec3 ExposureBasedTonemap(vec3 TexColor)
{
	// exposure based tone mapping
	vec3 result = vec3(1.0) - exp(-exposure * TexColor);

	// gamma correction
	result = pow(result, vec3(1.0 / gamma));
	
	return(result);
}

vec3 Uncharted2Tonemap(vec3 TexColor)
{
	return ((TexColor * (A * TexColor + C * B) + D * E) / (TexColor * (A * TexColor + B) + D * F)) - E / F;
}
