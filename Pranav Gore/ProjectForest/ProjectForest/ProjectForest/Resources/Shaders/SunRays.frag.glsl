#version 450 core

in vec2 out_texcoord;
out vec4 FragColor;

uniform sampler2D LowBlurredSunTexture;
uniform sampler2D HighBlurredSunTexture;
uniform float Intensity;
uniform vec2 SunPosProj;
uniform vec3 Distortion;

void main(void)
{
	vec3 RadialBlur = vec3(0.0);
	vec2 TexCoord = out_texcoord;
	int RadialBlurSamples = 128;

	vec2 RadialBlurVector = (SunPosProj - TexCoord) / RadialBlurSamples;
	
	for (int i = 0; i < RadialBlurSamples; i++)
	{
		RadialBlur += texture2D(LowBlurredSunTexture, TexCoord).rgb;
		TexCoord += RadialBlurVector;
	}

	RadialBlur /= RadialBlurSamples;
	FragColor = vec4((texture2D(HighBlurredSunTexture, out_texcoord).rgb + RadialBlur) * Intensity, 1.0);
}
