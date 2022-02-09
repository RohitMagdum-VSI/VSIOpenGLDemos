#version 450 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D LowBlurredSunTexture;
uniform sampler2D HighBlurredSunTexture;

uniform float Dispersal;
uniform float HaloWidth;
uniform float Intensity;

uniform vec2 SunPosProj;
uniform vec3 Distortion;

vec3 texture2DDistorted(sampler2D Texture, vec2 TexCoord, vec2 Offset);

void main(void)
{
	vec3 RadialBlur = vec3(0.0);
	vec2 TexCoord = TexCoords;
	int RadialBlurSamples = 128;

	vec2 RadialBlurVector = (SunPosProj - TexCoord) / RadialBlurSamples;
	
	for (int i = 0; i < RadialBlurSamples; i++)
	{
		RadialBlur += texture2D(LowBlurredSunTexture, TexCoord).rgb;
		TexCoord += RadialBlurVector;
	}

	RadialBlur /= RadialBlurSamples;

	vec3 LensFlareHalo = vec3(0.0);
	TexCoord = 1.0 - TexCoords;
	vec2 LensFlareVector = (vec2(0.5) - TexCoord) * Dispersal;
	vec2 LensFlareOffset = vec2(0.0);

	for (int i = 0; i < 5; i++)
	{
		LensFlareHalo += texture2DDistorted(HighBlurredSunTexture, TexCoord, LensFlareOffset).rgb;
		LensFlareOffset += LensFlareVector;
	}

	LensFlareHalo += texture2DDistorted(HighBlurredSunTexture, TexCoord, normalize(LensFlareVector) * HaloWidth);
	LensFlareHalo /= 6.0;

	FragColor = vec4((texture2D(HighBlurredSunTexture, TexCoords).rgb + (RadialBlur + LensFlareHalo)) * Intensity, 1.0);

	//FragColor = vec4((texture2D(HighBlurredSunTexture, TexCoords).rgb + (RadialBlur)) * Intensity, 1.0);
}

vec3 texture2DDistorted(sampler2D Texture, vec2 TexCoord, vec2 Offset)
{
	return vec3(
		texture2D(Texture, TexCoord + Offset * Distortion.r).r,
		texture2D(Texture, TexCoord + Offset * Distortion.g).g,
		texture2D(Texture, TexCoord + Offset * Distortion.b).b
	);	
}
