#version 450 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D SunTexture;
uniform sampler2D SunDepthTexture;
uniform sampler2D DepthTexture;

void main(void)
{
	vec4 color;
	if (texture2D(DepthTexture, TexCoords).r < texture2D(SunDepthTexture, TexCoords).r)
	{
		color = vec4(vec3(0.0), 1.0);
	}
	else
	{
		color = vec4(texture2D(SunTexture, TexCoords).rgb, 1.0);
	}
	
	FragColor = color;
}
