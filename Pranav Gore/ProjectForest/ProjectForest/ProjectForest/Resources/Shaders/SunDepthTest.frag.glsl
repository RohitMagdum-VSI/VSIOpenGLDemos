#version 450 core

in vec2 out_texcoord;
out vec4 FragColor;

uniform sampler2D SunTexture;
uniform sampler2D DepthTexture;

void main(void)
{
	vec4 color;
	if (texture2D(DepthTexture, out_texcoord).r < 1.0)
	{
		color = vec4(vec3(0.0), 1.0);
	}
	else
	{
		color = vec4(texture2D(SunTexture, out_texcoord).rgb, 1.0);
	}
	
	FragColor = color;
}
