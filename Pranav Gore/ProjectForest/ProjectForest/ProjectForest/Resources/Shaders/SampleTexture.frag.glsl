#version 450 core

in vec2 out_texcoord;
out vec4 FragColor;

uniform sampler2D Diffuse;
uniform sampler2D Opacity;

void main(void)
{
	vec4 texcolor = texture(Diffuse, out_texcoord);

	if (texcolor.a < 0.1)
	{
		discard;
	}

	FragColor = texcolor;
	//FragColor = texture(Opacity, out_texcoord);
}
