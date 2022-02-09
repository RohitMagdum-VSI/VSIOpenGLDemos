#version 450 core

in vec2 out_texcoord;
out vec4 FragColor;

uniform sampler2D SunTexture;

void main(void)
{
	FragColor = texture(SunTexture, out_texcoord);
}
