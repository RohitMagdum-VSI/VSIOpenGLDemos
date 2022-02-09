#version 450 core

in vec2 OutTexCoord;
out vec4 FragColor;

uniform sampler2D Texture;

void main(void)
{
	FragColor = texture(Texture, OutTexCoord);
}
