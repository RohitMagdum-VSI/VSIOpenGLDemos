#version 450 core

out vec4 FragColor;

in vec3 TexCoords;

uniform samplerCube Cubemap;

void main()
{
	FragColor = texture(Cubemap, TexCoords);
}