#version 450 core

layout (location = 0) in vec3 vPosition;
layout (location = 3) in vec2 vTexCoords;

out vec2 out_texcoord;

void main(void)
{
	out_texcoord = 	vTexCoords;
	gl_Position = vec4(vPosition, 1.0);
}
