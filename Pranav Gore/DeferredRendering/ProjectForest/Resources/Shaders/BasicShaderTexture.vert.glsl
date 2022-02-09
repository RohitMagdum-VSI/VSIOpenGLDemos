#version 450 core

layout (location = 0) in vec3 vPosition;
layout (location = 3) in vec2 vTexCoords;

out vec2 out_texcoord;
uniform mat4 u_mvp_matrix;

void main(void)
{
	gl_Position = u_mvp_matrix * vec4(vPosition, 1.0);
	out_texcoord = vTexCoords;
}
