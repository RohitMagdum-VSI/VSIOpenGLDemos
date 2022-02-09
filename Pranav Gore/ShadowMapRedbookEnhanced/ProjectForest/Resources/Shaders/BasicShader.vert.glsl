#version 450 core

layout (location = 0) in vec4 vPosition;
layout (location = 1) in vec4 vColor;

out vec4 OutColor;
uniform mat4 u_mvp_matrix;

void main(void)
{
	gl_Position = u_mvp_matrix * vPosition;
	OutColor = vColor;
}
