#version 450 core

layout (location = 0) in vec4 vPos;
layout (location = 3) in vec2 vTexture0_Coord;

uniform mat4 u_mvp_matrix;

out vec4 vPosition;

void main(void)
{
	vPosition = vPos;
	gl_Position = u_mvp_matrix * vPos;
}
