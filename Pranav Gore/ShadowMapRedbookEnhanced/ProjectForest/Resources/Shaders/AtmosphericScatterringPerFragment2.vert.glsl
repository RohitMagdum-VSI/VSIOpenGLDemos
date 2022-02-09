#version 450 core

layout (location = 0) in vec3 vPosition;
out vec3 Position;

uniform mat4 u_mvp_matrix;

void main(void)
{
	Position = vPosition;
	gl_Position = u_mvp_matrix * vec4(vPosition, 1.0);
	
}
