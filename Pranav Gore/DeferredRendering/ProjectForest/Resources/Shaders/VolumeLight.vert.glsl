#version 450 core

layout (location = 0) in vec3 a_Position; // vertex location
layout (location = 3) in vec2 a_TexCoord; // vertex texture coordinate

uniform mat4 u_ModelViewMatrix; // convert the vertex from model space to screen space

out vec2 TexCoord;

void main(void)
{
	TexCoord = a_TexCoord;
	gl_Position = u_ModelViewMatrix * vec4(a_Position, 1.0);
}
