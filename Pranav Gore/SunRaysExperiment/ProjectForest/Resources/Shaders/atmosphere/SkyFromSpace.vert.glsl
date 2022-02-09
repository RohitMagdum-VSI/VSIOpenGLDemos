#version 450 core

layout (location = 0) in vec4 aPosition;

uniform mat4 uModelViewProjection;
out vec4 vPosition;

void main(void)
{
	vPosition = aPosition;
	gl_Position = uModelViewProjection * aPosition;
}