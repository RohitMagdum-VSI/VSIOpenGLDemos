#version 450 core

layout (location = 0) in vec3 aPosition;

uniform mat4 uMVPMatrix;

void main(void)
{
	gl_Position = uMVPMatrix * vec4(aPosition, 1.0);
}