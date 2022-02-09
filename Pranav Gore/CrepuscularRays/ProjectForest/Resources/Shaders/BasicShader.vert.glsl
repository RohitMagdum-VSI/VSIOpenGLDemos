#version 450 core

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec4 vColor;

out vec4 OutColor;
uniform mat4 uMVPMatrix;

void main(void)
{
	gl_Position = uMVPMatrix * vec4(vPosition, 1.0);
	OutColor = vColor;
}
