#version 450 core

layout (location = 0) in vec4 vPosition;
layout (location = 1) in vec3 vColor;

out vec4 OutColor;
uniform mat4 uMVPMatrix;

void main(void)
{
	gl_Position = uMVPMatrix * vPosition;
	OutColor = vec4(vColor, 1.0f);
}
