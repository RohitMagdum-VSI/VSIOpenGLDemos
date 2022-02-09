#version 450 core

layout (location = 0) in vec4 aPosition;
//layout (location = 3) in vec2 aTextureCoord;

uniform mat4 uModelViewProjection;
out vec4 vPosition;

void main(void)
{
	vPosition = aPosition;
	gl_Position = uModelViewProjection * aPosition;
}