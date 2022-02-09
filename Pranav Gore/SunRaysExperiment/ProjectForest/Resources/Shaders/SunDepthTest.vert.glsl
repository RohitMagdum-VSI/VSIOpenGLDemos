#version 450 core

layout (location = 0) in vec4 aPosition;
layout (location = 3) in vec2 aTexCoords;

out vec2 TexCoords;

void main()
{
	TexCoords = aTexCoords;
	gl_Position = aPosition;
}
