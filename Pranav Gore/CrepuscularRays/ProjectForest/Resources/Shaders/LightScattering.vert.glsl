#version 450 core

layout (location = 0) in vec3 aPosition;
layout (location = 3) in vec2 aTexCoords;

out vec2 TexCoords;

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

void main()
{
	TexCoords = aTexCoords;
	gl_Position = uProjectionMatrix * uViewMatrix * uModelMatrix * vec4(aPosition, 1.0f);
}
