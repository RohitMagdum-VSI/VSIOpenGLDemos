#version 450 core

layout (location = 0) in vec3 aPosition;
layout (location = 3) in vec2 aTexCoord;

out vec2 OutTexCoord;
uniform mat4 uMVPMatrix;

void main(void)
{
	
	OutTexCoord = aTexCoord;
	gl_Position = uMVPMatrix * vec4(aPosition, 1.0);
	//gl_Position = vec4(aPosition, 1.0);
}
