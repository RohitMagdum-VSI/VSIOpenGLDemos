#version 420 core

uniform mat4 MVMatrix;
uniform mat4 ProjMatrix;

out vec3 MCPosition;

layout (location = 0) in vec4 position;
layout (location = 2) in vec3 normal;
layout (location = 1) in vec2 texCo;

void main()
{
    vec4 P = MVMatrix * position;

	gl_Position = ProjMatrix * P;
    MCPosition = vec3(position) * 0.3   ;
}