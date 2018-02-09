#version 430 core

layout (location = 0) in vec4 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoords;

uniform mat4 mvMatrix;
uniform mat4 projMatrix;

out VS_OUT
{
    vec3 N;
    vec3 L;
    vec3 V;
}vs_out;

uniform vec3 lightPosition = vec3(100.0, -100.0, -100.0);

void main(void)
{
    vec4 P = mvMatrix * position;

    vs_out.N = mat3(mvMatrix) * normal;

    vs_out.L = lightPosition - P.xyz;

    vs_out.V = -P.xyz;

    gl_Position = projMatrix * P;
}