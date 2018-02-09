#version 430 core

layout (location = 0) in vec4 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoords;
layout (location = 3) in vec3 tangents;

out VS_OUT
{
    vec2 texcoord;
    vec3 eyeDir;
    vec3 lightDir;
    vec3 normal;
}vs_out;


uniform mat4 mvMatrix;
uniform mat4 projMatrix;
uniform vec3 lightPosition = vec3(100.0, -100.0, -100.0);

void main(void)
{
    vec4 P = mvMatrix * position;

    vec3 V = P.xyz;
    vec3 N = normalize(mat3(mvMatrix) * normal);
    vec3 T = normalize(mat3(mvMatrix) * tangents);

    vec3 B = cross(N, T);

    vec3 L = lightPosition - P.xyz;
    vs_out.lightDir = normalize(vec3(dot(L, T), dot(L, B), dot(L, N)));

    V = -P.xyz;
    vs_out.eyeDir = normalize(vec3(dot(V, T), dot(V, B), dot(V, N)));

    vs_out.texcoord = texCoords;

    vs_out.normal = N;

    gl_Position = projMatrix * P;
}