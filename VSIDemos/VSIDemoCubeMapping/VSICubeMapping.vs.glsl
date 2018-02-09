#version 430 core

out VS_OUT
{
    vec3 tc;
}vs_out;

uniform mat4 mvMatrix;
uniform mat4 projMatrix;

void main(void)
{
    vec3[4] vertices = vec3[4](vec3(-1.0, -1.0, 1.0),
                               vec3(1.0,  -1.0, 1.0),
                               vec3(-1.0,  1.0, 1.0),
                               vec3(1.0,   1.0, 1.0));
    vs_out.tc = mat3(mvMatrix) * vertices[gl_VertexID];

    gl_Position = vec4(vertices[gl_VertexID], 1.0);
}