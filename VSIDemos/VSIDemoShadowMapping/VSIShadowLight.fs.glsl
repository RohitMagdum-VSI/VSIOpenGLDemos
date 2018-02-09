#version 430 core

layout (location = 0) out vec4 color;

void main(void)
{
    color = vec4(gl_FragCoord.z);
}