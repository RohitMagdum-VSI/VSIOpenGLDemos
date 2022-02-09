#version 450 core
		
layout (location = 0) in vec4 vPosition;
layout (location = 3) in vec2 vTexCoord0;

out vec2 out_texcoord;

void main()
{
	out_texcoord = vTexCoord0;
	gl_Position = vPosition * 2.0 - 1.0;
}
