#version 450 core

out vec4 FragColor;	
in vec4 out_color;

void main(void)
{
	FragColor = out_color;
}
