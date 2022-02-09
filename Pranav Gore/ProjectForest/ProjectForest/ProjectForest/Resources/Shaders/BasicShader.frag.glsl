#version 450 core

in vec4 OutColor;
out vec4 FragColor;

void main(void)
{
	FragColor = OutColor;
	//FragColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
}
