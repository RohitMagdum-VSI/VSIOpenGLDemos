#version 450 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D Texture;
uniform int Width;
uniform float odh;

void main(void)
{
	vec3 Color = vec3(0.0);
	int wp1 = Width + 1;
	float Sum = 0.0;
	
	for (int y = -Width; y <= Width; y++)
	{
		float width = (wp1 - abs(float(y)));
		Color += texture2D(Texture, TexCoords + vec2(0.0, odh * y)).rgb * width;
		Sum += width;
	}
	
	FragColor = vec4(Color / Sum, 1.0);
}
