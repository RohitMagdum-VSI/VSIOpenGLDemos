#version 450 core

out vec4 FragColor;
in vec3 v3Direction;
in vec3 PrimaryColor;
in vec3 SecondaryColor;

uniform vec3 v3LightPos;
uniform float g;
uniform float g2;

void main(void)
{
	float fCos = dot(v3LightPos, v3Direction) / length(v3Direction);
	float fRayleighPhase = 1.0 + fCos * fCos;
	float fMeiPhase = (1.0 - g2) / (2.0 + g2) * (1.0 + fCos * fCos) / pow(1.0 + g2 - 2.0 * g * fCos, 1.5);
	vec3 Color = 1.0 - exp(-1.5 * (fRayleighPhase * PrimaryColor + fMeiPhase * SecondaryColor));
	//FragColor = vec4(pow(Color, vec3(1.0 / 2.2)), 1.0); // Gamma correction
	FragColor = vec4(Color, 1.0);
}
