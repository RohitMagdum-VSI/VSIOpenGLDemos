#version 450 core

in vec3 transformed_normals;
in vec4 FragPos;

uniform vec3 skyColor;
uniform vec3 groundColor;
uniform vec4 u_light_position;

out vec4 FragColor;

void main(void)
{
	vec3 normalized_normal = normalize(transformed_normals);
	vec3 light_direction = normalize(vec3(u_light_position) - FragPos.xyz);
	float cosTheta = dot(normalized_normal, light_direction);
	float a = cosTheta * 0.5 + 0.5;
	FragColor = vec4 (mix(groundColor, skyColor, a), 1.0);
}
