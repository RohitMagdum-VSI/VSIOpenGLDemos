#version 450 core

layout (location = 0) in vec3 vPosition;
layout (location = 2) in vec3 vNormal;
layout (location = 3) in vec2 vTexCoord0;
layout (location = 4) in vec3 vTangent;

out VS_OUT
{
	vec2 TexCoords0;
	vec3 FragPos;
	vec3 FragPosWorldCoord;
	vec3 FragPosEyeCoord;
	vec3 TangentLightDir;
	vec3 TangentViewPos;
	vec3 TangentFragPos;
	vec4 ShadowCoord;
} vs_out;

uniform mat4 u_model_matrix;
uniform mat4 u_view_matrix;
uniform mat4 u_projection_matrix;
uniform mat4 uShadowMatrix;

uniform vec3 u_view_position;
uniform vec3 u_light_direction;

void main(void)
{
	vs_out.FragPos = vec3(u_model_matrix * vec4(vPosition, 1.0));
	vs_out.FragPosWorldCoord = vPosition;
	vs_out.FragPosEyeCoord = vec3(u_view_matrix * u_model_matrix * vec4(vPosition, 1.0));
	vs_out.TexCoords0 = vTexCoord0;

	// Calculate ShadowCoord
	vec4 WorldPos = u_model_matrix * vec4(vPosition, 1.0);
	vs_out.ShadowCoord = uShadowMatrix * WorldPos;

	mat3 NormalMatrix = transpose(inverse(mat3(u_model_matrix)));

	vec3 N = normalize(NormalMatrix * vNormal);
	vec3 T = normalize(NormalMatrix * vTangent);
	T = normalize(T - dot(T, N) * N);
	vec3 B = cross(N, T);

	mat3 TBN = transpose(mat3(T, B, N));

	vs_out.TangentLightDir = TBN * u_light_direction;
	vs_out.TangentViewPos = TBN * u_view_position;
	vs_out.TangentFragPos = TBN * vs_out.FragPos;

	gl_Position = u_projection_matrix * u_view_matrix * vec4(vs_out.FragPos, 1.0);
}
