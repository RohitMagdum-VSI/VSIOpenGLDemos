#version 450 core

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec4 vColor;
layout (location = 2) in vec3 vNormal;
layout (location = 3) in vec2 vTexCoord0;	

out VS_OUT
{
	vec2 TexCoords0;
	vec3 FragPos;
	vec3 FragPosWorldCoord;
	vec3 FragPosEyeCoord;
	vec3 Normal;
	vec4 ShadowCoord;
	vec4 OutColor;
} vs_out;

uniform mat4 u_model_matrix;
uniform mat4 u_view_matrix;
uniform mat4 u_projection_matrix;
uniform mat4 uShadowMatrix;

void main(void)
{
	vs_out.FragPos = vec3(u_model_matrix * vec4(vPosition, 1.0));
	vs_out.FragPosWorldCoord = vPosition;
	vs_out.FragPosEyeCoord = vec3(u_view_matrix * u_model_matrix * vec4(vPosition, 1.0));
	//vs_out.FragPosEyeCoord = vec3(u_view_matrix * vec4(vPosition, 1.0));
	vs_out.Normal = mat3(transpose(inverse(u_model_matrix))) * vNormal;
	vs_out.TexCoords0 = vTexCoord0;
	vs_out.OutColor = vColor;

	// Calculate ShadowCoord
	vec4 WorldPos = u_model_matrix * vec4(vPosition, 1.0);
	vs_out.ShadowCoord = uShadowMatrix * WorldPos;

	gl_Position = u_projection_matrix * u_view_matrix * vec4(vs_out.FragPos, 1.0);
}
