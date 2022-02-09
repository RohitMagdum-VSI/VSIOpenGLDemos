#version 450 core

layout (location = 0) in vec3 aPosition;
layout (location = 2) in vec3 aNormal;
layout (location = 3) in vec2 aTexCoords;	

out VS_OUT
{
	vec3 FragPos;
	vec2 TexCoords;
	vec3 Normal;
} vs_out;

uniform mat4 u_model_matrix;
uniform mat4 u_view_matrix;
uniform mat4 u_projection_matrix;

void main(void)
{
	vs_out.FragPos = vec3(u_model_matrix * vec4(aPosition, 1.0));
	vs_out.TexCoords = aTexCoords;
	vs_out.Normal = mat3(transpose(inverse(u_model_matrix))) * aNormal;

	gl_Position = u_projection_matrix * u_view_matrix * vec4(vs_out.FragPos, 1.0);
}