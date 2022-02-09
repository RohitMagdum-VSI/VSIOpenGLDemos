#version 450 core

layout (location = 0) in vec4 vPosition;
layout (location = 2) in vec3 vNormal;

uniform mat4 u_model_matrix;
uniform mat4 u_view_matrix;
uniform mat4 u_projection_matrix;

out vec3 transformed_normals;
out vec4 FragPos;

void main(void)
{
	vec4 fragposition = u_model_matrix * vPosition;
	FragPos = fragposition;
	transformed_normals =  mat3(transpose(inverse(u_model_matrix))) * vNormal;
	gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;
}
