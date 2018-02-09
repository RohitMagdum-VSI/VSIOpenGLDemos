#version 430 core


layout (location = 0) in vec4 position;
layout (location = 2) in vec3 normal;
layout (location = 1) in vec2 texCoord;

out VS_OUT
{
	vec2 texCoord;
	vec3 N;
	vec3 L;
	vec3 V;
} vs_out;

uniform vec3 light_pos = vec3(100.0, 100.0, 100.0);

uniform mat4 mv_matrix;
uniform mat4 proj_matrix;
uniform mat4 view_matrix;

void main(void)
{
	vec4 P = mv_matrix * position;

	vs_out.N = mat3(mv_matrix) * normal;

	vs_out.L = light_pos - P.xyz;

	vs_out.V = -P.xyz;

	gl_Position = proj_matrix * P;
	vs_out.texCoord = texCoord;
}