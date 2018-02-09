#version 430 core


layout (location = 0) in vec4 position;
layout (location = 2) in vec3 normal;
layout (location = 1) in vec2 texCoord;

out VS_OUT
{
	vec2 texCoord;
	vec3 color;
} vs_out;

uniform mat4 mv_matrix;
uniform mat4 proj_matrix;
uniform mat4 view_matrix;

uniform vec3 light_pos = vec3(100.0, 100.0, 100.0);
uniform vec3 diffuse_albedo = vec3(0.2, 0.2, 0.2);
uniform vec3 specular_albedo = vec3(0.5, 0.5, 0.5);
uniform float specular_power = 128.0;
uniform vec3 ambient = vec3(0.1, 0.1, 0.1);

void main(void)
{ 
	vec4 P =  mv_matrix * position;

	vec3 N = mat3(mv_matrix) * normal;

	vec3 L = light_pos - P.xyz;

	vec3 V = -P.xyz;

	N = normalize(N);
	L = normalize(L);
	V = normalize(V);

	vec3 R = reflect(-L, N);

	vec3 diffuse = max(dot(N,L), 0.0) * diffuse_albedo;
	vec3 specular = pow(max(dot(R, V), 0.0), specular_power) *
	specular_albedo;

	vs_out.color = ambient + diffuse + specular;

	gl_Position = proj_matrix * P;
	vs_out.texCoord = texCoord;
}