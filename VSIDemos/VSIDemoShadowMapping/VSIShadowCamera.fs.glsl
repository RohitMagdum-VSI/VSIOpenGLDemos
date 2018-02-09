#version 430 core

layout (location = 0) out vec4 color;

layout (binding = 0) uniform sampler2DShadow shadow_tex;

in VS_OUT
{
    vec4 shadow_coord;
    vec3 N;
    vec3 L;
    vec3 V;
} fs_in;

uniform vec3 diffuse_albedo = vec3(0.9, 0.8, 1.0);
uniform vec3 specular_albedo = vec3(0.7);
uniform float specular_power = 300.0;
uniform bool full_shading = true;

void main(void)
{
    vec3 N = normalize(fs_in.N);
    vec3 L = normalize(fs_in.L);
    vec3 V = normalize(fs_in.V);

    vec3 R = reflect(-L, N);

    vec3 diffuse = max(dot(N, L), 0.0) * diffuse_albedo;
    vec3 specular = pow(max(dot(R, V), 0.0), specular_power) * specular_albedo;
    color = textureProj(shadow_tex, fs_in.shadow_coord) * mix(vec4(1.0), vec4(diffuse + specular, 1.0), bvec4(full_shading));
}