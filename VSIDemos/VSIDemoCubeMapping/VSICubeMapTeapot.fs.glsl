#version 430 core

uniform samplerCube tex_cubemap;

in VS_OUT
{
    vec3 normal;
    vec3 view;
}fs_in;

out vec4 color;

void main(void)
{
    vec3 r = reflect(fs_in.view, normalize(fs_in.normal));

    color = texture(tex_cubemap, r) * vec4(0.95, 0.80, 0.45, 1.0);
}