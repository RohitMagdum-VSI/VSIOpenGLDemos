#version 430 core

uniform float aspect = 0.75;

out VS_OUT
{
    vec3    sphere_center;
    float   radius;
} vs_out;

void main(void)
{
    vec4 vertices[4] = vec4[4](vec4(-1.0, -1.0, 0.5, 1.0),
                               vec4( 1.0, -1.0, 0.5, 1.0),
                               vec4(-1.0,  1.0, 0.5, 1.0),
                               vec4( 1.0,  1.0, 0.5, 1.0));
    vec4 pos = vertices[gl_VertexID];

    gl_Position = pos;

    vs_out.sphere_center = vec3(0.0, 0.0, 0.0);
    vs_out.radius = 4.0;
}
