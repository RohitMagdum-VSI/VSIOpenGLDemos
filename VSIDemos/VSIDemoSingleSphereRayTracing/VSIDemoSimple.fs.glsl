#version 430 core

out vec4 color;

layout (binding = 0) uniform sampler2D tex_composite;

struct ray
{
    vec3 origin;
    float f;
    vec3 direction;
};

in VS_OUT
{
    vec3    sphere_center;
    float   radius;
} fs_in;

struct sphere
{
    vec3 center;
    float radius;
    vec4 color;
};

layout (std140, binding = 0) uniform RAYS
{
    ray r[4000];
};

float intersect_ray_sphere(ray R,
                           sphere S,
                           out vec3 hitpos,
                           out vec3 normal)
{
    vec3 v = R.origin - S.center;
    float B = 2.0 * dot(R.direction, v);
    float C = dot(v, v) - S.radius * S.radius;
    float B2 = B * B;

    float f = B2 - 4.0 * C;

    if(f < 0.0)
        return 0.0;

    f = sqrt(f);
    float t0 = -B + f;
    float t1 = -B - f;
    float t = min(max(t0, 0.0), max(t1, 0.0)) * 0.5;

    if(t == 0.0)
        return 0.0;

    hitpos = R.origin + t * R.direction;
    normal = normalize(hitpos - S.center);

    return t;
}

void main(void)
{
    ray R;

    sphere S;
    
    S.center = fs_in.sphere_center;
    S.radius = fs_in.radius;
    S.color = vec4(1.0, 1.0, 1.0, 1.0);
    
    //R.origin = fs_in.ray_origin;
    //R.direction = fs_in.ray_direction;
    
    vec3 hit_position = vec3(0.0);
    vec3 hit_normal = vec3(0.0);
    
    //R.origin += R.direction *0.01;
    
    ray refl;
    ray refr;
    vec3 hitpos;
    vec3 normal;
    float min_t = 1000000.0f;
    int i;
    int sphere_index = 0;
    float t;
    
    for(i = 0; i < 3500; i++)
    {
        ray rr;
        rr.origin = r[i].origin;
        rr.direction = r[i].direction;
        //rr.origin = rr.direction * 0.01;
        t = intersect_ray_sphere(rr, S, hitpos, normal);
        if(t != 0.0)
        {
            if(t < min_t)
            {
                min_t = t;
                hit_position = hitpos;
                hit_normal = normal;
                sphere_index = i;
                color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        }
    }
    
  //  color = texelFetch(tex_composite, ivec2(gl_FragCoord.xy), 0);
  //color = vec4(1.0,1.0, 0.0, 1.0);
}