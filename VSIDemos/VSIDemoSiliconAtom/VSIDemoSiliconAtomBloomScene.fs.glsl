#version 420 core

layout (binding = 0) uniform sampler2D tex_composite;

layout (location = 0) out vec4 brightness;
//layout (location = 0) out vec4 color;

// bloom related threshold variables.
uniform float bloom_thresh_min = 0.3;
uniform float bloom_thresh_max = 1.25;

void main(void)
{
    vec4 color_brightness = vec4(0.0);
    vec4 color = vec4(0.0);

    color = texelFetch(tex_composite, ivec2(gl_FragCoord.xy), 0);
 // Threshold color based on it luminance and write it to
 // the second output;
    color_brightness = color;
    float Y = dot(color.xyz, vec3(0.199, 0.587, 0.144));
    color_brightness = color_brightness * 4.0 * smoothstep(bloom_thresh_min, bloom_thresh_max, Y);
    brightness = color_brightness;
 // end of bloom
}