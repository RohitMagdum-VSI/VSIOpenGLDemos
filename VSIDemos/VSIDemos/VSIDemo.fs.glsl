#version 430 core

out vec4 color;

in VS_OUT
{ 
    
    vec2 texCoord;
    vec3 color;
} fs_in;

uniform sampler2D tex;

void main(void)
{ 
    color =	texture(tex, fs_in.texCoord);
    color += vec4(fs_in.color, 1.0f);
}