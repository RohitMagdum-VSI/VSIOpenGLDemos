#version 410 core

in vec2 TexCoords;
out vec4 Fragcolor;
uniform sampler2D texture_diffuse;

void main( )
{
    Fragcolor = vec4( texture( texture_diffuse, TexCoords ));
	
}
