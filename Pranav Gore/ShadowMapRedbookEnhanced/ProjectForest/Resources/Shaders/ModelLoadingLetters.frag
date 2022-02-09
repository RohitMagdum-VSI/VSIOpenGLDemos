#version 330 core

in vec2 TexCoords;
out vec4 FragColor;

uniform vec4 u_model_color;
uniform sampler2D texture_diffuse;
uniform sampler2D texture_opacity;

void main( )
{
	//FragColor = vec4( texture( texture_diffuse, TexCoords ));	
	FragColor = u_model_color;	
	
	//if(FragColor.a < 0.1)
	//{
	//	discard;
	//}
	
	//vec4 t0 = vec4( texture( texture_diffuse, TexCoords ));	
	//vec4 t1 = vec4( texture( texture_opacity, TexCoords ));	
	
	//FragColor = mix(t0, t1, t1.a);
}