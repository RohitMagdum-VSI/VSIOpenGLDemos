#version 450							
								
layout (location = 0) in vec4 vPosition;			

out vec2 out_textureCoord;

uniform mat4 u_mvp_matrix;	

void main(void)						
{										
	gl_Position = u_mvp_matrix * vec4(vPosition.xy,0.0,1.0);
	out_textureCoord = vPosition.zw ;
}	