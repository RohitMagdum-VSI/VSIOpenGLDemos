#version 450 core				
								
layout (location = 0) in vec3 vPosition;			
layout (location = 1) in vec4 vColor;
layout (location = 2) in vec3 vNormal;
layout (location = 3) in vec2 vTexture0_Coord;	

out vec2 out_texture0_coord;
out vec4 out_color;

uniform mat4 u_model_matrix;
uniform mat4 u_view_matrix;
uniform mat4 u_projection_matrix;

void main(void)						
{										
	out_color = vColor;
	out_texture0_coord = vTexture0_Coord;
	gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vec4(vPosition, 1.0);
}
