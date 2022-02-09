#version 450 core

in vec4 out_color;
in vec2 out_texture0_coord;

out vec4 FragColor;

uniform sampler2D u_texture0_bg_sampler; 
uniform sampler2D u_texture1_r_sampler;  
uniform sampler2D u_texture2_g_sampler;  
uniform sampler2D u_texture3_b_sampler;  
uniform sampler2D u_texture4_blendmap_sampler;

void main(void)
{
	vec4 blendMapColor = texture(u_texture4_blendmap_sampler,out_texture0_coord); 
	float backTextureAmount = 1 - (blendMapColor.r + blendMapColor.g + blendMapColor.b); 
	vec2 tiledCoords = out_texture0_coord * 10;

	vec4 backgroundTextureColor = texture(u_texture0_bg_sampler, tiledCoords)* backTextureAmount;
	vec4 rTextureColor = texture(u_texture1_r_sampler, tiledCoords) *blendMapColor.r;
	vec4 gTextureColor = texture(u_texture2_g_sampler, tiledCoords) *blendMapColor.g;
	vec4 bTextureColor = texture(u_texture3_b_sampler, tiledCoords) *blendMapColor.b;

	vec4 totalColor = backgroundTextureColor + rTextureColor + gTextureColor + bTextureColor;
	
	FragColor = totalColor;

	//FragColor = texture(u_texture0_bg_sampler, out_texture0_coord);
}
