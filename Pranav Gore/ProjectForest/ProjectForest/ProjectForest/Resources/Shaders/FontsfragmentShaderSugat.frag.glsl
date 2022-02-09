#version 450

in vec2 out_textureCoord;

out vec4 FragColor;

uniform sampler2D texture0_sampler_u;
uniform vec3 textColor_u;

void main(void){
	vec4 sampled = vec4(1.0,1.0,1.0,texture(texture0_sampler_u,out_textureCoord).r);
	
	FragColor = vec4(textColor_u,1.0)* sampled;
	
}
