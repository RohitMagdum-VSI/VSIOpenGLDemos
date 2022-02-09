#version 450 core

out vec4 FragColor;

layout (location = 3) in vec2 out_texcoord;

uniform sampler2D HDRTexture;
uniform float exposure;

void main(void)
{
	const float gamma = 2.2;
	vec3 HDRColor = texture(HDRTexture, out_texcoord).rgb;

	// Reinhard
	// vec3 result = HDRColor / (HDRColor + vec3(1.0));

	// exposure
	vec3 result = vec3(1.0) - exp(-HDRColor * exposure);

	// gamma correction
	result = pow(result, vec3(1.0 / gamma));
	FragColor = vec4(result, 1.0);
	//FragColor = texture(HDRTexture, out_texcoord);
}
