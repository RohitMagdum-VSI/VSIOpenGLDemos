#version 450 core

layout (location = 0) out vec3 Position;
layout (location = 1) out vec3 Normal;
layout (location = 2) out vec4 AlbedoSpecular;

in VS_OUT
{
	vec3 FragPos;
	vec2 TexCoords;
	vec3 Normal;
} fs_in;


uniform sampler2D TextureDiffuse1;
uniform sampler2D TextureSpecular1;

void main(void)
{
	// store the fragment position vector in the first gbuffer texture
	Position = fs_in.FragPos;

	// store the per fragment normals into gbuffer
	Normal = normalize(fs_in.Normal);

	// store diffuse per fragment color
	AlbedoSpecular.rgb = texture(TextureDiffuse1, fs_in.TexCoords).rgb;

	// store specular intensity	
	AlbedoSpecular.a = texture(TextureSpecular1, fs_in.TexCoords).r;
}