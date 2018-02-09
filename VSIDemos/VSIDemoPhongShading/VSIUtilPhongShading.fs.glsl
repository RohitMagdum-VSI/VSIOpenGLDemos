#version 410 core

layout (location = 0) out vec4 color;

in VS_OUT
{
	vec2 texCoord;
	vec3 N;
	vec3 L;
	vec3 V;
}fs_in;

uniform sampler2D tex;

uniform vec3 diffuse_albedo = vec3(0.3, 0.3, 0.3);
uniform vec3 specular_albedo = vec3(0.6, 0.6, 0.6);
uniform float specular_power = 256.0;
uniform vec3 ambient = vec3(0.5, 0.5, 0.5);

void main(void)
{
	vec4 text = texture2D(tex, fs_in.texCoord);

	vec3 N = normalize(fs_in.N);
	vec3 L = normalize(fs_in.L);
	vec3 V = normalize(fs_in.V);

	vec3 R = reflect(-L, N);

	vec3 diffuse = max(dot(N, L), 0.0) * diffuse_albedo;
	vec3 specular = pow(max(dot(R, V), 0.0), specular_power) * specular_albedo;

	color = vec4(ambient + diffuse + specular, 1.0);

	vec3 col = mix(color.rgb, text.rgb, text.a);

	color *= vec4(col, color.rgb * text.rgb);
}