#version 450 core

out vec4 FragColor;	

in VS_OUT
{
	vec3 FragPos;
	vec3 Normal;
	vec4 OutColor;
} fs_in;

//struct for light
struct Light
{
	vec3 direction;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
};

uniform Light SunLight;
uniform vec3 u_view_position;

vec3 CalculateSunLight(Light DirectionalLight, vec4 GrassColor, vec3 normal, vec3 fragpos, vec3 viewdir);

void main(void)
{
	vec3 phong_ads;
	vec3 normalized_normal = normalize(fs_in.Normal);
	vec3 view_direction = normalize(u_view_position - fs_in.FragPos);

	phong_ads += CalculateSunLight(SunLight, fs_in.OutColor, normalized_normal, fs_in.FragPos, view_direction);

	FragColor = vec4(phong_ads, 1.0);
	//FragColor = out_color;
}

//function to calculate directional light
vec3 CalculateSunLight(Light SunLight, vec4 GrassColor, vec3 normalized_normal, vec3 fragpos, vec3 view_direction)
{
	// Ambient Term
	vec3 ambient = SunLight.ambient * GrassColor.rgb;

	// Diffuse Term
	vec3 light_direction = normalize(-SunLight.direction);
	float diffuse_multiplier = max(dot(normalized_normal, light_direction), 0.0);
	vec3 diffuse = SunLight.diffuse * diffuse_multiplier * GrassColor.rgb;

	// Specular Term
	vec3 half_vector = normalize(light_direction + view_direction);
	float specular_multiplier = pow(max(dot(normalized_normal, half_vector), 0.0), 50.0);
	vec3 specular = SunLight.specular * specular_multiplier * GrassColor.rgb;
	
	return(ambient + diffuse + specular);
	//return(ambient + diffuse);
}
