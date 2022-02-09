#version 450 core

out vec4 FragColor;	

in VS_OUT
{
	vec3 FragPos;
	vec3 Normal;
	vec4 OutColor;
} fs_in;

//struct for light
//struct for light
struct Light
{
	vec3 position;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;

	float constant_attenuation;
	float linear_attenuation;
	float quadratic_attenuation;
};

struct Spotlight
{
	vec3 position;
	vec3 direction;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;

	float constant_attenuation;
	float linear_attenuation;
	float quadratic_attenuation;

	float inner_cutoff;
	float outer_cutoff;
};

uniform int torch;
uniform int num_lights;
uniform Light PointLight[2];
uniform Spotlight Torch;
uniform vec3 u_view_position;

vec3 CalculatePointLight(Light PointLight, vec4 GrassColor, vec3 normal, vec3 fragpos, vec3 viewdir);
vec3 CalculateSpotLight(Spotlight Torch, vec4 GrassColor, vec3 normal, vec3 fragpos, vec3 viewdir);

void main(void)
{
	vec3 phong_ads;
	vec3 normalized_normal = normalize(fs_in.Normal);
	vec3 view_direction = normalize(u_view_position - fs_in.FragPos);

	for (int i = 0; i < num_lights; i++)
	{
		phong_ads += CalculatePointLight(PointLight[i], fs_in.OutColor, normalized_normal, fs_in.FragPos, view_direction);
	}
	
	if (torch == 1)
	{
		phong_ads += CalculateSpotLight(Torch, fs_in.OutColor, normalized_normal, fs_in.FragPos, view_direction);
	}
		
	FragColor = vec4(phong_ads, 1.0);
	//FragColor = out_color;
}



//function to calculate directional light
vec3 CalculatePointLight(Light PointLight, vec4 GrassColor, vec3 normalized_normal, vec3 fragpos, vec3 view_direction)
{
	// Ambient term
	vec3 ambient = PointLight.ambient * GrassColor.rgb;

	// Diffuse term
	vec3 light_direction = normalize(PointLight.position - fs_in.FragPos);
	float diffuse_multiplier = max(dot(normalized_normal, light_direction), 0.0);
	vec3 diffuse = PointLight.diffuse * diffuse_multiplier * GrassColor.rgb;

	// Specular Term
	vec3 half_vector = normalize(light_direction + view_direction);
	float specular_multiplier = pow(max(dot(normalized_normal, half_vector), 0.0), 32.0f);
	vec3 specular = PointLight.specular * specular_multiplier * GrassColor.rgb;
	
	// Attenuation
	float distance = length(PointLight.position - fs_in.FragPos);
	float attenuation = 1.0 / (PointLight.constant_attenuation + PointLight.linear_attenuation * distance + PointLight.quadratic_attenuation * (distance * distance));

	ambient *= attenuation;
	diffuse *= attenuation;
	specular *= attenuation;

	return(ambient + diffuse + specular);
	//return(ambient + diffuse);
}

vec3 CalculateSpotLight(Spotlight Torch, vec4 GrassColor, vec3 normalized_normal, vec3 fragpos, vec3 view_direction)
{
	// Ambient term
	vec3 ambient = Torch.ambient * GrassColor.rgb;

	// Diffuse term
	vec3 light_direction = normalize(Torch.position - fs_in.FragPos);
	float diffuse_multiplier = max(dot(normalized_normal, light_direction), 0.0);
	vec3 diffuse = Torch.diffuse * diffuse_multiplier * GrassColor.rgb;

	// Specular Term
	vec3 half_vector = normalize(light_direction + view_direction);
	float specular_multiplier = pow(max(dot(normalized_normal, half_vector), 0.0), 32.0f);
	vec3 specular = Torch.specular * specular_multiplier * GrassColor.rgb;

	// soft edges
	float theta = dot(light_direction, normalize(-Torch.direction));
	float epsilon = (Torch.inner_cutoff - Torch.outer_cutoff);
	float intensity = clamp((theta - Torch.outer_cutoff) / epsilon, 0.0, 1.0);

	diffuse *= intensity;
	specular *= intensity;

	// attenuation
	float distance = length(Torch.position - fs_in.FragPos);
	float attenuation = 1.0 / (Torch.constant_attenuation + Torch.linear_attenuation * distance + Torch.quadratic_attenuation * (distance * distance));

	ambient *= attenuation;
	diffuse *= attenuation;
	specular *= attenuation;

	return(ambient + diffuse + specular);
}