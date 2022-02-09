#version 450 core

in VS_OUT
{
	vec2 TexCoords0;
	vec3 FragPos;
	vec3 Normal;
} fs_in;

out vec4 FragColor;

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

struct Material
{
	sampler2D DiffuseMap;
	sampler2D SpecularMap;
	float MaterialShininess;
};

uniform Material TerrainTexture;
uniform Light PointLight;
uniform vec3 u_view_position;

//function declarations
vec3 CalculatePointLight(Light PointLight, Material TerrainTexture, vec3 normal, vec3 fragpos, vec3 viewdir);

void main(void)
{
	vec3 phong_ads;
	vec3 normalized_normal = normalize(fs_in.Normal);
	vec3 view_direction = normalize(u_view_position - fs_in.FragPos);

	phong_ads += CalculatePointLight(PointLight, TerrainTexture, normalized_normal, fs_in.FragPos, view_direction);
		
	//Gamma correction
	phong_ads = pow(phong_ads, vec3(1.0 / 2.2));
	FragColor = vec4(phong_ads, 1.0);
}


//function to calculate directional light
vec3 CalculatePointLight(Light PointLight, Material TerrainTexture, vec3 normalized_normal, vec3 fragpos, vec3 view_direction)
{
	// Ambient term
	vec3 ambient = PointLight.ambient * vec3(texture(TerrainTexture.DiffuseMap, fs_in.TexCoords0));

	// Diffuse term
	vec3 light_direction = normalize(PointLight.position - fs_in.FragPos);
	float diffuse_multiplier = max(dot(normalized_normal, light_direction), 0.0);
	vec3 diffuse = PointLight.diffuse * diffuse_multiplier * vec3(texture(TerrainTexture.DiffuseMap, fs_in.TexCoords0));

	// Specular Term
	vec3 half_vector = normalize(light_direction + view_direction);
	float specular_multiplier = pow(max(dot(normalized_normal, half_vector), 0.0), TerrainTexture.MaterialShininess);
	vec3 specular = PointLight.specular * specular_multiplier * vec3(texture(TerrainTexture.DiffuseMap, fs_in.TexCoords0));
	
	// Attenuation
	float distance = length(PointLight.position - fs_in.FragPos);
	float attenuation = 1.0 / (PointLight.constant_attenuation + PointLight.linear_attenuation * distance + PointLight.quadratic_attenuation * (distance * distance));

	ambient *= attenuation;
	diffuse *= attenuation;
	specular *= attenuation;

	//return(ambient + diffuse + specular);
	return(ambient + diffuse);
}
