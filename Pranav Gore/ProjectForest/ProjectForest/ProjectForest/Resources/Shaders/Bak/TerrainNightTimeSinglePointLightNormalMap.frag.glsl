#version 450 core

in VS_OUT
{
	vec2 TexCoords0;
	vec3 FragPos;
	vec3 TangentLightPos;
	vec3 TangentViewPos;
	vec3 TangentFragPos;
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
	sampler2D NormalMap;
	float MaterialShininess;
};

uniform Material TerrainTexture;
uniform Light PointLight;


//function declarations
vec3 CalculatePointLight(Light PointLight, Material TerrainTexture, vec3 normal, vec3 fragpos, vec3 viewdir);

void main(void)
{
	vec3 phong_ads;

	// obtain normal from normal map
	vec3 normal = texture(TerrainTexture.NormalMap, fs_in.TexCoords0).rgb;
	//normal = normalize(normal);
	normal = normalize(normal * 2.0 - 1.0);

	vec3 view_direction = normalize(fs_in.TangentViewPos - fs_in.TangentFragPos);

	phong_ads += CalculatePointLight(PointLight, TerrainTexture, normal, fs_in.FragPos, view_direction);
		
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
	vec3 light_direction = normalize(fs_in.TangentLightPos - fs_in.TangentFragPos);
	float diffuse_multiplier = max(dot(normalized_normal, light_direction), 0.0);
	vec3 diffuse = PointLight.diffuse * diffuse_multiplier * vec3(texture(TerrainTexture.DiffuseMap, fs_in.TexCoords0));

	// Specular Term
	vec3 half_vector = normalize(light_direction + view_direction);
	float specular_multiplier = pow(max(dot(normalized_normal, half_vector), 0.0), TerrainTexture.MaterialShininess);
	vec3 specular = PointLight.specular * specular_multiplier * vec3(texture(TerrainTexture.DiffuseMap, fs_in.TexCoords0));
	
	// Attenuation
	/*float distance = length(fs_in.TangentViewPos - fs_in.TangentFragPos);
	float attenuation = 1.0 / (PointLight.constant_attenuation + PointLight.linear_attenuation * distance + PointLight.quadratic_attenuation * (distance * distance));

	ambient *= attenuation;
	diffuse *= attenuation;
	specular *= attenuation;*/

	//return(ambient + diffuse + specular);
	return(ambient + diffuse);
}
