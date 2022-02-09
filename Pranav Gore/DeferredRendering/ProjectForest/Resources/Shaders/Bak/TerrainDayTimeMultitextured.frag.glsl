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
	vec3 direction;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
};

struct Material
{
	sampler2D DiffuseMap;
	sampler2D SpecularMap;
	float MaterialShininess;
};

uniform Material TerrainTexture;
uniform Light SunLight;
uniform vec3 u_view_position;

//function declarations
vec3 CalculateSunLight(Light DirectionalLight, Material TerrainTexture, vec3 normal, vec3 fragpos, vec3 viewdir);

void main(void)
{
	vec3 phong_ads;
	vec3 normalized_normal = normalize(fs_in.Normal);
	vec3 view_direction = normalize(u_view_position - fs_in.FragPos);

	phong_ads += CalculateSunLight(SunLight, TerrainTexture, normalized_normal, fs_in.FragPos, view_direction);
		
	//Gamma correction
	phong_ads = pow(phong_ads, vec3(1.0 / 2.2));
	FragColor = vec4(phong_ads, 1.0);
}


//function to calculate directional light
vec3 CalculateSunLight(Light SunLight, Material TerrainTexture, vec3 normalized_normal, vec3 fragpos, vec3 view_direction)
{
	// Ambient Term
	vec3 ambient = SunLight.ambient * vec3(texture(TerrainTexture.DiffuseMap, fs_in.TexCoords0));

	// Diffuse Term
	vec3 light_direction = normalize(-SunLight.direction);
	float diffuse_multiplier = max(dot(normalized_normal, light_direction), 0.0);
	vec3 diffuse = SunLight.diffuse * diffuse_multiplier * vec3(texture(TerrainTexture.DiffuseMap, fs_in.TexCoords0));

	// Specular Term
	vec3 half_vector = normalize(light_direction + view_direction);
	float specular_multiplier = pow(max(dot(normalized_normal, half_vector), 0.0), TerrainTexture.MaterialShininess);
	vec3 specular = SunLight.specular * specular_multiplier * vec3(texture(TerrainTexture.DiffuseMap, fs_in.TexCoords0)) ;
	
	/*return(ambient + diffuse + specular);*/
	return(ambient + diffuse);
}
