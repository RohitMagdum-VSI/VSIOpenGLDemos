#version 450 core

in vec2 TexCoords;

out vec4 FragColor;

uniform sampler2D PositionTexture;
uniform sampler2D NormalTexture;
uniform sampler2D AlbedoSpecularTexture;

vec3 CalculateSunLight(Light DirectionalLight, vec3 TextureColor, vec3 normal, vec3 viewdir);

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

void main()
{
	// retirve data from geometry buffer
	vec3 FragPos = texture(PositionTexture, TexCoords).rgb;
	vec3 Normal = texture(NormalTexture, TexCoords).rgb;
	vec3 Diffuse = texture(AlbedoSpecularTexture, TexCoords).rgb;
	float Specular = texture(AlbedoSpecularTexture, TexCoords).a;

	// Calculate the light as usual
	vec3 normalized_normal = normalize(Normal);
	vec3 view_direction = normalize(u_view_position - FragPos);

	phong_ads += CalculateSunLight(SunLight, Diffuse, normalized_normal, fs_in.FragPos, view_direction);
	
	FragColor = vec4(phong_ads, 1.0);
}

vec3 CalculateSunLight(Light DirectionalLight, vec3 TextureColor, vec3 normal, vec3 viewdir)
{
	// Ambient Term
	vec3 ambient = SunLight.ambient * FinalTerrainColor;

	// Diffuse Term
	vec3 light_direction = normalize(-SunLight.direction);
	float diffuse_multiplier = max(dot(normalized_normal, light_direction), 0.0);
	vec3 diffuse = SunLight.diffuse * diffuse_multiplier * FinalTerrainColor;

	// Specular Term
	vec3 half_vector = normalize(light_direction + view_direction);
	float specular_multiplier = pow(max(dot(normalized_normal, half_vector), 0.0), 128.0);
	vec3 specular = SunLight.specular * specular_multiplier * FinalTerrainColor;
	
	return(ambient + diffuse + specular);
}