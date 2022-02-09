#version 450 core

in VS_OUT
{
	vec2 TexCoords0;
	vec3 FragPos;
	vec3 FragPosWorldCoord;
	vec3 FragPosEyeCoord;
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
	sampler2D DiffuseTexture;
	sampler2D SpecularTexture;
	float Shininess;	
};

uniform Light SunLight;
uniform Material material;
uniform vec3 u_view_position;

uniform sampler2D u_texture0; 

uniform vec4 fogColor = vec4(0.7, 0.8, 0.9, 0.0);

//function declarations
vec3 CalculateSunLight(Light DirectionalLight, Material mat, vec3 normal, vec3 fragpos, vec3 viewdir);
vec3 ApplyFog(vec3 Color, float distance);

void main(void)
{
	vec3 phong_ads;
	vec3 normalized_normal = normalize(fs_in.Normal);
	vec3 view_direction = normalize(u_view_position - fs_in.FragPos);

	//vec3 TextureColor = vec3(texture(u_texture0, fs_in.TexCoords0));

	phong_ads += CalculateSunLight(SunLight, material, normalized_normal, fs_in.FragPos, view_direction);
	
	//FragColor = vec4(phong_ads, 1.0);
	// here it must be fs_in.FragPosWorldCoord or fs_in.FragPos in distance() function - refer http://www.iquilezles.org/www/articles/fog/fog.htm
	FragColor = vec4(ApplyFog(phong_ads, distance(fs_in.FragPosWorldCoord, u_view_position)), 1.0);
}


//function to calculate directional light
vec3 CalculateSunLight(Light SunLight, Material material, vec3 normalized_normal, vec3 fragpos, vec3 view_direction)
{
	// Ambient Term
	vec3 ambient = SunLight.ambient * texture(material.DiffuseTexture, fs_in.TexCoords0).rgb;

	// Diffuse Term
	vec3 light_direction = normalize(-SunLight.direction);
	float diffuse_multiplier = max(dot(normalized_normal, light_direction), 0.0);
	vec3 diffuse = SunLight.diffuse * diffuse_multiplier * texture(material.DiffuseTexture, fs_in.TexCoords0).rgb;

	// Specular Term
	vec3 half_vector = normalize(light_direction + view_direction);
	float specular_multiplier = pow(max(dot(normalized_normal, half_vector), 0.0), material.Shininess);
	vec3 specular = SunLight.specular * specular_multiplier * texture(material.SpecularTexture, fs_in.TexCoords0).rgb;
	
	return(ambient + diffuse + specular);
}

vec3 ApplyFog(vec3 Color, float distance)
{
	float b = 0.001;

	float fogAmount = 1.0 - exp(-distance * b);
	vec3 fogColor = vec3(0.5, 0.6, 0.7);
	return(mix(Color, fogColor, fogAmount));
}
