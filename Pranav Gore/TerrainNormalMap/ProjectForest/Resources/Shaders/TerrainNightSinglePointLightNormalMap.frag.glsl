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

uniform sampler2D u_texture0_bg_sampler; 
uniform sampler2D u_texture1_r_sampler;  
uniform sampler2D u_texture2_g_sampler;  
uniform sampler2D u_texture3_b_sampler;  
uniform sampler2D u_texture4_blendmap_sampler;
uniform sampler2D u_texture5_normalmap_sampler;

uniform Material TerrainTexture;
uniform Light PointLight;

//function declarations
vec3 CalculatePointLight(Light PointLight, Material TerrainTexture, vec3 normal, vec3 fragpos, vec3 viewdir);
vec4 CalculateBlendMapColor();

void main(void)
{
	vec3 phong_ads;

	// obtain normal from normal map
	//vec3 normal = texture(TerrainTexture.NormalMap, fs_in.TexCoords0).rgb;
	vec3 normal = vec3(texture(u_texture5_normalmap_sampler, fs_in.TexCoords0));
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
	vec3 ambient = PointLight.ambient * vec3(texture(u_texture0_bg_sampler, fs_in.TexCoords0));

	// Diffuse term
	vec3 light_direction = normalize(fs_in.TangentLightPos - fs_in.TangentFragPos);
	float diffuse_multiplier = max(dot(light_direction, normalized_normal), 0.0);
	vec3 diffuse = PointLight.diffuse * diffuse_multiplier * vec3(texture(u_texture0_bg_sampler, fs_in.TexCoords0));

	// Specular Term
	vec3 half_vector = normalize(light_direction + view_direction);
	float specular_multiplier = pow(max(dot(normalized_normal, half_vector), 0.0), 32.0);
	vec3 specular = PointLight.specular * specular_multiplier * vec3(texture(u_texture0_bg_sampler, fs_in.TexCoords0));
	
	// Attenuation
	float distance = length(fs_in.TangentViewPos - fs_in.TangentFragPos);
	float attenuation = 1.0 / (PointLight.constant_attenuation + PointLight.linear_attenuation * distance + PointLight.quadratic_attenuation * (distance * distance));

	ambient *= attenuation;
	diffuse *= attenuation;
	specular *= attenuation;

	return(ambient + diffuse + specular);
	//return(ambient + diffuse);
}

vec4 CalculateBlendMapColor()
{
	vec4 blendMapColor = texture(u_texture4_blendmap_sampler, fs_in.TexCoords0); 
	float backTextureAmount = 1 - (blendMapColor.r + blendMapColor.g + blendMapColor.b); 
	vec2 tiledCoords = fs_in.TexCoords0;

	vec4 backgroundTextureColor = texture(u_texture0_bg_sampler, tiledCoords)* backTextureAmount;
	vec4 rTextureColor = texture(u_texture1_r_sampler, tiledCoords) *blendMapColor.r;
	vec4 gTextureColor = texture(u_texture2_g_sampler, tiledCoords) *blendMapColor.g;
	vec4 bTextureColor = texture(u_texture3_b_sampler, tiledCoords) *blendMapColor.b;

	vec4 totalColor = backgroundTextureColor + rTextureColor + gTextureColor + bTextureColor;
	
	return(totalColor);
}

