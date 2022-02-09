#version 450 core

in VS_OUT
{
	vec2 TexCoords0;
	vec3 FragPos;
	vec3 FragPosWorldCoord;
	vec3 FragPosEyeCoord;
	vec3 Normal;
	vec4 OutColor;
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

uniform sampler2D u_texture0_bg_sampler; 
uniform sampler2D u_texture1_r_sampler;  
uniform sampler2D u_texture2_g_sampler;  
uniform sampler2D u_texture3_b_sampler;  
uniform sampler2D u_texture4_blendmap_sampler;

uniform vec4 fogColor = vec4(0.7, 0.8, 0.9, 0.0);

//function declarations
vec3 CalculateSunLight(Light DirectionalLight, vec3 TextureColor, vec3 normal, vec3 fragpos, vec3 viewdir);
vec4 CalculateBlendMapColor();

// pixel color
// camera to point distance
// camera to point vector
// sun light direction

vec3 ApplyFog(vec3 Color, float distance, vec3 rayDir, vec3 sunDir);

void main(void)
{
	vec3 phong_ads;
	vec3 normalized_normal = normalize(fs_in.Normal);
	vec3 view_direction = normalize(u_view_position - fs_in.FragPos);

	vec3 FinalTerrainColor = vec3(CalculateBlendMapColor());

	phong_ads += CalculateSunLight(SunLight, FinalTerrainColor, normalized_normal, fs_in.FragPos, view_direction);
	
	//FragColor = vec4(phong_ads, 1.0);
	// here it must be fs_in.FragPosWorldCoord or fs_in.FragPos in distance() function - refer http://www.iquilezles.org/www/articles/fog/fog.htm

	float dist = distance(fs_in.FragPos, u_view_position);
	vec3 raydir = normalize(fs_in.FragPos - u_view_position);

	FragColor = vec4(ApplyFog(phong_ads, dist, raydir, -SunLight.direction), 1.0);
}


//function to calculate directional light
vec3 CalculateSunLight(Light SunLight, vec3 FinalTerrainColor, vec3 normalized_normal, vec3 fragpos, vec3 view_direction)
{
	// Ambient Term
	vec3 ambient = SunLight.ambient * FinalTerrainColor;

	// Diffuse Term
	vec3 light_direction = normalize(-SunLight.direction);
	float diffuse_multiplier = max(dot(normalized_normal, light_direction), 0.0);
	vec3 diffuse = SunLight.diffuse * diffuse_multiplier * FinalTerrainColor;

	// Specular Term
	vec3 half_vector = normalize(light_direction + view_direction);
	float specular_multiplier = pow(max(dot(normalized_normal, half_vector), 0.0), TerrainTexture.MaterialShininess);
	vec3 specular = SunLight.specular * specular_multiplier * FinalTerrainColor;
	
	/*return(ambient + diffuse + specular);*/
	return(ambient + diffuse);
}

vec4 CalculateBlendMapColor()
{
	vec4 blendMapColor = texture(u_texture4_blendmap_sampler,fs_in.TexCoords0); 
	float backTextureAmount = 1 - (blendMapColor.r + blendMapColor.g + blendMapColor.b); 
	vec2 tiledCoords = fs_in.TexCoords0 * 80;

	vec4 backgroundTextureColor = texture(u_texture0_bg_sampler, tiledCoords)* backTextureAmount;
	vec4 rTextureColor = texture(u_texture1_r_sampler, tiledCoords) *blendMapColor.r;
	vec4 gTextureColor = texture(u_texture2_g_sampler, tiledCoords) *blendMapColor.g;
	vec4 bTextureColor = texture(u_texture3_b_sampler, tiledCoords) *blendMapColor.b;

	vec4 totalColor = backgroundTextureColor + rTextureColor + gTextureColor + bTextureColor;
	
	return(totalColor);
}

vec3 ApplyFog(vec3 Color, float distance, vec3 rayDir, vec3 sunDir)
{
	float b = 0.001;

	float fogAmount = 1.0 - exp(-distance * b);
	float sunAmount = max(dot(rayDir, sunDir), 0.0);
	// mix the bluish and yellowish color
	vec3 fogColor = mix(vec3(0.5, 0.6, 0.7), vec3(1.0, 0.9, 0.7), pow(sunAmount, 8.0));
	return(mix(Color, fogColor, fogAmount));
}
