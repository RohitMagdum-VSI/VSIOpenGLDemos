#version 450 core

in VS_OUT
{
	vec2 TexCoords0;
	vec3 FragPos;
	vec3 Normal;
	vec4 OutColor;
	//vec3 vPos;
	float visibility;
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

vec3 fog_color = vec3(0.5, 0.6, 0.7);

//function declarations
vec3 CalculateSunLight(Light DirectionalLight, vec3 TextureColor, vec3 normal, vec3 fragpos, vec3 viewdir);
vec4 CalculateBlendMapColor();

//vec4 fog(vec4 color)
//{
//	float distance = length(u_view_position - fs_in.FragPos);
//	vec3 sunDir = SunLight.direction;
//	vec3 rayDir = u_view_position;
//	//	vec3 rayOri = v3CameraPos;  
//	float de = 0.025 * smoothstep(0.0, 6.0, 20.0 - fs_in.vPos.y);
//	float di = 0.035 * smoothstep(0.0, 40.0, 25.0 - fs_in.vPos.y);
//	float extinction = exp(-distance * de);
//	float inscattering = exp(-distance * di);
//	float sunAmount = max( dot( rayDir, sunDir ), 0.0 ); 
//	vec3 fogColor  = mix( fog_color, vec3(1.0, 0.9, 0.0), pow(sunAmount, 8.0)); 
//	float fogAmount = exp(-rayDir.y*di) * (1.0 - inscattering);
//	return((color * extinction) + vec4((fogColor * fogAmount), 1.0));
//
//}

void main(void)
{
	vec3 phong_ads;
	vec3 normalized_normal = normalize(fs_in.Normal);
	vec3 view_direction = normalize(u_view_position - fs_in.FragPos);

	vec3 FinalTerrainColor = vec3(CalculateBlendMapColor());

	phong_ads += CalculateSunLight(SunLight, FinalTerrainColor, normalized_normal, fs_in.FragPos, view_direction);
	
	//Gamma correction
	//phong_ads = pow(phong_ads, vec3(1.0 / 2.2));
	FragColor = vec4(phong_ads, 1.0);
	//FragColor = fog(vec4(phong_ads, 1.0));
	
	//FragColor = mix(vec4(fog_color, 1.0), vec4(phong_ads, 1.0), fs_in.visibility);
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
