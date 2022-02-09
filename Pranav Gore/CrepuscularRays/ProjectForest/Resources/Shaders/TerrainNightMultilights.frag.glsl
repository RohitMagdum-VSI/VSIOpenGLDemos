#version 450 core

in VS_OUT
{
	vec3 vPos;
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

struct Material
{
	sampler2D DiffuseMap;
	sampler2D SpecularMap;
	float MaterialShininess;
};

uniform sampler2D u_texture0_bg_sampler; 
uniform sampler2D u_texture1_r_sampler;  
uniform sampler2D u_texture2_g_sampler;  
uniform sampler2D u_texture3_b_sampler;  
uniform sampler2D u_texture4_blendmap_sampler;

uniform int torch;
uniform int num_lights;
uniform Material TerrainTexture;
uniform Light PointLight[2];
uniform Spotlight Torch;
uniform vec3 u_view_position;

// variables
vec3 fog_color = vec3(0.7, 0.8, 0.9);

vec4 fog(vec4 color) 
{ 
	float distance = length(u_view_position - fs_in.FragPos);
//	vec3 sunDir = vec3(0.0, 100.0, -100.0);  
	vec3 rayDir = u_view_position;  
//	vec3 rayOri = v3CameraPos;  
	float de = 0.025 * smoothstep(0.0, 6.0, 20.0 - fs_in.vPos.y); 
	float di = 0.035 * smoothstep(0.0, 40.0, 25.0 - fs_in.vPos.y); 
	float extinction = exp(-distance * de); 
	float inscattering = exp(-distance * di); 
//	float sunAmount = max( dot( rayDir, sunDir ), 0.0 ); 
//	vec3 fogColor  = mix( fog_color, vec3(1.0, 0.9, 0.0), pow(sunAmount, 8.0)); 
	float fogAmount = exp(-rayDir.y*di) * (1.0- inscattering); 
	return((color * extinction) + vec4((fog_color * fogAmount), 1.0));
	
}

//function declarations
vec3 CalculatePointLight(Light PointLight, vec3 TextureColor, vec3 normal, vec3 fragpos, vec3 viewdir);
vec3 CalculateSpotLight(Spotlight Torch, vec3 TextureColor, vec3 normal, vec3 fragpos, vec3 viewdir);
vec4 CalculateBlendMapColor();

void main(void)
{
	vec3 phong_ads;
	vec3 normalized_normal = normalize(fs_in.Normal);
	vec3 view_direction = normalize(u_view_position - fs_in.FragPos);
	vec3 FinalTerrainColor = vec3(CalculateBlendMapColor());

	for (int i = 0; i < num_lights; i++)
	{
		phong_ads += CalculatePointLight(PointLight[i], FinalTerrainColor, normalized_normal, fs_in.FragPos, view_direction);
	}
	
	if (torch == 1)
	{
		phong_ads += CalculateSpotLight(Torch, FinalTerrainColor, normalized_normal, fs_in.FragPos, view_direction);
	}
	
		
	//Gamma correction
	//phong_ads = pow(phong_ads, vec3(1.0 / 2.2)); 
	FragColor = vec4(phong_ads, 1.0);
	//FragColor = fog(vec4(phong_ads, 1.0));
}


//function to calculate directional light
vec3 CalculatePointLight(Light PointLight, vec3 TextureColor, vec3 normalized_normal, vec3 fragpos, vec3 view_direction)
{
	// Ambient term
	vec3 ambient = PointLight.ambient * TextureColor;

	// Diffuse term
	vec3 light_direction = normalize(PointLight.position - fs_in.FragPos);
	float diffuse_multiplier = max(dot(normalized_normal, light_direction), 0.0);
	vec3 diffuse = PointLight.diffuse * diffuse_multiplier * TextureColor;

	// Specular Term
	vec3 half_vector = normalize(light_direction + view_direction);
	float specular_multiplier = pow(max(dot(normalized_normal, half_vector), 0.0), 32.0f);
	vec3 specular = PointLight.specular * specular_multiplier * TextureColor;
	
	// Attenuation
	float distance = length(PointLight.position - fs_in.FragPos);
	float attenuation = 1.0 / (PointLight.constant_attenuation + PointLight.linear_attenuation * distance + PointLight.quadratic_attenuation * (distance * distance));

	ambient *= attenuation;
	diffuse *= attenuation;
	specular *= attenuation;

	//return(ambient + diffuse + specular);
	return(ambient + diffuse);
}

vec3 CalculateSpotLight(Spotlight Torch, vec3 TextureColor, vec3 normalized_normal, vec3 fragpos, vec3 view_direction)
{
	// Ambient term
	vec3 ambient = Torch.ambient * TextureColor;

	// Diffuse term
	vec3 light_direction = normalize(Torch.position - fs_in.FragPos);
	float diffuse_multiplier = max(dot(normalized_normal, light_direction), 0.0);
	vec3 diffuse = Torch.diffuse * diffuse_multiplier * TextureColor;

	// Specular Term
	vec3 half_vector = normalize(light_direction + view_direction);
	float specular_multiplier = pow(max(dot(normalized_normal, half_vector), 0.0), 32.0f);
	vec3 specular = Torch.specular * specular_multiplier * TextureColor;

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
