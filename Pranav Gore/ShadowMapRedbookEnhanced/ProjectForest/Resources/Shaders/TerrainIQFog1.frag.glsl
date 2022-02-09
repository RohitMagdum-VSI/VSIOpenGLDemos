#version 450 core

in VS_OUT
{
	vec2 TexCoords0;
	vec3 FragPos;
	vec3 FragPosWorldCoord;
	vec3 FragPosEyeCoord;
	vec3 Normal;
	vec4 ShadowCoord;
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

uniform sampler2DShadow DepthTexture;
uniform vec3 uLightPosition;
uniform float uXPixelOffset;
uniform float uYPixelOffset;

uniform vec4 fogColor = vec4(0.7, 0.8, 0.9, 0.0);

//function declarations
vec3 CalculateSunLight(Light DirectionalLight, vec3 TextureColor, vec3 normal, vec3 fragpos, vec3 viewdir);
vec4 CalculateBlendMapColor();
vec3 ApplyFog(vec3 Color, float distance);
float Lookup(vec2 offset);
float CalculatePCF(void);

void main(void)
{
	vec3 phong_ads;
	vec3 normalized_normal = normalize(fs_in.Normal);
	vec3 view_direction = normalize(u_view_position - fs_in.FragPos);

	vec3 FinalTerrainColor = vec3(CalculateBlendMapColor());

	phong_ads += CalculateSunLight(SunLight, FinalTerrainColor, normalized_normal, fs_in.FragPos, view_direction);
	
	FragColor = vec4(phong_ads, 1.0);
	// here it must be fs_in.FragPosWorldCoord or fs_in.FragPos in distance() function - refer http://www.iquilezles.org/www/articles/fog/fog.htm
	//FragColor = vec4(ApplyFog(phong_ads, distance(fs_in.FragPosWorldCoord, u_view_position)), 1.0);
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

	// Calculate the contributon of Shadow
	//float Shadow = textureProj(DepthTexture, fs_in.ShadowCoord);
	
	float Shadow = CalculatePCF();

	/*return(ambient + diffuse + specular);*/
	return(ambient + Shadow * diffuse);
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

vec3 ApplyFog(vec3 Color, float distance)
{
	float b = 0.001;

	float fogAmount = 1.0 - exp(-distance * b);
	vec3 fogColor = vec3(0.5, 0.6, 0.7);
	return(mix(Color, fogColor, fogAmount));
}

float CalculatePCF(void)
{
	float Shadow;
	// avoid counter shadow
	if (fs_in.ShadowCoord.w > 1.0)
	{
		// 8x8 kernel PCF
		/*float x, y;
		for (y = -3.5; y <= 3.5; y += 1.0)
		{
			for (x = -3.5; x <= 3.5; x += 1.0)
			{
				Shadow += Lookup(vec2(x, y));
			}
		}
		Shadow /= 64.0;*/

		// 8x8 PCF Wide kernel (step is 10 instead of 1)
		/*float x, y;
		for (y = -30.5; y <= 30.5; y += 10.0)
		{
			for (x = -30.5; x <= 30.5; x += 10.0)
			{
				Shadow += Lookup(vec2(x, y));
			}
		}
		Shadow /= 64.0;*/

		// 4x4 kernel PCF
		float x, y;
		for (y = -1.5; y <= 1.5; y += 1.0)
		{
			for (x = -1.5; x <= 1.5; x += 1.0)
			{
				Shadow += Lookup(vec2(x, y));
			}
		}
		Shadow /= 16.0;

		// 4x4 PCF Wide Kernel (step is 10 instead of 1)
		/*float x, y;
		for (y = -10.5; y <= 10.5; y += 10.0)
		{
			for (x = -10.5; x <= 10.5; x += 10.0)
			{
				Shadow += Lookup(vec2(x, y));
			}
		}
		Shadow /= 16.0;*/

		// 4x4 PCF Dithered
		// Use Modulo to vary the sample pattern
		// vec2 o = mod(floor(gl_FragCoord.xy), 2.0);

		// Shadow += Lookup(vec2(-1.5, 1.5) + o);
		// Shadow += Lookup(vec2(0.5, 1.5) + o);
		// Shadow += Lookup(vec2(-1.5, -0.5) + o);
		// Shadow += Lookup(vec2(0.5, -0.5) + o);
		// Shadow *= 0.25;
	}

	return Shadow;
}

float Lookup(vec2 offset)
{
	return textureProj(DepthTexture, fs_in.ShadowCoord + vec4(offset.x * uXPixelOffset * fs_in.ShadowCoord.w, offset.y * uYPixelOffset * fs_in.ShadowCoord.w, 0.05, 0.0));
}