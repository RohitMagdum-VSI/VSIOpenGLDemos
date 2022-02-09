#version 450 core

in VS_OUT
{
	vec2 TexCoords0;
	vec3 FragPos;
	vec3 FragPosWorldCoord;
	vec3 FragPosEyeCoord;
	vec3 TangentLightDir;
	vec3 TangentViewPos;
	vec3 TangentFragPos;
	vec4 ShadowCoord;
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
	sampler2D NormalTexture;
	float Shininess;	
};

uniform Light SunLight;
uniform Material material;
uniform vec3 u_view_position;

uniform vec3 uLightPosition;
uniform sampler2DShadow DepthTexture;
uniform float uXPixelOffset;
uniform float uYPixelOffset;

uniform sampler2D u_texture0;

//function declarations
vec3 CalculateSunLight(Light DirectionalLight, Material mat, vec3 normal, vec3 fragpos, vec3 viewdir);
vec3 ApplyFog(vec3 Color, float distance);
float Lookup(vec2 offset);
float CalculatePCF(void);

void main(void)
{
	vec3 phong_ads;

	// get the normal vector from NormalMap texture
	vec3 normal = texture(material.NormalTexture, fs_in.TexCoords0).rgb;

	// transform the normal vector to range [-1, 1]
	vec3 normalized_normal = normalize(normal * 2.0 - 1.0);

	vec3 view_direction = normalize(fs_in.TangentViewPos - fs_in.TangentFragPos);

	//vec3 TextureColor = vec3(texture(u_texture0, fs_in.TexCoords0));

	phong_ads += CalculateSunLight(SunLight, material, normalized_normal, fs_in.FragPos, view_direction);
	
	FragColor = vec4(phong_ads, 1.0);
	//FragColor = vec4(ApplyFog(phong_ads, distance(fs_in.FragPosWorldCoord, u_view_position)), 1.0);
}


//function to calculate directional light
vec3 CalculateSunLight(Light SunLight, Material material, vec3 normalized_normal, vec3 fragpos, vec3 view_direction)
{
	// Ambient Term
	vec3 ambient = SunLight.ambient * texture(material.DiffuseTexture, fs_in.TexCoords0).rgb;

	// Diffuse Term
	vec3 light_direction = normalize(-fs_in.TangentLightDir);
	float diffuse_multiplier = max(dot(normalized_normal, light_direction), 0.0);
	vec3 diffuse = SunLight.diffuse * diffuse_multiplier * texture(material.DiffuseTexture, fs_in.TexCoords0).rgb;

	// Specular Term
	vec3 half_vector = normalize(light_direction + view_direction);
	float specular_multiplier = pow(max(dot(normalized_normal, half_vector), 0.0), material.Shininess);
	vec3 specular = SunLight.specular * specular_multiplier * texture(material.SpecularTexture, fs_in.TexCoords0).rgb;
	
	// Calculate the contributon of Shadow
	//float Shadow = textureProj(DepthTexture, fs_in.ShadowCoord);

	float Shadow = CalculatePCF();

	return(ambient + Shadow * (diffuse + specular));
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
	/*	float x, y;
		for (y = -3.5; y <= 3.5; y += 1.0)
		{
			for (x = -3.5; x <= 3.5; x += 1.0)
			{
				Shadow += Lookup(vec2(x, y));
			}
		}
		Shadow /= 64.0;
*/
		// 8x8 PCF Wide kernel (step is 10 instead of 1)
		/*float x, y;
		for (y = -30.5; y <= 30.5; y += 10.0)
		{
			for (x = -30.5; x <= 30.5; x += 10.0)
			{
				Shadow += Lookup(vec2(x, y));
			}
		}
		Shadow /= 64.0;
*/
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
	return textureProj(DepthTexture, fs_in.ShadowCoord + vec4(offset.x * uXPixelOffset * fs_in.ShadowCoord.w, offset.y * uYPixelOffset * fs_in.ShadowCoord.w, 0.0, 0.0));
}