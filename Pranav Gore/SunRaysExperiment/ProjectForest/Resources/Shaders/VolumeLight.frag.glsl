#version 450 core

in vec2 TexCoord;
out vec4 FragColor;

uniform vec2 u_LightPosition;	// Position of the light on screen
uniform sampler2D u_OcclusionMap; // Occlusion map texture
uniform sampler2D u_SceneRender; // Scene render texture

const int NUM_SAMPLES = 150; // No. of steps between current fragment and light position
const float EXPOSURE = 0.0225; // Brightness of the light
const float DECAY = 1.0; // Brightness decay factor over distance
const float DENSITY = 1.0; // Illumination decay factor after each sample
const float WEIGHT = 0.75; // Weight of each sample

void main(void)
{
	float IlluminationDecay = 1.0;

	// Find howmuch distance we will travel per sample
	vec2 DeltaTexCoord = TexCoord - u_LightPosition;
	DeltaTexCoord *= 1.0 / float(NUM_SAMPLES) * DENSITY;

	// start at the current location of the fragment
	vec2 CurrentTexCoord = TexCoord;

	// by default no samples have been read
	vec4 Color = vec4(0.0);

	// Read each sample between the fragment and the light position to find out the colouring

	for (int i = 0; i < NUM_SAMPLES; i++)
	{
		CurrentTexCoord -= DeltaTexCoord;
		vec4 OcclusionSample = texture2D(u_OcclusionMap, CurrentTexCoord);
		OcclusionSample *= IlluminationDecay * WEIGHT;
		Color += OcclusionSample;
		IlluminationDecay *= DECAY;
	}

	// Adjust the brightness
	Color *= EXPOSURE;

	// Read the colour from the scene render
	Color += texture2D(u_SceneRender, TexCoord);

	// Clamp the colour and send it off
	FragColor = clamp(Color, vec4(0.0), vec4(1.0));
}