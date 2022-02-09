// GPU Gems 3 - Chapter 13
// Volumetric Light Scattering as a Post-Process

#version 450 core

in vec2 TexCoords;
out vec4 FragColor;

uniform float Exposure;
uniform float Decay;
uniform float Density;
uniform float Weight;

uniform vec2 LightPositionOnScreen;
uniform sampler2D Texture;

const int NUM_SAMPLES = 100;

void main()
{
	// Calculate vector from fragment to light source in screen space.
	vec2 DeltaTexCoord = vec2(TexCoords - LightPositionOnScreen);

	// Create the temporary texture coordinate.
	vec2 TempTexCoord = TexCoords;

	// Divide by number of samples and scale by control factor.
	DeltaTexCoord *= 1.0 / float(NUM_SAMPLES) * Density;

	// Store initial sample.
	vec4 OutColor = vec4(0.0);

	// Set up illumination decay factor.
	float IlluminationDecay = 1.0;

	// Evaluate summation from Equation 3 NUM_SAMPLES iterations.
	for (int i = 0; i < NUM_SAMPLES; i++)
	{
		// Step sample location along ray.
		TempTexCoord -= DeltaTexCoord;

		// Retrieve sample at new location.
		vec4 Sample = texture2D(Texture, TempTexCoord);

		// Apply sample attenuation scale/decay factors.
		Sample *= IlluminationDecay * Weight;

		// Accumulate combined color.
		OutColor += Sample;

		// Update exponential decay factor.
		IlluminationDecay *= Decay;
	}

	// Output final color with a further scale control factor
	OutColor *= Exposure;
	FragColor = OutColor;
}
