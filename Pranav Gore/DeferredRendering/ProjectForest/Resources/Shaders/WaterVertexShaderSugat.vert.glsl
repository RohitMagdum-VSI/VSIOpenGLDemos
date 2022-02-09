#version 450

layout (location = 0) in vec3 position;

out vec2 textureCoords;
out vec3 toCameraVector; 
out vec3 toLightVector[2]; 
out vec4 clipSpace; 

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;
uniform vec3 lightPosition[2]; 


void main(void) {

	vec4 worldPosition = modelMatrix * vec4(position, 1.0);	
	//worldPosition.y = 15.0; 
	vec4 transformedPos = projectionMatrix * viewMatrix * worldPosition;
 
	gl_Position = transformedPos; 	
	clipSpace = transformedPos; 	
		
	toCameraVector = vec3(inverse(viewMatrix) * vec4(0.0, 0.0, 0.0, 1.0)) - vec3(worldPosition); 	
	textureCoords = vec2(worldPosition.x, worldPosition.z) / 16;
	
	for (int i = 0; i < 2; i++) {
		toLightVector[i] = lightPosition[i] - vec3(worldPosition); 
	}
 
}