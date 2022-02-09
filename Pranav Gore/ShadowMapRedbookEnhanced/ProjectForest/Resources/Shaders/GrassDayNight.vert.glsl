#version 450 core

layout (location = 0) in vec4 vPosition;
layout (location = 1) in vec4 vColor;
layout (location = 2) in vec3 vNormal;
layout (location = 5) in vec4 Brightness;
layout (location = 6) in mat4 InstanceMatrix;
		
uniform mat4 u_model_matrix;
uniform mat4 u_view_matrix;
uniform mat4 u_projection_matrix;

uniform float GrassAreaRadius;
uniform float WaveTime;
uniform float WaveStrength;

out VS_OUT
{
	vec3 FragPos;
	vec3 Normal;
	vec4 OutColor;
} vs_out;

void main()
{
	vs_out.FragPos = vec3(InstanceMatrix * vPosition);
	vs_out.Normal = mat3(transpose(inverse(InstanceMatrix))) * vNormal;
	//vs_out.OutColor = vColor;

	mat4 modelview = u_view_matrix * InstanceMatrix;

	//extract our scale vectors so we can keep them (we don't need Z because we don't zero out the 2nd modelview column, since this is a cylindrical billboard, not a spherical one)
	//vec3 scaleX = modelview[0].xyz;
	//vec3 scaleY = modelview[1].xyz;
		
	// zero out first column for a cylindrical billboard
	//modelview[0][0] = 0;
	//modelview[0][1] = 0;
	//modelview[0][2] = 0;

	//zero out the second column for a cylindrical billboard
	//modelview[1][0] = 0;
	//modelview[1][1] = 0;
	//modelview[1][2] = 0;
	
	//compute a wave amount that is dependant on the camera position and the height of the current vertex
	
	float waveAmount = (-0.5 + sin(WaveTime + (InstanceMatrix[3][0] + InstanceMatrix[3][2]))) * WaveStrength * vPosition.y;
	vec4 vertex = vec4(vPosition.x + waveAmount, vPosition.y, vPosition.z + waveAmount, 1.0);

	//now compute the position of this vertex based on the calculated wave and the cylindrical billboard
	vec4 pos = modelview * vertex;

	//fade out based on the distance from the camera
	//float opacity = 1.0 - (-pos.z / GrassAreaRadius);
		
	//assign colour based on brightness and distance
	//vs_out.OutColor = vColor * Brightness;
		
	vs_out.OutColor = vColor;
	//vs_out.OutColor.a = opacity;
		
	gl_Position = u_projection_matrix * pos;
}
