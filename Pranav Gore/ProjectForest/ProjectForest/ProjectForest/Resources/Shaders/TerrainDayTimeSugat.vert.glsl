#version 450 core

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec4 vColor;
layout (location = 2) in vec3 vNormal;
layout (location = 3) in vec2 vTexCoord0;	

out VS_OUT
{
	vec2 TexCoords0;
	vec3 FragPos;
	vec3 Normal;
	vec4 OutColor;
	//vec3 vPos;
	float visibility;
} vs_out;

uniform mat4 u_model_matrix;
uniform mat4 u_view_matrix;
uniform mat4 u_projection_matrix;

//for water
uniform vec4 clipPlane;

const float density = 0.007;
const float gradient = 1.5;

void main(void)
{
	vs_out.FragPos = vec3(u_model_matrix * vec4(vPosition, 1.0));
	
	gl_ClipDistance[0] = dot(u_model_matrix * vec4(vPosition, 1.0), clipPlane);


	vs_out.Normal = mat3(transpose(inverse(u_model_matrix))) * vNormal;
	vs_out.TexCoords0 = vTexCoord0;
	vs_out.OutColor = vColor;
	//vs_out.vPos = vPosition;
	
	vec4 positionRelativeToCamera = u_view_matrix * vec4(vs_out.FragPos,1.0);
	float distance = length(positionRelativeToCamera.xyz);
	vs_out.visibility = exp(-pow(distance * density, gradient));
	vs_out.visibility = clamp(vs_out.visibility, 0.0, 1.0);

	gl_Position = u_projection_matrix * u_view_matrix * vec4(vs_out.FragPos, 1.0);
}
