#version 410 core
layout ( location = 0 ) in vec3 in_position;
layout ( location = 2 ) in vec3 in_normal;
layout ( location = 3 ) in vec2 in_texCoords;
layout (location = 10)   in ivec4 bone_ids;     // INT pointer
layout (location = 11)   in vec4 weights;

out vec2 TexCoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

const int MAX_BONES = 100;
uniform mat4 bones[MAX_BONES];

void main( )
{
	mat4 bone_transform = bones[bone_ids[0]] * weights[0];
		bone_transform += bones[bone_ids[1]] * weights[1];
		bone_transform += bones[bone_ids[2]] * weights[2];
		bone_transform += bones[bone_ids[3]] * weights[3];
		
		
	vec4 boned_position = bone_transform * vec4(in_position, 1.0); // transformed by bones
		
    gl_Position = projection * view * model * boned_position;
    TexCoords = in_texCoords;
}
