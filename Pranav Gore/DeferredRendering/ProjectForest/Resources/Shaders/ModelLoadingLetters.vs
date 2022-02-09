#version 330 core
layout ( location = 0 ) in vec3 in_position;
layout ( location = 2 ) in vec3 in_normal;
layout ( location = 3 ) in vec2 in_texCoords;

out vec2 TexCoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec4 clipPlane;

void main( )
{	
    gl_Position = projection * view * model * vec4(in_position, 1.0);
    
    gl_ClipDistance[0] = dot(model * vec4(in_position, 1.0), clipPlane);

    TexCoords = in_texCoords;
}