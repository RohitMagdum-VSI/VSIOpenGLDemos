#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "../common/Common.h"
#include "BasicPlanetShader.h"

extern FILE* gpFile;

bool InitializeBasicPlanetShaderProgram(BASIC_PLANET_SHADER* pShaderObj)
{
	pShaderObj->ShaderObject.uiVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar* pchVertexShaderSource =
		"#version 430 core\n" \
		"\n" \
		"in vec4 vPosition;\n" \
		"in vec2 vTexture0_Coord;\n" \
		"in vec3 vNormal;\n" \
		"uniform mat4 u_model_matrix;\n" \
		"uniform mat4 u_view_matrix;\n" \
		"uniform mat4 u_projection_matrix;\n" \
		"uniform vec4 u_light_position;\n" \
		"out vec3 out_transformed_normals;\n" \
		"out vec3 out_light_direction;\n" \
		"out vec3 out_viewer_vector;\n" \
		"out vec2 out_texture0_coord;\n" \
		"void main(void)\n" \
		"{\n" \
		"vec4 eye_coordinates = u_view_matrix * u_model_matrix * vPosition;\n" \
		"out_transformed_normals = mat3(u_view_matrix * u_model_matrix) * vNormal;\n" \
		"out_light_direction = vec3(u_light_position) - eye_coordinates.xyz;\n" \
		"out_viewer_vector = -eye_coordinates.xyz;\n" \
		"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;\n" \
		"out_texture0_coord = vTexture0_Coord;\n" \
		"}\n";

	glShaderSource(pShaderObj->ShaderObject.uiVertexShaderObject, 1, (const GLchar**)&pchVertexShaderSource, NULL);

	glCompileShader(pShaderObj->ShaderObject.uiVertexShaderObject);
	GLint iInfoLogLength = 0;
	GLint iShaderCompiledStatus = 0;
	char* szInfoLog = NULL;

	glGetShaderiv(pShaderObj->ShaderObject.uiVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(pShaderObj->ShaderObject.uiVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(pShaderObj->ShaderObject.uiVertexShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Basic Planet VertexShader compilation log: %s\n", szInfoLog);
				free(szInfoLog);
				return false;
			}
		}
	}

	pShaderObj->ShaderObject.uiFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar* pchFragmentShaderSource =
		"#version 430 core\n" \
		"\n" \
		"in vec2 out_texture0_coord;\n" \
		"in vec3 out_transformed_normals;\n" \
		"in vec3 out_light_direction;\n" \
		"in vec3 out_viewer_vector;\n" \
		"uniform vec3 u_La;\n" \
		"uniform vec3 u_Ld;\n" \
		"uniform vec3 u_Ls;\n" \
		"uniform vec3 u_Ka;\n" \
		"uniform vec3 u_Kd;\n" \
		"uniform vec3 u_Ks;\n" \
		"uniform float u_material_shininess;\n" \
		"out vec4 FragColor;\n" \
		"uniform sampler2D u_texture0_sampler;\n" \
		"void main(void)\n" \
		"{\n" \
		"vec3 ads_light_color;\n" \
		"vec3 normalized_transformed_normals = normalize(out_transformed_normals);\n" \
		"vec3 normalized_light_direction = normalize(out_light_direction);\n" \
		"vec3 normalized_viewer_vector = normalize(out_viewer_vector);\n" \
		"vec3 ambient = u_La * u_Ka;\n" \
		"float tn_dot_ld = max(dot(normalized_transformed_normals, normalized_light_direction), 0.0);\n" \
		"vec3 diffuse = u_Ld * u_Kd * tn_dot_ld;\n" \
		"vec3 reflection_vector = reflect(-normalized_light_direction, normalized_transformed_normals);\n" \
		"vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector, normalized_viewer_vector), 0.0), u_material_shininess);\n" \
		"ads_light_color = ambient + diffuse + specular;\n" \
		"vec4 texColor = texture(u_texture0_sampler, out_texture0_coord);\n" \
		"FragColor = texColor * vec4(ads_light_color, 1.0);\n" \
		"}\n";

	glShaderSource(pShaderObj->ShaderObject.uiFragmentShaderObject, 1, (const GLchar**)&pchFragmentShaderSource, NULL);

	glCompileShader(pShaderObj->ShaderObject.uiFragmentShaderObject);
	glGetShaderiv(pShaderObj->ShaderObject.uiFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(pShaderObj->ShaderObject.uiFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(pShaderObj->ShaderObject.uiFragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Basic Planet FragmentShader compilation log: %s\n", szInfoLog);
				free(szInfoLog);
				return false;
			}
		}
	}

	pShaderObj->ShaderObject.uiShaderProgramObject = glCreateProgram();

	glAttachShader(pShaderObj->ShaderObject.uiShaderProgramObject, pShaderObj->ShaderObject.uiVertexShaderObject);
	glAttachShader(pShaderObj->ShaderObject.uiShaderProgramObject, pShaderObj->ShaderObject.uiFragmentShaderObject);

	glBindAttribLocation(pShaderObj->ShaderObject.uiShaderProgramObject, OGL_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(pShaderObj->ShaderObject.uiShaderProgramObject, OGL_ATTRIBUTE_TEXTURE0, "vTexture0_Coord");
	glBindAttribLocation(pShaderObj->ShaderObject.uiShaderProgramObject, OGL_ATTRIBUTE_NORMAL, "vNormal");

	glLinkProgram(pShaderObj->ShaderObject.uiShaderProgramObject);
	GLint iShaderProgramLinkStatus = 0;
	glGetProgramiv(pShaderObj->ShaderObject.uiShaderProgramObject, GL_LINK_STATUS, &iShaderProgramLinkStatus);
	if (iShaderProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(pShaderObj->ShaderObject.uiShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			GLsizei written;
			glGetProgramInfoLog(pShaderObj->ShaderObject.uiShaderProgramObject, iInfoLogLength, &written, szInfoLog);
			fprintf(gpFile, "Basic Planet Shader Program Link log: %s\n", szInfoLog);
			free(szInfoLog);
			return false;
		}
	}

	pShaderObj->uiModelMatrixUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_model_matrix");
	pShaderObj->uiViewMatrixUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_view_matrix");
	pShaderObj->uiProjectionMatrixUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_projection_matrix");
	pShaderObj->uiTextureSamplerUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_texture0_sampler");
	
	pShaderObj->uiLaUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_La");
	pShaderObj->uiLdUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_Ld");
	pShaderObj->uiLsUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_Ls");
	pShaderObj->uiLightPositionUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_light_position");
	
	pShaderObj->uiKaUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_Ka");
	pShaderObj->uiKdUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_Kd");
	pShaderObj->uiKsUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_Ks");
	pShaderObj->uiMaterialShininessUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_material_shininess");

	return true;
}

void UnInitializeBasicPlanetShaderProgram(BASIC_PLANET_SHADER* pShaderObj)
{
	glDetachShader(pShaderObj->ShaderObject.uiShaderProgramObject, pShaderObj->ShaderObject.uiVertexShaderObject);
	glDetachShader(pShaderObj->ShaderObject.uiShaderProgramObject, pShaderObj->ShaderObject.uiFragmentShaderObject);

	glDeleteShader(pShaderObj->ShaderObject.uiVertexShaderObject);
	pShaderObj->ShaderObject.uiVertexShaderObject = 0;

	glDeleteShader(pShaderObj->ShaderObject.uiFragmentShaderObject);
	pShaderObj->ShaderObject.uiFragmentShaderObject = 0;

	glDeleteProgram(pShaderObj->ShaderObject.uiShaderProgramObject);
	pShaderObj->ShaderObject.uiShaderProgramObject = 0;

	return;
}