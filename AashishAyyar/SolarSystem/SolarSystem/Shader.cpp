#include "Common.h"

extern FILE *gpFile;

//
//	Shader utility functions 
//
BOOLEAN CheckCompileStatus(GLuint shaderObject) 
{
	GLint iInfoLogLength = 0;
	GLint iShaderCompiledStatus = 0;
	char *szInfoLog = NULL;

	glGetShaderiv(shaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(shaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(shaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Compile log: %s\n", szInfoLog);
				free(szInfoLog);
				return FALSE;
			}
		}
	}

	return TRUE;
}

BOOLEAN CheckLinkStatus(GLuint programObject) 
{
	GLint iInfoLogLength = 0;
	char *szInfoLog = NULL;

	GLint iShaderProgramLinkStatus = 0;
	glGetProgramiv(programObject, GL_LINK_STATUS, &iShaderProgramLinkStatus);
	if (iShaderProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(programObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);
			GLsizei written;
			glGetProgramInfoLog(programObject, iInfoLogLength, &written, szInfoLog);
			fprintf(gpFile, "Link log: %s\n", szInfoLog);
			free(szInfoLog);
			return FALSE;
		}
	}

	return TRUE;
}

void CleanupShader(P_SHADER pShader)
{
	glDetachShader(pShader->renderProgram, pShader->vso);
	glDetachShader(pShader->renderProgram, pShader->fso);

	glDeleteShader(pShader->vso);
	pShader->vso = 0;

	glDeleteShader(pShader->fso);
	pShader->fso = 0;

	glDeleteProgram(pShader->renderProgram);
	pShader->renderProgram = 0;
}

//
//	Planet Shader
//
BOOL InitPlanetShaders(PLANET_SHADER &Planet)
{
	//
	//	Vertex Shader
	//
	Planet.Shader.vso = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *vs =
	{
		"#version 440 core																		\n"
		"																						\n"
		"in vec4 vPosition;																		\n"
		"in vec3 vNormal;																		\n"
		"in vec2 vTexture0_Coord;																\n"
		"																						\n"
		"uniform mat4 u_model_matrix;															\n"
		"uniform mat4 u_view_matrix;															\n"
		"uniform mat4 u_projection_matrix;														\n"
		"uniform vec4 u_light_position;															\n"
		"																						\n"
		"out vec3 out_transformed_normals;														\n"
		"out vec3 out_light_direction;															\n"
		"out vec3 out_viewer_vector;															\n"
		"out vec2 out_texture0_coord;															\n"
		"																						\n"
		"																						\n"
		"void main(void)																		\n"
		"{																						\n"
		"	vec4 eye_coordinates = u_view_matrix * u_model_matrix * vPosition;					\n"
		"	out_transformed_normals = mat3(u_view_matrix * u_model_matrix) * vNormal;			\n"
		"	out_light_direction = vec3(u_light_position) - eye_coordinates.xyz;					\n"
		"	out_viewer_vector = -eye_coordinates.xyz;											\n"
		"																						\n"
		"	gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;		\n"
		"	out_texture0_coord = vTexture0_Coord;												\n"
		"}																						\n"
	};

	glShaderSource(Planet.Shader.vso, 1, (const GLchar **)&vs, NULL);
	glCompileShader(Planet.Shader.vso);

	if (!CheckCompileStatus(Planet.Shader.vso))
	{
		CleanupPlanetShader(Planet);
		return FALSE;
	}

	//
	//	Fragment Shader
	//

	Planet.Shader.fso = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *fs =
	{
		"#version 440 core																											\n"
		"																															\n"
		"in vec3 out_transformed_normals;																							\n"
		"in vec3 out_light_direction;																								\n"
		"in vec3 out_viewer_vector;																									\n"
		"in vec2 out_texture0_coord;																								\n"
		"																															\n"
		"uniform vec3 u_La;																											\n"
		"uniform vec3 u_Ld;																											\n"
		"uniform vec3 u_Ls;																											\n"
		"uniform vec3 u_Ka;																											\n"
		"uniform vec3 u_Kd;																											\n"
		"uniform vec3 u_Ks;																											\n"
		"uniform float u_material_shininess;																						\n"
		"uniform sampler2D u_texture0_sampler;																						\n"
		"																															\n"
		"out vec4 FragColor;																										\n"
		"																															\n"
		"void main(void)																											\n"
		"{																															\n"
		"	vec3 ads_light_color;																									\n"
		"	vec3 normalized_transformed_normals = normalize(out_transformed_normals);												\n"
		"	vec3 normalized_light_direction = normalize(out_light_direction);														\n"
		"	vec3 normalized_viewer_vector = normalize(out_viewer_vector);															\n"
		"	vec3 ambient = u_La * u_Ka;																								\n"
		"																															\n"
		"	float tn_dot_ld = max(dot(normalized_transformed_normals, normalized_light_direction), 0.0f);							\n"
		"	vec3 diffuse = u_Ld * u_Kd * tn_dot_ld;																					\n"
		"	vec3 reflection_vector = reflect(-normalized_light_direction, normalized_transformed_normals);							\n"
		"	vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector, normalized_viewer_vector), 0.0), u_material_shininess);	\n"
		"																															\n"
		"	ads_light_color = ambient + diffuse + specular;																			\n"
		"	vec4 textureColor = texture(u_texture0_sampler, out_texture0_coord);													\n"
		"	FragColor = textureColor * vec4(ads_light_color, 1.0);																	\n"
		"}																															\n"
	};

	glShaderSource(Planet.Shader.fso, 1, (const GLchar **)&fs, NULL);
	glCompileShader(Planet.Shader.fso);
	if (!CheckCompileStatus(Planet.Shader.fso))
	{
		CleanupPlanetShader(Planet);
		return FALSE;
	}

	//
	//	Program Object
	//
	Planet.Shader.renderProgram = glCreateProgram();

	glAttachShader(Planet.Shader.renderProgram, Planet.Shader.vso);
	glAttachShader(Planet.Shader.renderProgram, Planet.Shader.fso);

	glBindAttribLocation(Planet.Shader.renderProgram, OGL_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(Planet.Shader.renderProgram, OGL_ATTRIBUTE_TEXTURE0, "vTexture0_Coord");
	glBindAttribLocation(Planet.Shader.renderProgram, OGL_ATTRIBUTE_NORMAL, "vNormal");

	glLinkProgram(Planet.Shader.renderProgram);
	if (!CheckLinkStatus(Planet.Shader.renderProgram))
	{
		CleanupPlanetShader(Planet);
		return FALSE;
	}

	//
	//	Initialize Uniforms 
	//

	Planet.modelMatrixUniform = glGetUniformLocation(Planet.Shader.renderProgram, "u_model_matrix");
	Planet.viewMatrixUniform = glGetUniformLocation(Planet.Shader.renderProgram, "u_view_matrix");
	Planet.projectionMatrixUniform = glGetUniformLocation(Planet.Shader.renderProgram, "u_projection_matrix");
	Planet.textureSamplerUniform = glGetUniformLocation(Planet.Shader.renderProgram, "u_texture0_sampler");

	Planet.laUniform = glGetUniformLocation(Planet.Shader.renderProgram, "u_La");
	Planet.ldUniform = glGetUniformLocation(Planet.Shader.renderProgram, "u_Ld");
	Planet.lsUniform = glGetUniformLocation(Planet.Shader.renderProgram, "u_Ls");
	Planet.lightPosUniform = glGetUniformLocation(Planet.Shader.renderProgram, "u_light_position");

	Planet.kaUniform = glGetUniformLocation(Planet.Shader.renderProgram, "u_Ka");
	Planet.kdUniform = glGetUniformLocation(Planet.Shader.renderProgram, "u_Kd");
	Planet.ksUniform = glGetUniformLocation(Planet.Shader.renderProgram, "u_Ks");

	Planet.materialShininessUniform = glGetUniformLocation(Planet.Shader.renderProgram, "u_material_shininess");

	return TRUE;
}

void CleanupPlanetShader(PLANET_SHADER &pPlanet)
{
	CleanupShader(&pPlanet.Shader);
}

//
//	Color Shader
//
BOOL InitColorShaders(COLOR_SHADER &ColorShader)
{
	//
	//	Vertex Shader
	//
	ColorShader.Shader.vso = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *vs =
	{
		"#version 440 core							\n"
		"											\n"
		"in vec4 vPosition;							\n"
		"in vec3 vColor;							\n"
		"											\n"
		"out vec3 v_out_color;						\n"
		"uniform mat4 u_mvp_matrix;					\n"
		"											\n"
		"void main(void)							\n"
		"{											\n"
		"	gl_Position = u_mvp_matrix * vPosition;	\n"
		"	v_out_color = vColor;					\n"
		"}											\n"
	};

	glShaderSource(ColorShader.Shader.vso, 1, (const GLchar **)&vs, NULL);
	glCompileShader(ColorShader.Shader.vso);
	if (!CheckCompileStatus(ColorShader.Shader.vso))
	{
		CleanupColorShader(ColorShader);
		return FALSE;
	}

	//
	//	Fragment Shader
	//

	ColorShader.Shader.fso = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *fs =
	{
		"#version 440 core													\n"
		"																	\n"
		"in vec3 v_out_color;												\n"
		"out vec4 FragColor;												\n"
		"																	\n"
		"void main(void)													\n"
		"{																	\n"
		"	FragColor = vec4(v_out_color, 1.0f);							\n"
		"}																	\n"
	};

	glShaderSource(ColorShader.Shader.fso, 1, (const GLchar **)&fs, NULL);
	glCompileShader(ColorShader.Shader.fso);
	if (!CheckCompileStatus(ColorShader.Shader.fso))
	{
		CleanupColorShader(ColorShader);
		return FALSE;
	}

	//
	//	Program Object
	//
	ColorShader.Shader.renderProgram = glCreateProgram();

	glAttachShader(ColorShader.Shader.renderProgram, ColorShader.Shader.vso);
	glAttachShader(ColorShader.Shader.renderProgram, ColorShader.Shader.fso);

	glBindAttribLocation(ColorShader.Shader.renderProgram, OGL_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(ColorShader.Shader.renderProgram, OGL_ATTRIBUTE_COLOR, "vColor");

	glLinkProgram(ColorShader.Shader.renderProgram);
	if (!CheckLinkStatus(ColorShader.Shader.renderProgram))
	{
		CleanupColorShader(ColorShader);
		return FALSE;
	}

	//
	//	Initialize Uniforms 
	//

	ColorShader.mvpUniform = glGetUniformLocation(ColorShader.Shader.renderProgram, "u_mvp_matrix");

	return TRUE;
}

void CleanupColorShader(COLOR_SHADER &ColorShader)
{
	CleanupShader(&ColorShader.Shader);
}

//
//	Picking Shader
//
BOOL InitPickingShader(PICKING_SHADER &PickingShader) 
{
	//
	//	Vertex Shader
	//
	PickingShader.Shader.vso = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *vs =
	{
		"#version 440 core																	\n"
		"																					\n"
		"in vec4 vPosition;																	\n"
		"																					\n"
		"uniform mat4 u_model_matrix;														\n"
		"uniform mat4 u_view_matrix;														\n"
		"uniform mat4 u_projection_matrix;													\n"
		"																					\n"
		"void main(void)																	\n"
		"{																					\n"
		"	gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;	\n"
		"}																					\n"
	};

	glShaderSource(PickingShader.Shader.vso, 1, (const GLchar **)&vs, NULL);
	glCompileShader(PickingShader.Shader.vso);
	if (!CheckCompileStatus(PickingShader.Shader.vso))
	{
		CleanupPickingShader(PickingShader);
		return FALSE;
	}

	//
	//	Fragment Shader
	//

	PickingShader.Shader.fso = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *fs =
	{
		"#version 440 core														\n"
		"																		\n"
		"out vec4 FragColor;													\n"
		"																		\n"
		"uniform int u_object_index_uniform;									\n"
		"																		\n"
		"void main(void)														\n"
		"{																		\n"
		"	FragColor = vec4(u_object_index_uniform / 255.0, 1.0f, 0.0, 1.0);	\n"
		"}																		\n"
	};

	glShaderSource(PickingShader.Shader.fso, 1, (const GLchar **)&fs, NULL);
	glCompileShader(PickingShader.Shader.fso);
	if (!CheckCompileStatus(PickingShader.Shader.fso))
	{
		CleanupPickingShader(PickingShader);
		return FALSE;
	}

	//
	//	Program Object
	//
	PickingShader.Shader.renderProgram = glCreateProgram();

	glAttachShader(PickingShader.Shader.renderProgram, PickingShader.Shader.vso);
	glAttachShader(PickingShader.Shader.renderProgram, PickingShader.Shader.fso);

	glBindAttribLocation(PickingShader.Shader.renderProgram, OGL_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(PickingShader.Shader.renderProgram, OGL_ATTRIBUTE_COLOR, "vColor");

	glLinkProgram(PickingShader.Shader.renderProgram);
	if (!CheckLinkStatus(PickingShader.Shader.renderProgram))
	{
		CleanupPickingShader(PickingShader);
		return FALSE;
	}

	//
	//	Initialize Uniforms 
	//

	PickingShader.modelMatrixUniform = glGetUniformLocation(PickingShader.Shader.renderProgram, "u_model_matrix");
	PickingShader.viewMatrixUniform = glGetUniformLocation(PickingShader.Shader.renderProgram, "u_view_matrix");
	PickingShader.projectionMatrixUniform = glGetUniformLocation(PickingShader.Shader.renderProgram, "u_projection_matrix");
	PickingShader.objectIDUniform = glGetUniformLocation(PickingShader.Shader.renderProgram, "u_object_index_uniform");


	return TRUE;
}

void CleanupPickingShader(PICKING_SHADER &PickingShader) 
{
	CleanupShader(&PickingShader.Shader);
}