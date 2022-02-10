//For Shader Object
GLint model_gShaderProgramObject = 0;


//For Uniform
GLuint RRJ_ModelMatrixUniform;
GLuint RRJ_ViewMatrixUniform;
GLuint RRJ_ProjectionMatrixUniform;
GLuint RRJ_La_Uniform;
GLuint RRJ_Ld_Uniform;
GLuint RRJ_Ls_Uniform;
GLuint RRJ_LightPositionUniform;
GLuint RRJ_Ka_Uniform;
GLuint RRJ_Kd_Uniform;
GLuint RRJ_Ks_Uniform;
GLuint RRJ_MaterialShininess_Uniform;
GLuint RRJ_LKeyPressUniform;


//For Lights
GLfloat RRJ_LightAmbient[] = { 0.0f, 0.0f, 0.0f, 0.0f };
GLfloat RRJ_LightDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat RRJ_LightSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat RRJ_LightPosition[] = { 100.0f, 100.0f, 100.0f, 1.0f };

//For Material
GLfloat RRJ_MaterialAmbient[] = { 0.0f, 0.0f, 0.0f, 0.0f };
GLfloat RRJ_MaterialDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat RRJ_MaterialSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat RRJ_Shininess = 128.0f;


//Function Pointer for Display
typedef void (*pfnDisplay)(MODEL*);


//For Texture
GLuint model_samplerUniform;


void initialize_ModelWithLight(void) {

	void uninitialize(void);

	GLuint RRJ_iVertexShaderObject;
	GLuint RRJ_iFragmentShaderObject;

	/*********** Vertex Shader **********/
	RRJ_iVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *szVertexShaderSourceCode =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec3 vNormal;" \
		"in vec2 vTex;" \
		"out vec2 outTex;" \

		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \
		"uniform vec4 u_light_position;" \
		"out vec3 viewer_vector_VS;" \
		"out vec3 tNorm_VS;" \
		"out vec3 lightDirection_VS;" \
		"void main(void)" \
		"{" \

			"outTex = vTex;" \
 
			"vec4 eye_coordinate = u_view_matrix * u_model_matrix * vPosition;" \
			"viewer_vector_VS = vec3(-eye_coordinate);" \
			"tNorm_VS = mat3(u_view_matrix * u_model_matrix) * vNormal;" \
			"lightDirection_VS = vec3(u_light_position - eye_coordinate);" \
			"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \
		"}";

	glShaderSource(RRJ_iVertexShaderObject, 1, (const GLchar**)&szVertexShaderSourceCode, NULL);

	glCompileShader(RRJ_iVertexShaderObject);

	GLint iShaderCompileStatus = 0;
	GLint iInfoLogLength = 0;
	GLchar *szInfoLog = NULL;

	glGetShaderiv(RRJ_iVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE) {
		//Error ahe
		glGetShaderiv(RRJ_iVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetShaderInfoLog(RRJ_iVertexShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Vertex Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				DestroyWindow(ghwnd);

			}
		}
	}

	/********** Fragment Shader **********/
	RRJ_iFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *szFragmentShaderSourceCode =
		"#version 450 core" \
		"\n" \

		"in vec2 outTex;" \
		"uniform sampler2D u_sampler;" \


		"in vec3 viewer_vector_VS;" \
		"in vec3 tNorm_VS;" \
		"in vec3 lightDirection_VS;" \
		"out vec4 FragColor;" \
		"uniform vec3 u_La;" \
		"uniform vec3 u_Ld;" \
		"uniform vec3 u_Ls;" \
		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		"uniform float u_shininess;" \
		"uniform int u_LKeyPress;" \
		"void main(void)" \
		"{" \

			"vec4 color = vec4(0.0f);" \

			"if(u_LKeyPress == 1){" \
				"vec3 normalize_viewer_vector = normalize(viewer_vector_VS);" \
				"vec3 normalize_tNorm = normalize(tNorm_VS);" \
				"vec3 normalize_lightDirection = normalize(lightDirection_VS);" \
				"vec3 reflection_vector = reflect(-normalize_lightDirection, normalize_tNorm);" \
				"float s_dot_n = max(dot(normalize_lightDirection, normalize_tNorm), 0.0);" \
				"float r_dot_v = max(dot(reflection_vector, normalize_viewer_vector), 0.0);" \
				"vec3 ambient = u_La * u_Ka;" \
				"vec3 diffuse = u_Ld * u_Kd * s_dot_n;" \
				"vec3 specular = u_Ls * u_Ks * pow(r_dot_v, u_shininess);" \
				"vec3 Phong_ADS_Light = ambient + diffuse;" \
				"color = vec4(Phong_ADS_Light, 1.0) * texture(u_sampler, outTex);" \
			"}" \
			"else{" \
				"color = vec4(1.0, 1.0, 1.0, 1.0);" \
			"}" \


			"FragColor = color;" \
		"}";

	glShaderSource(RRJ_iFragmentShaderObject, 1,
		(const GLchar**)&szFragmentShaderSourceCode, NULL);

	glCompileShader(RRJ_iFragmentShaderObject);

	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;
	glGetShaderiv(RRJ_iFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE) {
		//Error ahe
		glGetShaderiv(RRJ_iFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetShaderInfoLog(RRJ_iFragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Fragment Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				DestroyWindow(ghwnd);
			}
		}
	}

	

	/********** Shader Program Object **********/
	model_gShaderProgramObject = glCreateProgram();

	glAttachShader(model_gShaderProgramObject, RRJ_iVertexShaderObject);
	glAttachShader(model_gShaderProgramObject, RRJ_iFragmentShaderObject);

	/********** Bind Vertex Attribute **********/
	glBindAttribLocation(model_gShaderProgramObject, ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(model_gShaderProgramObject, ATTRIBUTE_NORMAL, "vNormal");
	glBindAttribLocation(model_gShaderProgramObject, ATTRIBUTE_TEXCOORD0, "vTex");

	glLinkProgram(model_gShaderProgramObject);

	int iProgramLinkStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;
	glGetProgramiv(model_gShaderProgramObject, GL_LINK_STATUS, &iProgramLinkStatus);
	if (iProgramLinkStatus == GL_FALSE) {
		glGetProgramiv(model_gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetProgramInfoLog(model_gShaderProgramObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Shader Progame Object Linking Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				DestroyWindow(ghwnd);
			}
		}
	}

	/********** Getting Uniforms Location **********/
	RRJ_ModelMatrixUniform = glGetUniformLocation(model_gShaderProgramObject, "u_model_matrix");
	RRJ_ViewMatrixUniform = glGetUniformLocation(model_gShaderProgramObject, "u_view_matrix");
	RRJ_ProjectionMatrixUniform = glGetUniformLocation(model_gShaderProgramObject, "u_projection_matrix");
	RRJ_La_Uniform = glGetUniformLocation(model_gShaderProgramObject, "u_La");
	RRJ_Ld_Uniform = glGetUniformLocation(model_gShaderProgramObject, "u_Ld");
	RRJ_Ls_Uniform = glGetUniformLocation(model_gShaderProgramObject, "u_Ls");
	RRJ_LightPositionUniform = glGetUniformLocation(model_gShaderProgramObject, "u_light_position");
	RRJ_Ka_Uniform = glGetUniformLocation(model_gShaderProgramObject, "u_Ka");
	RRJ_Kd_Uniform = glGetUniformLocation(model_gShaderProgramObject, "u_Kd");
	RRJ_Ks_Uniform = glGetUniformLocation(model_gShaderProgramObject, "u_Ks");
	RRJ_MaterialShininess_Uniform = glGetUniformLocation(model_gShaderProgramObject, "u_shininess");
	RRJ_LKeyPressUniform = glGetUniformLocation(model_gShaderProgramObject, "u_LKeyPress");

	model_samplerUniform = glGetUniformLocation(model_gShaderProgramObject, "u_sampler");

}


void uninitialize_ModelWithLight(void){

	if(model_gShaderProgramObject){

		glUseProgram(model_gShaderProgramObject);

		GLint iShaderCount;
		GLint iShaderNumber;

		glGetProgramiv(model_gShaderProgramObject, GL_ATTACHED_SHADERS, &iShaderCount);

		GLuint *pShaders = (GLuint*)malloc(sizeof(GLuint) * iShaderCount);

		if(pShaders){

			glGetAttachedShaders(model_gShaderProgramObject, iShaderCount, &iShaderCount, pShaders);

			for(iShaderNumber = 0; iShaderNumber < iShaderCount; iShaderNumber++){

				glDetachShader(model_gShaderProgramObject, pShaders[iShaderNumber]);
				glDeleteShader(pShaders[iShaderNumber]);

				fprintf(gpFile, "\nShader %d Detached and Deleted", iShaderNumber+1);
			}

			free(pShaders);
			pShaders = NULL;

		}

		glUseProgram(0);

		glDeleteProgram(model_gShaderProgramObject);
		model_gShaderProgramObject = 0;

	}
}



void display_ModelWithLight(pfnDisplay func, MODEL *m, mat4 modelMat) {

	mat4 RRJ_translateMatrix;
	mat4 RRJ_rotateMatrix;
	mat4 RRJ_modelMatrix;
	mat4 RRJ_viewMatrix;


	glUseProgram(model_gShaderProgramObject);
	

		RRJ_translateMatrix = mat4::identity();
		RRJ_rotateMatrix = mat4::identity();
		RRJ_modelMatrix = mat4::identity();
		RRJ_viewMatrix = mat4::identity();

		// RRJ_translateMatrix = translate(0.0f, 0.0f, -2.0f);
		// // RRJ_rotateMatrix = rotate(RRJ_angle_Sphere, 1.0f, 0.0f, 0.0f) * rotate(RRJ_angle_Sphere, 0.0f, 1.0f, 0.0f) * rotate(RRJ_angle_Sphere, 0.0f, 0.0f, 1.0f);
		// RRJ_modelMatrix = RRJ_modelMatrix * RRJ_translateMatrix * RRJ_rotateMatrix;
		RRJ_viewMatrix = lookat(c.cameraPosition, c.cameraPosition + c.cameraFront, c.cameraUp);


		glUniformMatrix4fv(RRJ_ModelMatrixUniform, 1, GL_FALSE, modelMat);
		glUniformMatrix4fv(RRJ_ViewMatrixUniform, 1, GL_FALSE, RRJ_viewMatrix);
		glUniformMatrix4fv(RRJ_ProjectionMatrixUniform, 1, GL_FALSE, PerspectiveProjectionMatrix);


		glUniform1i(RRJ_LKeyPressUniform, 1);

		glUniform3fv(RRJ_La_Uniform, 1, RRJ_LightAmbient);
		glUniform3fv(RRJ_Ld_Uniform, 1, RRJ_LightDiffuse);
		glUniform3fv(RRJ_Ls_Uniform, 1, RRJ_LightSpecular);
		glUniform4fv(RRJ_LightPositionUniform, 1, RRJ_LightPosition);

		glUniform3fv(RRJ_Ka_Uniform, 1, RRJ_MaterialAmbient);
		glUniform3fv(RRJ_Kd_Uniform, 1, RRJ_MaterialDiffuse);
		glUniform3fv(RRJ_Ks_Uniform, 1, RRJ_MaterialSpecular);
		glUniform1f(RRJ_MaterialShininess_Uniform, RRJ_Shininess);

		if(m->texture){
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, m->texture);
			glUniform1i(model_samplerUniform, 0);
		}

		func(m);


		if(m->texture){
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, 0);
			
		}

	glUseProgram(0);

}
