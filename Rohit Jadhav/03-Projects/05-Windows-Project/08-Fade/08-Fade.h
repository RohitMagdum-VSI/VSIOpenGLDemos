//For Shader
GLuint vertexShaderObject_Fade;
GLuint fragmentShaderObject_Fade;
GLuint shaderProgramObject_Fade;

//For Uniform
GLuint mvpUniform_Fade;
GLuint fadeUniform_Fade;


//For Projection
mat4 orthoMatrix;


//For Rect
GLuint vao_Rect_Fade;
GLuint vbo_Rect_Position_Fade;

GLfloat fade_viewPortWidth = 1366.0f;
GLfloat fade_viewPortHeight = 768.0f;


void initialize_Fade(void){

	void uninitialize(void);
	void loadAllFonts(void);

	//Shader Object;
	GLint iVertexShaderObject;
	GLint iFragmentShaderObject;


	/********** Vertex Shader **********/
	iVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *szVertexShaderSourceCode =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \

		"uniform mat4 u_mvp_matrix;" \
		
		"void main() {" \

			"gl_Position = u_mvp_matrix * vPosition;" \
		
		"}";

	glShaderSource(iVertexShaderObject, 1,
		(const GLchar**)&szVertexShaderSourceCode, NULL);

	glCompileShader(iVertexShaderObject);

	GLint iShaderCompileStatus;
	GLint iInfoLogLength;
	GLchar *szInfoLog = NULL;
	glGetShaderiv(iVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE) {
		glGetShaderiv(iVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetShaderInfoLog(iVertexShaderObject, iInfoLogLength,
					&written, szInfoLog);
				fprintf(gpFile, "Fade : Vertex Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				DestroyWindow(ghwnd);
				exit(0);
			}
		}
	}

	/********** Fragment Shader **********/
	iFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *szFragmentShaderSourceCode =
		"#version 450 core" \
		"\n" \
		"uniform float u_fade;" \

		"out vec4 FragColor;" \

		"void main(void) {" \
			"FragColor = vec4(0.0, 0.0, 0.0, u_fade);" \
		"}";

	glShaderSource(iFragmentShaderObject, 1,
		(const GLchar**)&szFragmentShaderSourceCode, NULL);

	glCompileShader(iFragmentShaderObject);

	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetShaderiv(iFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE) {
		glGetShaderiv(iFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetShaderInfoLog(iFragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Fade : Fragment Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				DestroyWindow(ghwnd);
				exit(0);
			}
		}
	}

	/********** Shader Program Object **********/
	shaderProgramObject_Fade = glCreateProgram();

	glAttachShader(shaderProgramObject_Fade, iVertexShaderObject);
	glAttachShader(shaderProgramObject_Fade, iFragmentShaderObject);

	glBindAttribLocation(shaderProgramObject_Fade, ATTRIBUTE_POSITION, "vPosition");

	glLinkProgram(shaderProgramObject_Fade);

	GLint iProgramLinkingStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;
	glGetProgramiv(shaderProgramObject_Fade, GL_LINK_STATUS, &iProgramLinkingStatus);
	if (iProgramLinkingStatus == GL_FALSE) {
		glGetProgramiv(shaderProgramObject_Fade, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetProgramInfoLog(shaderProgramObject_Fade, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Fade : Shader Program Object Linking Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				DestroyWindow(ghwnd);
				exit(0);
			}
		}
	}


	mvpUniform_Fade = glGetUniformLocation(shaderProgramObject_Fade, "u_mvp_matrix");
	fadeUniform_Fade = glGetUniformLocation(shaderProgramObject_Fade, "u_fade");



	


	GLfloat Rect_Vertices[] = {
			fade_viewPortWidth / 2.0f, fade_viewPortHeight / 2.0f, 0.0f,
			-fade_viewPortWidth / 2.0f, fade_viewPortHeight / 2.0f, 0.0f,
			-fade_viewPortWidth / 2.0f, -fade_viewPortHeight / 2.0f, 0.0f,
			fade_viewPortWidth / 2.0f, -fade_viewPortHeight / 2.0f, 0.0f,
	};



	/********** Vao Rect On Which We Apply Texture **********/
	glGenVertexArrays(1, &vao_Rect_Fade);
	glBindVertexArray(vao_Rect_Fade);

	/********** Position **********/
	glGenBuffers(1, &vbo_Rect_Position_Fade);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_Position_Fade);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Rect_Vertices),
		Rect_Vertices,
		GL_STATIC_DRAW);

	glVertexAttribPointer(ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

}

void uninitialize_Fade(void){

	if(vbo_Rect_Position_Fade){
		glDeleteBuffers(1, &vbo_Rect_Position_Fade);
		vbo_Rect_Position_Fade = 0;
	}

	if(vao_Rect_Fade){
		glDeleteVertexArrays(1, &vao_Rect_Fade);
		vao_Rect_Fade = 0;
	}


	if(shaderProgramObject_Fade){

		glUseProgram(shaderProgramObject_Fade);

			if(fragmentShaderObject_Fade){
				glDetachShader(shaderProgramObject_Fade, fragmentShaderObject_Fade);
				glDeleteShader(fragmentShaderObject_Fade);
				fragmentShaderObject_Fade = 0;
			}

			if(vertexShaderObject_Fade){
				glDetachShader(shaderProgramObject_Fade, vertexShaderObject_Fade);
				glDeleteShader(vertexShaderObject_Fade);
				vertexShaderObject_Fade = 0;
			}

		glUseProgram(0);
		glDeleteProgram(shaderProgramObject_Fade);
		shaderProgramObject_Fade = 0;
	}

}




void display_Fade(GLfloat fadeValue){

	mat4 modelViewMatrix;
	mat4 modelViewProjectionMatrix;
	
	orthoMatrix = mat4::identity();

	orthoMatrix = ortho(
				-fade_viewPortWidth / 2.0, fade_viewPortWidth / 2.0,	// L, R
				-fade_viewPortHeight/ 2.0, fade_viewPortHeight / 2.0,	// B, T
				-1.0, 1.0);							// N, F

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glUseProgram(shaderProgramObject_Fade);


		/********** Rectangle **********/
		modelViewMatrix = mat4::identity();
		modelViewProjectionMatrix = mat4::identity();

		modelViewProjectionMatrix = orthoMatrix * modelViewMatrix;

		glUniformMatrix4fv(mvpUniform_Fade, 1, GL_FALSE, modelViewProjectionMatrix);
		glUniform1f(fadeUniform_Fade, fadeValue);


		

		glBindVertexArray(vao_Rect_Fade);
			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		glBindVertexArray(0);

			
	glUseProgram(0);

	glDisable(GL_BLEND);

}