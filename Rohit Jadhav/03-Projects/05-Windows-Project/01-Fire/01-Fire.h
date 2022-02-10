// ************************************
// FIRE
// ************************************


//For Shader Object
GLint Fire_gShaderProgramObject = 0;


//For Rect
GLuint Fire_Vao_Rect;
GLuint Fire_Vbo_Rect_Pos;
GLuint Fire_Vbo_Rect_Texcoord;

//For Fire
GLuint guiFireColor;
GLuint guiFireAlpha;
GLuint guiFireNoise;

GLfloat Fire_gfOctave[] = {1.0f, 2.0f, 3.0f};
GLfloat Fire_gfSpeed[] = {1.3f, 2.1f, 2.3f};
GLfloat Fire_gfAnimationTime = 0.0f;
const GLfloat Fire_gfRate = 0.01f;



//For Uniforms
GLuint Fire_Terrain_mvpUniform;		
GLuint Fire_choiceUniform;

//For Fire
GLuint samplerFireColorUniform;
GLuint samplerFireAlphaUniform;
GLuint samplerFireNoiseUniform;

GLuint Fire_octaveUniform;
GLuint Fire_speedUniform;
GLuint Fire_animationTimeUniform;




void initialize_Fire(void){

	GLuint iVertexShaderObject;
	GLuint iFragmentShaderObject;
	GLuint LoadTexture(TCHAR[], GLint flag);

	/*********** Vertex Shader **********/

	iVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *szVertexShaderSourceCode =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec2 vTexCoord;" \

		"out vec2 outTexCoord;" \
		"out vec2 outTexCoord1;" \
		"out vec2 outTexCoord2;" \
		"out vec2 outTexCoord3;" \

		"uniform vec3 u_octave;" \
		"uniform vec3 u_speed;" \
		"uniform float u_time;" \

		"uniform mat4 u_mvp_matrix;" \
		

		"void main(void)" \
		"{" \
			"gl_Position = u_mvp_matrix * vPosition;" \
			
			"outTexCoord = vTexCoord;" \

			"outTexCoord1 = vTexCoord * u_octave.x;" \
			"outTexCoord1.y += (u_time * u_speed.x);" \

			"outTexCoord2 = vTexCoord * u_octave.y;" \
			"outTexCoord2.y += (u_time * u_speed.y);" \

			"outTexCoord3 = vTexCoord * u_octave.z;" \
			"outTexCoord3.y += (u_time * u_speed.z);" \


		"}";

	glShaderSource(iVertexShaderObject, 1, (const GLchar**)&szVertexShaderSourceCode, NULL);

	glCompileShader(iVertexShaderObject);

	GLint iShaderCompileStatus = 0;
	GLint iInfoLogLength = 0;
	GLchar *szInfoLog = NULL;

	glGetShaderiv(iVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE) {
		//Error ahe
		glGetShaderiv(iVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetShaderInfoLog(iVertexShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Vertex Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				DestroyWindow(ghwnd);

			}
		}
	}

	/********** Fragment Shader **********/
	iFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *szFragmentShaderSourceCode =
		"#version 450 core" \
		"\n" \

		"in vec2 outTexCoord;" \
		"in vec2 outTexCoord1;" \
		"in vec2 outTexCoord2;" \
		"in vec2 outTexCoord3;" \

		"uniform sampler2D u_samplerFireColor;" \
		"uniform sampler2D u_samplerFireAlpha;" \
		"uniform sampler2D u_samplerFireNoise;" \

	


		"uniform int u_choice;" \

		"out vec4 FragColor;" \
		"void main(void)" \
		"{" \

			"vec4 noise1, noise2, noise3, finalNoise;" \
			"vec2 texcoord;" \
			"float perTurbulance;" \
			"vec2 fireTexcoord;" \
			"vec4 fireColor, fireAlpha;" \


			"noise1 = texture(u_samplerFireNoise, outTexCoord1);" \
			"noise1 = (noise1 - 0.50f) * 2.0f;" \
			"noise1.xy = noise1.xy * vec2(0.1f, 0.2f);" \


			"noise2 = texture(u_samplerFireNoise, outTexCoord2);" \
			"noise2 = (noise2 - 0.50f) * 2.0f;" \
			"noise2.xy = noise2.xy * vec2(0.1f, 0.3f);" \


			"noise3 = texture(u_samplerFireNoise, outTexCoord3);" \
			"noise3 = (noise3 - 0.50f) * 2.0f;" \
			"noise3.xy = noise3.xy * vec2(0.1f, 0.1f);" \


			"finalNoise = noise1 + noise2 + noise3;" \

			"perTurbulance = ((1.0f - outTexCoord.y) * 0.8f) + 0.5f;" \
			
			"fireTexcoord = (finalNoise.xy * perTurbulance) + outTexCoord;" \

			"fireColor = texture(u_samplerFireColor, fireTexcoord);" \
			"fireAlpha = texture(u_samplerFireAlpha, fireTexcoord);" \

			//"fireColor.a = (fireAlpha.r + fireAlpha.g + fireAlpha.b + fireAlpha.a) / 4.0f;" \
			
			"fireColor.a = fireAlpha.r;" \

			// "FragColor = vec4(fireColor.rgb, fireColor.a);" \

			"FragColor = vec4(fireColor);" \

		"}";

	glShaderSource(iFragmentShaderObject, 1,
		(const GLchar**)&szFragmentShaderSourceCode, NULL);

	glCompileShader(iFragmentShaderObject);

	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;
	glGetShaderiv(iFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE) {
		//Error ahe
		glGetShaderiv(iFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetShaderInfoLog(iFragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Fragment Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				DestroyWindow(ghwnd);
			}
		}
	}


	/********** Shader Program Object **********/
	Fire_gShaderProgramObject = glCreateProgram();

	glAttachShader(Fire_gShaderProgramObject, iVertexShaderObject);
	glAttachShader(Fire_gShaderProgramObject, iFragmentShaderObject);


	/********** Bind Vertex Attribute **********/
	glBindAttribLocation(Fire_gShaderProgramObject, ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(Fire_gShaderProgramObject, ATTRIBUTE_TEXCOORD0, "vTexCoord");

	glLinkProgram(Fire_gShaderProgramObject);

	int iProgramLinkStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;
	glGetProgramiv(Fire_gShaderProgramObject, GL_LINK_STATUS, &iProgramLinkStatus);
	if (iProgramLinkStatus == GL_FALSE) {
		glGetProgramiv(Fire_gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetProgramInfoLog(Fire_gShaderProgramObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Shader Progame Object Linking Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				DestroyWindow(ghwnd);
			}
		}
	}

	/********** Getting Uniforms Location **********/
	Fire_Terrain_mvpUniform = glGetUniformLocation(Fire_gShaderProgramObject, "u_mvp_matrix");
	Fire_choiceUniform = glGetUniformLocation(Fire_gShaderProgramObject, "u_choice");

	samplerFireColorUniform = glGetUniformLocation(Fire_gShaderProgramObject, "u_samplerFireColor");
	samplerFireAlphaUniform = glGetUniformLocation(Fire_gShaderProgramObject, "u_samplerFireAlpha");
	samplerFireNoiseUniform = glGetUniformLocation(Fire_gShaderProgramObject, "u_samplerFireNoise");

	Fire_octaveUniform = glGetUniformLocation(Fire_gShaderProgramObject, "u_octave");
	Fire_speedUniform = glGetUniformLocation(Fire_gShaderProgramObject, "u_speed");
	Fire_animationTimeUniform = glGetUniformLocation(Fire_gShaderProgramObject, "u_time");




	/********** Rect Vertices Information **********/
	GLfloat Rect_Pos[] = {
		
		//Front
		1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,

	};


	GLfloat Rect_Texcoord[] = {
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
	};


	/********** Creating Vertex Array Object **********/
	glGenVertexArrays(1, &Fire_Vao_Rect);
	glBindVertexArray(Fire_Vao_Rect);

		/********** Creating Vertex Buffer Object Position *********/
		glGenBuffers(1, &Fire_Vbo_Rect_Pos);
		glBindBuffer(GL_ARRAY_BUFFER, Fire_Vbo_Rect_Pos);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Rect_Pos), Rect_Pos, GL_STATIC_DRAW);
		glVertexAttribPointer(ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		/********** Creating Vertex Buffer Object Color *********/
		glGenBuffers(1, &Fire_Vbo_Rect_Texcoord);
		glBindBuffer(GL_ARRAY_BUFFER, Fire_Vbo_Rect_Texcoord);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Rect_Texcoord), Rect_Texcoord, GL_STATIC_DRAW);
		glVertexAttribPointer(ATTRIBUTE_TEXCOORD0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(ATTRIBUTE_TEXCOORD0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	
	glBindVertexArray(0);


	glEnable(GL_TEXTURE_2D);

	//1 -> Texture with clamp 0 -> texture without clamp
	guiFireColor = LoadTexture(MAKEINTRESOURCE(ID_FIRE_COLOR), 1);
	guiFireAlpha= LoadTexture(MAKEINTRESOURCE(ID_FIRE_ALPHA), 1);
	guiFireNoise = LoadTexture(MAKEINTRESOURCE(ID_FIRE_NOISE), 0);

}

void uninitialize_Fire(void){

	// Fire
	if(guiFireColor){
		glDeleteTextures(1, &guiFireColor);
		guiFireColor = 0;
	}

	if(guiFireAlpha){
		glDeleteTextures(1, &guiFireAlpha);
		guiFireAlpha = 0;
	}

	if(guiFireNoise){
		glDeleteTextures(1, &guiFireNoise);
		guiFireNoise = 0;
	}

	if (Fire_Vbo_Rect_Texcoord) {
		glDeleteBuffers(1, &Fire_Vbo_Rect_Texcoord);
		Fire_Vbo_Rect_Texcoord = 0;
	}

	if (Fire_Vbo_Rect_Pos) {
		glDeleteBuffers(1, &Fire_Vbo_Rect_Pos);
		Fire_Vbo_Rect_Pos = 0;
	}

	if (Fire_Vao_Rect) {
		glDeleteVertexArrays(1, &Fire_Vao_Rect);
		Fire_Vao_Rect = 0;
	}




	if(Fire_gShaderProgramObject){

		glUseProgram(Fire_gShaderProgramObject);

		GLint iShaderCount;
		GLint iShaderNumber;

		glGetProgramiv(Fire_gShaderProgramObject, GL_ATTACHED_SHADERS, &iShaderCount);

		GLuint *pShaders = (GLuint*)malloc(sizeof(GLuint) * iShaderCount);

		if(pShaders){

			glGetAttachedShaders(Fire_gShaderProgramObject, iShaderCount, &iShaderCount, pShaders);

			for(iShaderNumber = 0; iShaderNumber < iShaderCount; iShaderNumber++){

				glDetachShader(Fire_gShaderProgramObject, pShaders[iShaderNumber]);
				glDeleteShader(pShaders[iShaderNumber]);

				fprintf(gpFile, "Fire: Shader %d Detached and Deleted\n", iShaderNumber+1);
			}

			free(pShaders);
			pShaders = NULL;

		}

		glUseProgram(0);

		glDeleteProgram(Fire_gShaderProgramObject);
		Fire_gShaderProgramObject = 0;

	}

}


void update_Fire(void){
	Fire_gfAnimationTime -= Fire_gfRate;
}


void display_Fire(void){

	glUseProgram(Fire_gShaderProgramObject);
		
		mat4 translateMatrix;
		mat4 modelViewMatrix;
		mat4 modelViewProjectionMatrix;


		// Rect
		translateMatrix = mat4::identity();
		modelViewMatrix = mat4::identity();
		modelViewProjectionMatrix = mat4::identity();

		translateMatrix = translate(0.0f, 0.0f, -4.0f);
		modelViewMatrix = modelViewMatrix * translateMatrix * scale(1.0f, 1.0f, 1.0f);

		modelViewProjectionMatrix = PerspectiveProjectionMatrix * modelViewMatrix;

		glUniformMatrix4fv(Fire_Terrain_mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);


		glUniform1i(Fire_choiceUniform, 1);
	
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, guiFireColor);
		glUniform1i(samplerFireColorUniform, 0);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, guiFireAlpha);
		glUniform1i(samplerFireAlphaUniform, 1);

		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, guiFireNoise);
		glUniform1i(samplerFireNoiseUniform, 2);

		glUniform3fv(Fire_octaveUniform, 1, Fire_gfOctave);
		glUniform3fv(Fire_speedUniform, 1, Fire_gfSpeed);
		glUniform1f(Fire_animationTimeUniform, Fire_gfAnimationTime);

		
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 

		glBindVertexArray(Fire_Vao_Rect);
			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		glBindVertexArray(0);


		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, 0);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, 0);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, 0);


		glDisable(GL_BLEND);

	glUseProgram(0);

	update_Fire();
}

