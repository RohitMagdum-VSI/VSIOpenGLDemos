//For Shader
GLuint gShaderProgramObject_Models;
GLuint gShaderProgramObject_GodRays;

//For Projection
mat4 godray_gPerspectiveProjectionMatrix;
mat4 godray_gOrthoProjectionMatrix;


//For Cube
GLuint texture_Cube;
GLuint kundali_Cube;
GLuint vao_Cube;
GLuint vbo_Cube_Position;
GLuint vbo_Cube_Texture;
GLfloat angle_Cube = 2.0f;

//For Sphere
GLuint vao_Sphere;
GLuint vbo_Sphere_Position;
GLuint vbo_Sphere_Normal;
GLuint vbo_Sphere_Texcoord;
GLuint vbo_Sphere_Element;


const int STACK = 30;
const int SLICES = 30;

float sphere_Position[STACK * SLICES * 3];
float sphere_Normal[STACK * SLICES * 3];
float sphere_Texcoord[STACK * SLICES * 2];
unsigned short sphere_Index[STACK * SLICES * 6];

unsigned int gNumOfElements_RRJ;
unsigned int gNumOfVertices_RRJ;

void  myMakeSphere(float, int, int);



//For Movement
GLfloat gfMoveFactor = 0.50f;

GLfloat lhX = -30.0f;
GLfloat lhY = 43.20f;
GLfloat lhZ = 0.0f;

GLfloat sX = 1.00f;
GLfloat sY = 1.00f;
GLfloat sZ = 1.00f;

GLfloat sphereX = 5.0f;
GLfloat sphereY = 5.0f;
GLfloat sphereZ = -16.0f;
GLfloat XOffset = 0.0f;
GLfloat YOffset = 0.0f;




//For Uniforms
GLuint godray_mvpUniform;
GLuint godray_samplerUniform;
GLuint godray_choiceUniform;


//For GodRays

//For Rectangle
GLuint godray_vao_Rect;
GLuint godray_vbo_Rect_Position;
GLuint godray_vbo_Rect_TexCoord;

GLuint godray_samplerFirstPassUniform;
GLuint mvpOrtho_Uniform_GodRays;
GLuint mvpPerspective_Uniform_GodRays;
GLuint viewPortUniform_GodRays;


GLuint godray_exposureUniform;
GLuint godray_decayUniform;
GLuint godray_densityUniform;
GLuint godray_weightUniform;
GLuint godray_lightPositionUniform;
GLuint godray_samplesUniform;

GLfloat godray_gfExposure = 0.0034f;
GLfloat godray_gfDecay = 1.0f;
GLfloat godray_gfDensity = 0.95f;
GLfloat godray_gfWeight = 5.65f;



//For Framebuffer
GLuint godray_frameBufferObject;
GLuint godray_renderBufferObject_Depth;	

#define SCALE_RATE 2

GLint godray_viewPortWidth = 1366;
GLint godray_viewPortHeight = 768;

GLint godray_viewPort_FBO_Width = godray_viewPortWidth / SCALE_RATE;
GLint godray_viewPort_FBO_Height = godray_viewPortHeight / SCALE_RATE;

// GLfloat godray_gfvLightPosition[2] = {godray_viewPort_FBO_Width, godray_viewPort_FBO_Height};
GLfloat godray_gfvLightPosition[4] = {0.0f, 0.0f, 0.0f, 1.0f};



//For Light House Effect
GLuint godray_movingLightOffsetUniform;
GLfloat godray_movingLightOffset[3] = {0.0f, 0.0f, 0.0f};


void initialize_GodRays_Final(void){

	void initialize_Objects();
	void initialize_GodRays();

	initialize_Objects();
	initialize_GodRays();	


}


void initialize_Objects(void){

	void uninitialize(void);
	void resize(int, int);
	

	//Shader Object;
	GLint iVertexShaderObject;
	GLint iFragmentShaderObject;


	/********** Vertex Shader **********/
	iVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *szVertexShaderSourceCode =
		"#version 450 core" \
		"\n" \

		//For Choice
		"uniform int u_choice;" \

		//For Attribute
		"in vec4 vPosition;" \
		"in vec2 vTexCoord;" \
		"out vec2 outTexCoord;" \

		//For Cube
		"uniform mat4 u_mvp_matrix;" \

		"void main(void)" \
		"{" \
			"gl_Position = u_mvp_matrix * vPosition;" \
			"outTexCoord = vTexCoord;" \

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
				fprintf(gpFile, "Vertex Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				exit(0);
			}
		}
	}

	/********** Fragment Shader **********/
	iFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *szFragmentShaderSourceCode =
		"#version 450 core" \
		"\n" \

		//For Choice
		"uniform int u_choice;" \

		//For Cube
		"in vec2 outTexCoord;" \
		"uniform sampler2D u_sampler;" \

		//For Output
		"out vec4 FragColor;" \

		"void main(void)" \
		"{" \


			"if(u_choice == 1){" \
				"FragColor = vec4(0.250f, 0.250f, 0.25f, 1.0f);" \
			"}" \

			"else if(u_choice == 2){" \
				"FragColor = texture(u_sampler, outTexCoord);" \
			"}" \

			"else if(u_choice == 3){" \
				"FragColor = vec4(vec3(0.50f), 1.0f) * texture(u_sampler, vec2(outTexCoord.x, 1.0f - outTexCoord.y));" \
			"}" \

			"else if(u_choice == 4){" \
				"FragColor = vec4(1.0f, 0.50f, 0.0f, 1.0f);" \
			"}" \

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
				fprintf(gpFile, "Fragment Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				exit(0);
			}
		}
	}

	/********** Shader Program Object **********/
	gShaderProgramObject_Models = glCreateProgram();

	glAttachShader(gShaderProgramObject_Models, iVertexShaderObject);
	glAttachShader(gShaderProgramObject_Models, iFragmentShaderObject);

	glBindAttribLocation(gShaderProgramObject_Models, ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(gShaderProgramObject_Models, ATTRIBUTE_TEXCOORD0, "vTexCoord");

	glLinkProgram(gShaderProgramObject_Models);

	GLint iProgramLinkingStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;
	glGetProgramiv(gShaderProgramObject_Models, GL_LINK_STATUS, &iProgramLinkingStatus);
	if (iProgramLinkingStatus == GL_FALSE) {
		glGetProgramiv(gShaderProgramObject_Models, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject_Models, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Shader Program Object Linking Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				exit(0);
			}
		}
	}

	



	//For Cube
	godray_mvpUniform = glGetUniformLocation(gShaderProgramObject_Models, "u_mvp_matrix");
	godray_samplerUniform = glGetUniformLocation(gShaderProgramObject_Models, "u_sampler");
	godray_choiceUniform = glGetUniformLocation(gShaderProgramObject_Models, "u_choice");



	/********** Positions **********/
	GLfloat Cube_Vertices[] = {
		//Top
		1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,
		//Bottom
		1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		//Front
		1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		//Back
		1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,
		//Right
		1.0f, 1.0f, -1.0f,
		1.0f, 1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, -1.0f,
		//Left
		-1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f
	};


	/************ TexCoord **********/
	GLfloat Cube_TexCoord[] = {
		//Top
		1.0f, 1.0,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		//Back
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		//Face
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		//Back
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		//Right
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		//Left
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};



	/********** Vao Cube **********/
	glGenVertexArrays(1, &vao_Cube);
	glBindVertexArray(vao_Cube);

		/******** Position **********/
		glGenBuffers(1, &vbo_Cube_Position);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Cube_Position);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(Cube_Vertices),
			Cube_Vertices,
			GL_STATIC_DRAW);

		glVertexAttribPointer(ATTRIBUTE_POSITION,
			3,
			GL_FLOAT,
			GL_FALSE,
			0, NULL);

		glEnableVertexAttribArray(ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		/********** Texture ***********/
		glGenBuffers(1, &vbo_Cube_Texture);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Cube_Texture);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(Cube_TexCoord),
			Cube_TexCoord,
			GL_STATIC_DRAW);

		glVertexAttribPointer(ATTRIBUTE_TEXCOORD0,
			2,
			GL_FLOAT,
			GL_FALSE,
			0, NULL);
		glEnableVertexAttribArray(ATTRIBUTE_TEXCOORD0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);




	/********** Position, Normal and Elements **********/
	myMakeSphere(1.0f, STACK, SLICES);


	/********** Sphere Vao **********/
	glGenVertexArrays(1, &vao_Sphere);
	glBindVertexArray(vao_Sphere);

		/********** Position **********/
		glGenBuffers(1, &vbo_Sphere_Position);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Sphere_Position);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(sphere_Position),
			sphere_Position,
			GL_STATIC_DRAW);

		glVertexAttribPointer(ATTRIBUTE_POSITION,
			3,
			GL_FLOAT,
			GL_FALSE,
			0, NULL);

		glEnableVertexAttribArray(ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		/********** Normals **********/
		glGenBuffers(1, &vbo_Sphere_Normal);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Sphere_Normal);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(sphere_Normal),
			sphere_Normal,
			GL_STATIC_DRAW);

		glVertexAttribPointer(ATTRIBUTE_NORMAL,
			3,
			GL_FLOAT,
			GL_FALSE,
			0, NULL);

		glEnableVertexAttribArray(ATTRIBUTE_NORMAL);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		/********** TexCoord **********/
		glGenBuffers(1, &vbo_Sphere_Texcoord);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Sphere_Texcoord);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(sphere_Texcoord),
			sphere_Texcoord,
			GL_STATIC_DRAW);

		glVertexAttribPointer(ATTRIBUTE_TEXCOORD0,
			2,
			GL_FLOAT,
			GL_FALSE,
			0, NULL);

		glEnableVertexAttribArray(ATTRIBUTE_TEXCOORD0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		/********** Element Vbo **********/
		glGenBuffers(1, &vbo_Sphere_Element);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(sphere_Index), sphere_Index, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


	glBindVertexArray(0);


	glEnable(GL_TEXTURE_2D);
	kundali_Cube = LoadTexture(MAKEINTRESOURCE(ID_BITMAP_KUNDALI), 0);



	
}


void initialize_GodRays(void){


	void uninitialize(void);
	void resize(int, int);
	
	//Shader Object;
	GLint iVertexShaderObject;
	GLint iFragmentShaderObject;


	/********** Vertex Shader **********/
	iVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *szVertexShaderSourceCode =
		"#version 450 core" \
		"\n" \

		//For Choice
		"uniform int u_choice;" \

		//For Attribute
		"in vec4 vPosition;" \
		"in vec2 vTexCoord;" \
		"out vec2 outTexCoord;" \
		"out vec4 outClipSpaceCoord;" \

		// For Screen Space Light Position;
		"uniform mat4 u_mvp_perspective;" \
		"uniform vec4 u_v4LightPosition;" \
		
		

		//For Cube
		"uniform mat4 u_mvp_ortho;" \



		"void main(void)" \
		"{" \
			"gl_Position = u_mvp_ortho * vPosition;" \
			// "outClipSpaceCoord = u_mvp_perspective * u_v4LightPosition;" \
			
			"outTexCoord = vTexCoord;" \

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
				fprintf(gpFile, "Vertex Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				exit(0);
			}
		}
	}

	/********** Fragment Shader **********/
	iFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *szFragmentShaderSourceCode =
		"#version 450 core" \
		"\n" \

		//For Choice
		"uniform int u_choice;" \

		//For Cube
		"in vec2 outTexCoord;" \
		"in vec4 outClipSpaceCoord;" \

		"uniform sampler2D u_sampler_firstpass;" \


		//For God Rays
		"uniform float u_fExposure;" \
		"uniform float u_fDecay;" \
		"uniform float u_fDensity;" \
		"uniform float u_fWeight;" \
		
		// "const int NUM_SAMPLES = 100;" \

		"uniform int u_NUM_SAMPLES;" \

		// For Screen Space Light Position;
		"uniform mat4 u_mvp_perspective;" \
		"uniform vec4 u_v4LightPosition;" \

		//For Light House Effect
		"uniform vec3 u_offset;" \
		


		//For Output
		"out vec4 FragColor;" \

		"void main(void)" \
		"{" \


			"vec2 lightPos = u_v4LightPosition.xy + u_offset.xy;" \
			
			"vec2 v2DeltaTexCoord = outTexCoord.xy - lightPos;" \
	
			"vec2 v2TexCoord = outTexCoord;" \
			"v2DeltaTexCoord *= 1.0f / float(u_NUM_SAMPLES) * u_fDensity;" \
			"float fIlluminationDecay = 1.0f;" \

			"for(int i = 0; i < u_NUM_SAMPLES; i++){" \
				"v2TexCoord = v2TexCoord - v2DeltaTexCoord;" \
				"vec4 v4Sample = texture2D(u_sampler_firstpass, v2TexCoord);" \

				"v4Sample = v4Sample * fIlluminationDecay * u_fWeight;" \

				"FragColor += v4Sample;" \

				"fIlluminationDecay *= u_fDecay;" \

			"}" \

			"FragColor *= u_fExposure;" \

			// "FragColor = vec4(out_v4LightPosition.xy, 0.0f, 1.0f);" \
		
			//"FragColor = texture2D(u_sampler_firstpass, outTexCoord);;" \

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
				fprintf(gpFile, "Fragment Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				exit(0);
			}
		}
	}

	/********** Shader Program Object **********/
	gShaderProgramObject_GodRays = glCreateProgram();

	glAttachShader(gShaderProgramObject_GodRays, iVertexShaderObject);
	glAttachShader(gShaderProgramObject_GodRays, iFragmentShaderObject);

	glBindAttribLocation(gShaderProgramObject_GodRays, ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(gShaderProgramObject_GodRays, ATTRIBUTE_TEXCOORD0, "vTexCoord");

	glLinkProgram(gShaderProgramObject_GodRays);

	GLint iProgramLinkingStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;
	glGetProgramiv(gShaderProgramObject_GodRays, GL_LINK_STATUS, &iProgramLinkingStatus);
	if (iProgramLinkingStatus == GL_FALSE) {
		glGetProgramiv(gShaderProgramObject_GodRays, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject_GodRays, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Shader Program Object Linking Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				exit(0);
			}
		}
	}


	mvpOrtho_Uniform_GodRays = glGetUniformLocation(gShaderProgramObject_GodRays, "u_mvp_ortho");
	mvpPerspective_Uniform_GodRays = glGetUniformLocation(gShaderProgramObject_GodRays, "u_mvp_perspective");
	godray_samplerFirstPassUniform = glGetUniformLocation(gShaderProgramObject_GodRays, "u_sampler_firstpass");
	godray_exposureUniform = glGetUniformLocation(gShaderProgramObject_GodRays, "u_fExposure");
	godray_decayUniform = glGetUniformLocation(gShaderProgramObject_GodRays, "u_fDecay");
	godray_densityUniform = glGetUniformLocation(gShaderProgramObject_GodRays, "u_fDensity");
	godray_weightUniform = glGetUniformLocation(gShaderProgramObject_GodRays, "u_fWeight");
	godray_lightPositionUniform = glGetUniformLocation(gShaderProgramObject_GodRays, "u_v4LightPosition");
	viewPortUniform_GodRays = glGetUniformLocation(gShaderProgramObject_GodRays, "u_v4ViewPort");

	godray_movingLightOffsetUniform = glGetUniformLocation(gShaderProgramObject_GodRays, "u_offset");
	godray_samplesUniform = glGetUniformLocation(gShaderProgramObject_GodRays, "u_NUM_SAMPLES");




	/********** Position and TexCoord **********/
	GLfloat Rect_Vertices[] = {
		1.0f, 1.0f, 0.0f,
		-1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
	};

	GLfloat Rect_TexCoord[] = {
		1.0f, 1.0f,
		0.0f, 1.0f, 
		0.0f, 0.0f,
		1.0f, 0.0f,
	};



	/********** Vao Rect On Which We Apply Texture **********/
	glGenVertexArrays(1, &godray_vao_Rect);
	glBindVertexArray(godray_vao_Rect);

		/********** Position **********/
		glGenBuffers(1, &godray_vbo_Rect_Position);
		glBindBuffer(GL_ARRAY_BUFFER, godray_vbo_Rect_Position);
		glBufferData(GL_ARRAY_BUFFER, 
				sizeof(Rect_Vertices),
				Rect_Vertices,
				GL_DYNAMIC_DRAW);

		glVertexAttribPointer(ATTRIBUTE_POSITION,
					3,
					GL_FLOAT,
					GL_FALSE,
					0, NULL);

		glEnableVertexAttribArray(ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		/********** Texture **********/
		glGenBuffers(1, &godray_vbo_Rect_TexCoord);
		glBindBuffer(GL_ARRAY_BUFFER, godray_vbo_Rect_TexCoord);
		glBufferData(GL_ARRAY_BUFFER,
				sizeof(Rect_TexCoord),
				Rect_TexCoord,
				GL_STATIC_DRAW);

		glVertexAttribPointer(ATTRIBUTE_TEXCOORD0,
					2,
					GL_FLOAT,
					GL_FALSE,
					0, NULL);

		glEnableVertexAttribArray(ATTRIBUTE_TEXCOORD0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);



	/********** FRAMEBUFFER **********/
	glGenFramebuffers(1, &godray_frameBufferObject);
	glBindFramebuffer(GL_FRAMEBUFFER, godray_frameBufferObject);

		/********** Texture **********/
		glGenTextures(1, &texture_Cube);
		glBindTexture(GL_TEXTURE_2D, texture_Cube);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, 
			GL_RGBA, 
			godray_viewPort_FBO_Width, godray_viewPort_FBO_Height, 0, 
			GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_Cube, 0);


		/********** For Depth **********/
		glGenRenderbuffers(1, &godray_renderBufferObject_Depth);
		glBindRenderbuffer(GL_RENDERBUFFER, godray_renderBufferObject_Depth);	
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, godray_viewPort_FBO_Width, godray_viewPort_FBO_Height);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);

		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, godray_renderBufferObject_Depth);




		/********** Checking *********/
		if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){
			fprintf(gpFile, "ERROR: glCheckFramebufferStatus\n");
			uninitialize();
			exit(0);
		}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

}



void uninitialize_Models(void){


	if(kundali_Cube){
		glDeleteTextures(1, &kundali_Cube);
		kundali_Cube = 0;
	}

	
	if (vbo_Sphere_Element) {
		glDeleteBuffers(1, &vbo_Sphere_Element);
		vbo_Sphere_Element = 0;
	}

	if(vbo_Sphere_Texcoord){
		glDeleteBuffers(1, &vbo_Sphere_Texcoord);
		vbo_Sphere_Texcoord = 0;
	}

	if (vbo_Sphere_Normal) {
		glDeleteBuffers(1, &vbo_Sphere_Normal);
		vbo_Sphere_Normal = 0;
	}

	if (vbo_Sphere_Position) {
		glDeleteBuffers(1, &vbo_Sphere_Position);
		vbo_Sphere_Position = 0;
	}

	if (vao_Sphere) {
		glDeleteVertexArrays(1, &vao_Sphere);
		vao_Sphere = 0;
	}




	if (vbo_Cube_Texture) {
		glDeleteBuffers(1, &vbo_Cube_Texture);
		vbo_Cube_Texture = 0;
	}

	if (vbo_Cube_Position) {
		glDeleteBuffers(1, &vbo_Cube_Position);
		vbo_Cube_Position = 0;
	}

	if (vao_Cube) {
		glDeleteVertexArrays(1, &vao_Cube);
		vao_Cube = 0;
	}


	GLsizei ShaderCount;
	GLsizei ShaderNumber;

	if (gShaderProgramObject_Models) {
		glUseProgram(gShaderProgramObject_Models);

		glGetProgramiv(gShaderProgramObject_Models, GL_ATTACHED_SHADERS, &ShaderCount);

		GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * ShaderCount);
		if (pShader) {
			glGetAttachedShaders(gShaderProgramObject_Models, ShaderCount, &ShaderCount, pShader);
			for (ShaderNumber = 0; ShaderNumber < ShaderCount; ShaderNumber++) {
				glDetachShader(gShaderProgramObject_Models, pShader[ShaderNumber]);
				glDeleteShader(pShader[ShaderNumber]);
				pShader[ShaderNumber] = 0;
			}
			free(pShader);
			pShader = NULL;
		}
		glUseProgram(0);
		glDeleteProgram(gShaderProgramObject_Models);
		gShaderProgramObject_Models = 0;

	}

}


void uninitialize_GodRays(void){


	if(texture_Cube){
		glDeleteTextures(1, &texture_Cube);
		texture_Cube = 0;
	}


	if(godray_renderBufferObject_Depth){
		glDeleteRenderbuffers(1, &godray_renderBufferObject_Depth);
		godray_renderBufferObject_Depth = 0;
	}

	if(godray_frameBufferObject){
		glDeleteFramebuffers(1, &godray_frameBufferObject);
		godray_frameBufferObject = 0;
	}



	if(godray_vbo_Rect_TexCoord){
		glDeleteBuffers(1, &godray_vbo_Rect_TexCoord);
		godray_vbo_Rect_TexCoord = 0;
	}

	if(godray_vbo_Rect_Position){
		glDeleteBuffers(1, &godray_vbo_Rect_Position);
		godray_vbo_Rect_Position = 0;
	}

	if(godray_vao_Rect){
		glDeleteVertexArrays(1, &godray_vao_Rect);
		godray_vao_Rect = 0;
	}

	GLsizei ShaderCount;
	GLsizei ShaderNumber;

	if (gShaderProgramObject_GodRays) {
		glUseProgram(gShaderProgramObject_GodRays);

		glGetProgramiv(gShaderProgramObject_GodRays, GL_ATTACHED_SHADERS, &ShaderCount);

		GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * ShaderCount);
		if (pShader) {
			glGetAttachedShaders(gShaderProgramObject_GodRays, ShaderCount, &ShaderCount, pShader);
			for (ShaderNumber = 0; ShaderNumber < ShaderCount; ShaderNumber++) {
				glDetachShader(gShaderProgramObject_GodRays, pShader[ShaderNumber]);
				glDeleteShader(pShader[ShaderNumber]);
				pShader[ShaderNumber] = 0;
			}
			free(pShader);
			pShader = NULL;
		}
		glUseProgram(0);
		glDeleteProgram(gShaderProgramObject_GodRays);
		gShaderProgramObject_GodRays = 0;

	}
}


void uninitialize_GodRays_Final(void){

	uninitialize_GodRays();
	uninitialize_Models();
}


void display_GodRays_Moon(void) {

	void update(void);

	mat4 translateMatrix;
	mat4 rotateMatrix;
	mat4 modelMatrix;
	mat4 viewMatrix;
	mat4 modelViewProjectionMatrix;
	mat4 lightPosition;

	static GLfloat angle_Model = 0.0f;


	//For Model As a Texture

	glViewport(0, 0, (GLsizei)godray_viewPort_FBO_Width, (GLsizei)godray_viewPort_FBO_Height);
	godray_gPerspectiveProjectionMatrix = mat4::identity();
	godray_gPerspectiveProjectionMatrix = perspective(45.0f, (float)godray_viewPort_FBO_Width / (GLfloat)godray_viewPort_FBO_Height, 0.1f, 100.0f);

	glBindFramebuffer(GL_FRAMEBUFFER, godray_frameBufferObject);
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		glUseProgram(gShaderProgramObject_Models);


		// Sphere
		translateMatrix = mat4::identity();
		modelMatrix = mat4::identity();
		viewMatrix = mat4::identity();
		modelViewProjectionMatrix = mat4::identity();

		translateMatrix = translate(sphereX, sphereY, sphereZ) * rotate(90.0f, 1.0f, 0.0f, 0.0f) * rotate(90.0f, 0.0f, 0.0f, 1.0f);
		

		modelMatrix = modelMatrix * translateMatrix;
		modelViewProjectionMatrix = godray_gPerspectiveProjectionMatrix * viewMatrix * modelMatrix;

		glUniformMatrix4fv(godray_mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);
		glUniform1i(godray_choiceUniform, 1);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, 0);
		glUniform1i(godray_samplerUniform, 0);


		glBindVertexArray(vao_Sphere);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element);
			glDrawElements(GL_TRIANGLES, STACK * SLICES * 6, GL_UNSIGNED_SHORT, 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glBindVertexArray(0);


		// // Sphere
		// translateMatrix = mat4::identity();
		// modelMatrix = mat4::identity();
		// viewMatrix = mat4::identity();
		// modelViewProjectionMatrix = mat4::identity();

		// translateMatrix = translate(sphereX - 0.30f, sphereY - 0.35f, sphereZ + 1.0f) * scale(0.7f, 0.7f, 0.7f);
		

		// modelMatrix = modelMatrix * translateMatrix;
		// modelViewProjectionMatrix = godray_gPerspectiveProjectionMatrix * viewMatrix * modelMatrix;

		// glUniformMatrix4fv(godray_mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);
		// glUniform1i(godray_choiceUniform, 2);

		// glActiveTexture(GL_TEXTURE0);
		// glBindTexture(GL_TEXTURE_2D, 0);
		// glUniform1i(godray_samplerUniform, 0);


		// glBindVertexArray(vao_Sphere);
		// 	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element);
		// 	glDrawElements(GL_TRIANGLES, STACK * SLICES * 6, GL_UNSIGNED_SHORT, 0);
		// 	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		// glBindVertexArray(0);

		// // Tree
		// translateMatrix = mat4::identity();
		// rotateMatrix = mat4::identity();
		// modelMatrix = mat4::identity();
		// viewMatrix = mat4::identity();
		// modelViewProjectionMatrix = mat4::identity();

		// translateMatrix = translate(cubeX, cubeY, cubeZ);
		// // rotateMatrix = rotate(angle_Cube, 1.0f, 0.0f, 0.0f) * rotate(angle_Cube, 0.0f, 1.0f, 0.0f) * rotate(angle_Cube, 0.0f, 0.0f, 1.0f);
		// modelMatrix = modelMatrix * translateMatrix * scale(0.2f, 0.2f, 0.2f) * rotateMatrix;
		// modelViewProjectionMatrix = godray_gPerspectiveProjectionMatrix * viewMatrix * modelMatrix;

		// glUniformMatrix4fv(godray_mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);
		// glUniform1i(godray_choiceUniform, 4);

		// display_Model(godray_gpModelTree);
		

		glUseProgram(0);
		

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	

	// //Draw Scene
	// glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	// glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(0, 0, (GLsizei)godray_viewPortWidth, (GLsizei)godray_viewPortHeight);
	godray_gPerspectiveProjectionMatrix = mat4::identity();
	godray_gPerspectiveProjectionMatrix = perspective(45.0f, (float)godray_viewPortWidth / (float)godray_viewPortHeight, 0.1f, 100.0f);
	godray_gOrthoProjectionMatrix = ortho(0, godray_viewPortWidth, 0, godray_viewPortHeight, -100.0f, 100.0f);
	glUseProgram(gShaderProgramObject_Models);


		// Sphere
		translateMatrix = mat4::identity();
		modelMatrix = mat4::identity();
		viewMatrix = mat4::identity();
		modelViewProjectionMatrix = mat4::identity();

		translateMatrix = translate(sphereX, sphereY, sphereZ) * rotate(90.0f, 1.0f, 0.0f, 0.0f) * rotate(90.0f, 0.0f, 0.0f, 1.0f);

		modelMatrix = modelMatrix * translateMatrix;
		modelViewProjectionMatrix = godray_gPerspectiveProjectionMatrix * viewMatrix * modelMatrix;

		glUniformMatrix4fv(godray_mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);
		glUniform1i(godray_choiceUniform, 3);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, kundali_Cube);
		glUniform1i(godray_samplerUniform, 0);

	
		glBindVertexArray(vao_Sphere);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element);
			glDrawElements(GL_TRIANGLES, STACK * SLICES * 6, GL_UNSIGNED_SHORT, 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glBindVertexArray(0);

		

		// // Tree
		// translateMatrix = mat4::identity();
		// rotateMatrix = mat4::identity();
		// modelMatrix = mat4::identity();
		// viewMatrix = mat4::identity();
		// modelViewProjectionMatrix = mat4::identity();

		// translateMatrix = translate(cubeX, cubeY, cubeZ);
		// // rotateMatrix = rotate(angle_Cube, 1.0f, 0.0f, 0.0f) * rotate(angle_Cube, 0.0f, 1.0f, 0.0f) * rotate(angle_Cube, 0.0f, 0.0f, 1.0f);
		// modelMatrix = modelMatrix * translateMatrix * scale(0.2f, 0.2f, 0.2f) * rotateMatrix;
		// modelViewProjectionMatrix = godray_gPerspectiveProjectionMatrix * viewMatrix * modelMatrix;

		// glUniformMatrix4fv(godray_mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);
		// glUniform1i(godray_choiceUniform, 4);
		// display_Model(godray_gpModelTree);
		

	glUseProgram(0);


	// godray_gfvLightPosition[0] = 0.0f;
	// godray_gfvLightPosition[1] = 0.0f;
	// godray_gfvLightPosition[2] = 0.0f;
	// godray_gfvLightPosition[3] = 1.0f;


	//For God Rays
	glViewport(0, 0, (GLsizei)godray_viewPortWidth, (GLsizei)godray_viewPortHeight);
	godray_gOrthoProjectionMatrix = mat4::identity();
	godray_gOrthoProjectionMatrix = ortho(
					-godray_viewPortWidth / 2.0f, godray_viewPortWidth / 2.0f,	// L, R
					-godray_viewPortHeight/ 2.0f, godray_viewPortHeight / 2.0f,	// B, T
					-1.0f, 1.0f);						// N, F

	godray_gPerspectiveProjectionMatrix = mat4::identity();
	godray_gPerspectiveProjectionMatrix = perspective(45.0f, (float)godray_viewPortWidth / (float)godray_viewPortHeight, 0.1f, 100.0f);




	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_DEPTH_BUFFER_BIT);

	glUseProgram(gShaderProgramObject_GodRays);


		// For Rectangle
		modelMatrix = mat4::identity();
		viewMatrix = mat4::identity();
		modelViewProjectionMatrix = mat4::identity();
		modelMatrix = modelMatrix;
		modelViewProjectionMatrix = godray_gOrthoProjectionMatrix * viewMatrix * modelMatrix;

		glUniformMatrix4fv(mvpOrtho_Uniform_GodRays, 1, GL_FALSE, modelViewProjectionMatrix);

		//For Light Source
		modelMatrix = mat4::identity();
		translateMatrix = mat4::identity();
		viewMatrix = mat4::identity();
		modelViewProjectionMatrix = mat4::identity();
		translateMatrix = translate(sphereX, sphereY, sphereZ);
		modelMatrix = modelMatrix * translateMatrix;
		modelViewProjectionMatrix = godray_gPerspectiveProjectionMatrix * viewMatrix * modelMatrix;


		vec4 data;
		data = my_gluProject(
				vec4(0.0f, 0.0f, 0.0f, 1.0f),
			 	viewMatrix * modelMatrix, 
			 	godray_gPerspectiveProjectionMatrix, 
				vec4(0.0f, 0.0f, 1366.0f, 768.0f)
			);


		godray_gfvLightPosition[0] = data[0] / 1366.0f;
		godray_gfvLightPosition[1] = data[1] /  768.0f ;

		// fprintf(gpFile, "Moon : %f, %f : %f, %f\n", 
		// 	godray_gfvLightPosition[0],
		// 	godray_gfvLightPosition[1],
		//  	data[0],
		//  	data[1]);


		glUniformMatrix4fv(mvpPerspective_Uniform_GodRays, 1, GL_FALSE, modelViewProjectionMatrix);


		
		glEnable(GL_TEXTURE_2D);                    // Enable 2D Texture Mapping
		glDisable(GL_DEPTH_TEST);                   // Disable Depth Testing
		glBlendFunc(GL_SRC_ALPHA,GL_ONE);           // Set Blending Mode
		glEnable(GL_BLEND);  

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture_Cube);
		glUniform1i(godray_samplerFirstPassUniform, 0);

		glUniform1f(godray_exposureUniform, godray_gfExposure);
		glUniform1f(godray_decayUniform, godray_gfDecay);
		glUniform1f(godray_densityUniform, godray_gfDensity);
		glUniform1f(godray_weightUniform, godray_gfWeight);
		glUniform1i(godray_samplesUniform, 100);

		

		godray_movingLightOffset[0] = 0.0f;
		godray_movingLightOffset[1] = 0.0f;

		glUniform4fv(godray_lightPositionUniform, 1, godray_gfvLightPosition);

		glUniform3fv(godray_movingLightOffsetUniform, 1, godray_movingLightOffset);
		

		GLfloat Rect_Vertices[] = {
			godray_viewPortWidth / 2.0f, godray_viewPortHeight / 2.0f, 0.0f,
			-godray_viewPortWidth / 2.0f, godray_viewPortHeight / 2.0f, 0.0f,
			-godray_viewPortWidth / 2.0f, -godray_viewPortHeight / 2.0f, 0.0f,
			godray_viewPortWidth / 2.0f, -godray_viewPortHeight / 2.0f, 0.0f,
		};

		glBindVertexArray(godray_vao_Rect);

			glBindBuffer(GL_ARRAY_BUFFER, godray_vbo_Rect_Position);
			glBufferData(GL_ARRAY_BUFFER, sizeof(Rect_Vertices), Rect_Vertices, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		glBindVertexArray(0);


		glEnable(GL_DEPTH_TEST);                    // Enable Depth Testing
		glDisable(GL_TEXTURE_2D);                   // Disable 2D Texture Mapping
		glDisable(GL_BLEND);                        // Disable Blending
		glBindTexture(GL_TEXTURE_2D,0); 

	
	glUseProgram(0);
}


void display_GodRays_LightHouse(void) {

	void update(void);

	mat4 translateMatrix;
	mat4 rotateMatrix;
	mat4 modelMatrix;
	mat4 viewMatrix;
	mat4 modelViewProjectionMatrix;
	mat4 lightPosition;

	static GLfloat angle_Model = 0.0f;

	update();

	
	//For Model As a Texture

	glViewport(0, 0, (GLsizei)godray_viewPort_FBO_Width, (GLsizei)godray_viewPort_FBO_Height);
	godray_gPerspectiveProjectionMatrix = mat4::identity();
	godray_gPerspectiveProjectionMatrix = perspective(45.0f, (float)godray_viewPort_FBO_Width / (GLfloat)godray_viewPort_FBO_Height, 0.1f, 100.0f);

	glBindFramebuffer(GL_FRAMEBUFFER, godray_frameBufferObject);
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		glUseProgram(gShaderProgramObject_Models);


		// Sphere
		translateMatrix = mat4::identity();
		modelMatrix = mat4::identity();
		viewMatrix = mat4::identity();
		modelViewProjectionMatrix = mat4::identity();

		translateMatrix = translate(lhX, lhY, lhZ) * scale(sX, sY, sZ);
		viewMatrix = lookat(c.cameraPosition, c.cameraPosition + c.cameraFront, c.cameraUp);

		modelMatrix = modelMatrix * translateMatrix;
		modelViewProjectionMatrix = godray_gPerspectiveProjectionMatrix * viewMatrix * modelMatrix;

		glUniformMatrix4fv(godray_mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);
		glUniform1i(godray_choiceUniform, 4);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, 0);
		glUniform1i(godray_samplerUniform, 0);


		glBindVertexArray(vao_Sphere);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element);
			glDrawElements(GL_TRIANGLES, STACK * SLICES * 6, GL_UNSIGNED_SHORT, 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glBindVertexArray(0);


	
		glUseProgram(0);
		

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	

	//Draw Scene
	// glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	// glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(0, 0, (GLsizei)godray_viewPortWidth, (GLsizei)godray_viewPortHeight);
	godray_gPerspectiveProjectionMatrix = mat4::identity();
	godray_gPerspectiveProjectionMatrix = perspective(45.0f, (float)godray_viewPortWidth / (float)godray_viewPortHeight, 0.1f, 100.0f);
	godray_gOrthoProjectionMatrix = ortho(0, godray_viewPortWidth, 0, godray_viewPortHeight, -100.0f, 100.0f);
	glUseProgram(gShaderProgramObject_Models);


		// Sphere
		translateMatrix = mat4::identity();
		modelMatrix = mat4::identity();
		viewMatrix = mat4::identity();
		modelViewProjectionMatrix = mat4::identity();

		translateMatrix = translate(lhX, lhY, lhZ) * scale(sX, sY, sZ);
		viewMatrix = lookat(c.cameraPosition, c.cameraPosition + c.cameraFront, c.cameraUp);
		

		modelMatrix = modelMatrix * translateMatrix;
		modelViewProjectionMatrix = godray_gPerspectiveProjectionMatrix * viewMatrix * modelMatrix;

		glUniformMatrix4fv(godray_mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);
		glUniform1i(godray_choiceUniform, 4);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, 0);
		glUniform1i(godray_samplerUniform, 0);


		glBindVertexArray(vao_Sphere);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element);
			glDrawElements(GL_TRIANGLES, STACK * SLICES * 6, GL_UNSIGNED_SHORT, 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glBindVertexArray(0);

		

	glUseProgram(0);



	//For God Rays
	glViewport(0, 0, (GLsizei)godray_viewPortWidth, (GLsizei)godray_viewPortHeight);
	godray_gOrthoProjectionMatrix = mat4::identity();
	godray_gOrthoProjectionMatrix = ortho(
					-godray_viewPortWidth / 2.0f, godray_viewPortWidth / 2.0f,	// L, R
					-godray_viewPortHeight/ 2.0f, godray_viewPortHeight / 2.0f,	// B, T
					-1.0f, 1.0f);						// N, F

	godray_gPerspectiveProjectionMatrix = mat4::identity();
	godray_gPerspectiveProjectionMatrix = perspective(45.0f, (float)godray_viewPortWidth / (float)godray_viewPortHeight, 0.1f, 100.0f);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_DEPTH_BUFFER_BIT);

	glUseProgram(gShaderProgramObject_GodRays);


		// For Rectangle
		modelMatrix = mat4::identity();
		viewMatrix = mat4::identity();
		modelViewProjectionMatrix = mat4::identity();
		modelMatrix = modelMatrix;
		modelViewProjectionMatrix = godray_gOrthoProjectionMatrix * viewMatrix * modelMatrix;

		glUniformMatrix4fv(mvpOrtho_Uniform_GodRays, 1, GL_FALSE, modelViewProjectionMatrix);

		//For Light Source
		modelMatrix = mat4::identity();
		translateMatrix = mat4::identity();
		viewMatrix = mat4::identity();
		modelViewProjectionMatrix = mat4::identity();
		
		translateMatrix = translate(lhX, lhY, lhZ) * scale(sX, sY, sZ);
		modelMatrix = modelMatrix * translateMatrix;
		viewMatrix = lookat(c.cameraPosition, c.cameraPosition + c.cameraFront, c.cameraUp);
		modelViewProjectionMatrix = godray_gPerspectiveProjectionMatrix * viewMatrix * modelMatrix;


		vec4 data;
		data = my_gluProject(
				vec4(0.0f, 0.0f, 0.0f, 1.0f),
			 	(viewMatrix * modelMatrix), 
			 	godray_gPerspectiveProjectionMatrix, 
				vec4(0.0f, 0.0f, 1366.0f, 768.0f)
			);


		godray_gfvLightPosition[0] = data[0] / 1366.0f;
		godray_gfvLightPosition[1] = data[1] /  768.0f ;

		// fprintf(gpFile, "LH : %f, %f : %f, %f\n", 
		// 	godray_gfvLightPosition[0],
		// 	godray_gfvLightPosition[1],
		//  	data[0],
		//  	data[1]);



		glUniformMatrix4fv(mvpPerspective_Uniform_GodRays, 1, GL_FALSE, modelViewProjectionMatrix);


		
		glEnable(GL_TEXTURE_2D);                    // Enable 2D Texture Mapping
		glDisable(GL_DEPTH_TEST);                   // Disable Depth Testing
		glBlendFunc(GL_SRC_ALPHA,GL_ONE);           // Set Blending Mode
		glEnable(GL_BLEND);  

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture_Cube);
		glUniform1i(godray_samplerFirstPassUniform, 0);

		glUniform1f(godray_exposureUniform, godray_gfExposure * 3.0f);
		glUniform1f(godray_decayUniform, godray_gfDecay);
		glUniform1f(godray_densityUniform, godray_gfDensity);
		glUniform1f(godray_weightUniform, godray_gfWeight);
		glUniform4fv(godray_lightPositionUniform, 1, godray_gfvLightPosition);

		glUniform1i(godray_samplesUniform, 100);


		glUniform3fv(godray_movingLightOffsetUniform, 1, godray_movingLightOffset);
		

		GLfloat Rect_Vertices[] = {
			godray_viewPortWidth / 2.0f, godray_viewPortHeight / 2.0f, 0.0f,
			-godray_viewPortWidth / 2.0f, godray_viewPortHeight / 2.0f, 0.0f,
			-godray_viewPortWidth / 2.0f, -godray_viewPortHeight / 2.0f, 0.0f,
			godray_viewPortWidth / 2.0f, -godray_viewPortHeight / 2.0f, 0.0f,
		};

		glBindVertexArray(godray_vao_Rect);

			glBindBuffer(GL_ARRAY_BUFFER, godray_vbo_Rect_Position);
			glBufferData(GL_ARRAY_BUFFER, sizeof(Rect_Vertices), Rect_Vertices, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		glBindVertexArray(0);


		glEnable(GL_DEPTH_TEST);                    // Enable Depth Testing
		glDisable(GL_TEXTURE_2D);                   // Disable 2D Texture Mapping
		glDisable(GL_BLEND);                        // Disable Blending
		glBindTexture(GL_TEXTURE_2D,0); 

	
	glUseProgram(0);

}



void update(void) {

	// static GLfloat factor = 0.005f;

	// angle_Cube += factor;

	// if(angle_Cube > 2.0f || angle_Cube < 1.0f)
	// 	factor = -factor;

	angle_Cube += 0.01f;

	// sphereX += 0.01f;

	// static GLint flag = 1;
	// if(flag == 1){

	// 	angle_Cube += 0.01f;
	// 	godray_movingLightOffset[0] = 0.2f * sin(angle_Cube);

	// 	if(godray_movingLightOffset[0] < 0.0f)
	// 		flag = 2;
	// }
	// else if(flag == 2){

	// 	angle_Cube -= 0.01f;
	// 	godray_movingLightOffset[0] = 0.2f * sin(angle_Cube);

	// 	// if(godray_movingLightOffset[0] < =0.0f)
	// 		// flag = 1;
	// }

	// static GLfloat val = -0.25f;
	// static GLint flag = 1;
	// const GLfloat end = 0.25f;

	// if(flag == 1){

	// 	val += 0.001f;
	// 	if(val > end)
	// 		flag = 2;

	// }
	// else if(flag == 2){
	// 	val -= 0.001f;
	// 	if(val < -end)
	// 		flag = 1;
	// }
	// godray_movingLightOffset[0] = val;
	

	godray_movingLightOffset[0] = 0.23f * cos(angle_Cube);

	
	// godray_movingLightOffset[1] = 0.3f * sin(120.0f);
	// godray_movingLightOffset[2] = 0.3f * sin(angle_Cube);

}




void myMakeSphere(float fRadius, int  iStack, int iSlices){

	float longitude;
	float latitude;
	float factorLat = (2.0 * PI) / (iStack);
	float factorLon = PI / (iSlices-1);

	float tx = 0.0f;
	float ty = 1.0f;
	float arcLength = (2.0f * PI * fRadius * (180.0f / 360.0f));

	float txFactor = 1.0f / (SLICES - 0);
	float tyFactor = 1.0f / (STACK - 0);

	fprintf(gpFile, "%f\n\n", arcLength);

	for(int i = 0; i < iStack; i++){
		
		latitude = -PI + i * factorLat;
		ty = 1.0f;

		for(int j = 0; j < iSlices; j++){

			longitude = (PI) - j * factorLon;

			//console.log(i + "/" + j + ": " + latitude + "/" + longitude);

			float x = fRadius * sin(longitude) * cos((latitude));
			float y = fRadius * sin(longitude) * sin((latitude));
			float z = fRadius * cos((longitude));

			sphere_Position[(i * iSlices * 3)+ (j * 3) + 0] = x;
			sphere_Position[(i * iSlices * 3)+ (j * 3) + 1] = y;
			sphere_Position[(i * iSlices * 3)+ (j * 3) + 2] = z;

			//zconsole.log(i + "/" + j + "   " + x + "/" + y + "/" + z);

			
			sphere_Normal[(i * iSlices * 3)+ (j * 3) + 0] = x;
			sphere_Normal[(i * iSlices * 3)+ (j * 3) + 1] = y;
			sphere_Normal[(i * iSlices * 3)+ (j * 3) + 2] = z;

			sphere_Texcoord[(i * iSlices * 2) + (j * 2) + 0] = tx;
			sphere_Texcoord[(i * iSlices * 2) + (j * 2) + 1] = ty;

			//fprintf(gpFile, "%f / %f\n", tx, ty);

			ty = ty - tyFactor;
		}

		tx = tx + txFactor;
	}


	int index = 0;
 	for(int i = 0; i < iStack ; i++){
 		for(int j = 0; j < iSlices ; j++){


 			if(i == (iStack - 1)){

 				unsigned short topLeft = (i * iSlices) + j;
	 			unsigned short bottomLeft = ((0) * iSlices) +(j);
	 			unsigned short topRight = topLeft + 1;
	 			unsigned short bottomRight = bottomLeft + 1;


	 			sphere_Index[index] = topLeft;
	 			sphere_Index[index + 1] = bottomLeft;
	 			sphere_Index[index + 2] = topRight;

	 			sphere_Index[index + 3] = topRight;
	 			sphere_Index[index + 4] = bottomLeft;
	 			sphere_Index[index + 5] = bottomRight;

 			}
 			else{

	 			unsigned short topLeft = (i * iSlices) + j;
	 			unsigned short bottomLeft = ((i + 1) * iSlices) +(j);
	 			unsigned short topRight = topLeft + 1;
	 			unsigned short bottomRight = bottomLeft + 1;


	 			sphere_Index[index] = topLeft;
	 			sphere_Index[index + 1] = bottomLeft;
	 			sphere_Index[index + 2] = topRight;

	 			sphere_Index[index + 3] = topRight;
	 			sphere_Index[index + 4] = bottomLeft;
	 			sphere_Index[index + 5] = bottomRight;
 			}

 			

 			index = index + 6;


 		}
 		

 	}
}
