


//For Shader Program Object;
GLint font_gShaderProgramObject;

//For Perspective Matric
mat4 font_gPerspectiveProjectionMatrix;

//For Smiley
GLuint font_vao_Rect;
GLuint font_vbo_Rect_Position;
GLuint font_vbo_Rect_TexCoord;

//For Texture
GLuint font_texture_Smiley;

//For Uniform
GLuint font_mvpUniform; 
GLuint font_samplerUniform;


struct Characters{
	GLuint textureId;
	vec2 size;
	vec2 bearing;
	unsigned int advance;
};


struct Characters TotalChar[128];

FT_Library gFt;
FT_Face gFace;

GLuint fontColorUniform;



void initialize_Font(void) {

	
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

		"out vec2 outTex;" \

		"uniform mat4 u_mvp_matrix;" \
		"void main(void)" \
		"{" \

			"outTex = vPosition.zw;" \
			"gl_Position = u_mvp_matrix * vec4(vPosition.xy, 0.0f, 1.0f);" \
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
				fprintf(gpFile, "Font : Vertex Shader Compilation Error: %s\n", szInfoLog);
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
		"in vec2 outTex;" \
		"out vec4 FragColor;" \
		
		"uniform sampler2D u_sampler;" \
		"uniform vec3 u_texColor;" \

		"void main(void)" \
		"{" \

			"vec4 sampled = vec4(1.0f, 1.0f, 1.0f, texture(u_sampler, outTex));" \
			"FragColor = vec4(u_texColor, 1.0f) * sampled;" \
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
				fprintf(gpFile, "Font : Fragment Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				DestroyWindow(ghwnd);
				exit(0);
			}
		}
	}

	/********** Shader Program Object **********/
	font_gShaderProgramObject = glCreateProgram();

	glAttachShader(font_gShaderProgramObject, iVertexShaderObject);
	glAttachShader(font_gShaderProgramObject, iFragmentShaderObject);

	glBindAttribLocation(font_gShaderProgramObject, ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(font_gShaderProgramObject, ATTRIBUTE_TEXCOORD0, "vTexCoord");

	glLinkProgram(font_gShaderProgramObject);

	GLint iProgramLinkingStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;
	glGetProgramiv(font_gShaderProgramObject, GL_LINK_STATUS, &iProgramLinkingStatus);
	if (iProgramLinkingStatus == GL_FALSE) {
		glGetProgramiv(font_gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetProgramInfoLog(font_gShaderProgramObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Font : Shader Program Object Linking Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				DestroyWindow(ghwnd);
				exit(0);
			}
		}
	}

	font_mvpUniform = glGetUniformLocation(font_gShaderProgramObject, "u_mvp_matrix");
	font_samplerUniform = glGetUniformLocation(font_gShaderProgramObject, "u_sampler");
	fontColorUniform = glGetUniformLocation(font_gShaderProgramObject, "u_texColor");



	loadAllFonts();


	/********** Vao Rect On Which We Apply Texture **********/
	glGenVertexArrays(1, &font_vao_Rect);
	glBindVertexArray(font_vao_Rect);

	/********** Position **********/
	glGenBuffers(1, &font_vbo_Rect_Position);
	glBindBuffer(GL_ARRAY_BUFFER, font_vbo_Rect_Position);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(GLfloat) * 6 * 4,
		NULL,
		GL_DYNAMIC_DRAW);

	glVertexAttribPointer(ATTRIBUTE_POSITION,
		4,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);


	fprintf(gpFile, "Font : initialize_Font() done\n");
}



void loadAllFonts(void){

	void uninitialize(void);


	if(FT_Init_FreeType(&gFt)){
		fprintf(gpFile, "loadAllFonts: FT_Init_FreeType() Failed\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}


	if(FT_New_Face(gFt, "C:\\Windows\\Fonts\\Arial.ttf", 0, &gFace)){
		fprintf(gpFile, "loadAllFonts: FT_New_Face() failed\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}
	else{


		FT_Set_Pixel_Sizes(gFace, 0, 48);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		for(unsigned char i = 0; i < 128; i++){

			if(FT_Load_Char(gFace, i, FT_LOAD_RENDER)){
				fprintf(gpFile, "loadAllFonts: FT_Load_Face() failed\n");
				uninitialize();
				DestroyWindow(ghwnd);
			}

			GLuint texture;

			glGenTextures(1, &texture);
			glBindTexture(GL_TEXTURE_2D, texture);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

			glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 
				gFace->glyph->bitmap.width, gFace->glyph->bitmap.rows,
				0,
				GL_RED,
				GL_UNSIGNED_BYTE, gFace->glyph->bitmap.buffer);

			glBindTexture(GL_TEXTURE_2D, 0);



			TotalChar[i].textureId = texture;
			TotalChar[i].size = vec2(gFace->glyph->bitmap.width, gFace->glyph->bitmap.rows);
			TotalChar[i].bearing = vec2(gFace->glyph->bitmap_left, gFace->glyph->bitmap_top);
			TotalChar[i].advance = (unsigned int)gFace->glyph->advance.x;
		}

	}

	FT_Done_Face(gFace);
	FT_Done_FreeType(gFt);
}


void uninitialize_Font(void) {


	for(int i = 0; i < 128; i++){
		if(TotalChar[i].textureId){
			glDeleteTextures(1, &TotalChar[i].textureId);
			TotalChar[i].textureId = 0;
		}
	}



	if (font_vbo_Rect_Position) {
		glDeleteBuffers(1, &font_vbo_Rect_Position);
		font_vbo_Rect_Position = 0;
	}

	if (font_vao_Rect) {
		glDeleteVertexArrays(1, &font_vao_Rect);
		font_vao_Rect = 0;
	}

	GLsizei ShaderCount;
	GLsizei ShaderNumber;

	if (font_gShaderProgramObject) {
		glUseProgram(font_gShaderProgramObject);

		glGetProgramiv(font_gShaderProgramObject, GL_ATTACHED_SHADERS, &ShaderCount);

		GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * ShaderCount);
		if (pShader) {
			glGetAttachedShaders(font_gShaderProgramObject, ShaderCount,
				&ShaderCount, pShader);
			for (ShaderNumber = 0; ShaderNumber < ShaderCount; ShaderNumber++) {
				glDetachShader(font_gShaderProgramObject, pShader[ShaderNumber]);
				glDeleteShader(pShader[ShaderNumber]);
				pShader[ShaderNumber] = 0;
			}
			free(pShader);
			pShader = NULL;
		}
		glDeleteProgram(font_gShaderProgramObject);
		font_gShaderProgramObject = 0;
		glUseProgram(0);
	}

	fprintf(gpFile, "Font : uninitialize_Font() done\n");

	
}



void display_Font(char *arr, float x, float y, float scale, vec3 color) {

	void RenderText(char*, float, float, float, vec3);

	mat4 translateMatrix;
	mat4 modelViewMatrix;
	mat4 modelViewProjectionMatrix;

	// glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(font_gShaderProgramObject);

	/********** Rectangle With Texture **********/
	translateMatrix = mat4::identity();
	modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();

	translateMatrix = translate(0.0f, 0.0f, -100.0f);
	modelViewMatrix = modelViewMatrix * translateMatrix;
	font_gPerspectiveProjectionMatrix = perspective(45.0f, (GLfloat)1366 / (GLfloat)768, 0.1f, 1000.0f);
	modelViewProjectionMatrix = font_gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(font_mvpUniform,
		1,
		GL_FALSE,
		modelViewProjectionMatrix);

	
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
	RenderText(arr, x, y, scale, color);
	// RenderText("Compute Group", -20.0f, 0.0f, 0.120f, vec3(1.0f, 1.0f, 0.0f));
	// RenderText("Presents...", 0.0f, -10.0f, 0.10f, vec3(1.0f, 1.0f, 0.0f));
	glDisable(GL_BLEND);

	//RenderText("N-Body Simulation", -30.0f, 10.0f, 0.15f, vec3(1.0f, 1.0f, 0.0f));

	glUseProgram(0);
}


void RenderText(char *arr, float x, float y, float scale, vec3 color){


	int len = strlen(arr);

	glBindVertexArray(font_vao_Rect);

	for(int i = 0; i < len; i++){

		struct Characters c = TotalChar[arr[i]];

		float xpos = x + c.bearing[0] * scale;
		float ypos = y - (c.size[1] - c.bearing[1]) * scale;

		float w = c.size[0] * scale;
		float h = c.size[1] * scale;


		float vertices[6][4] = {
			{xpos, ypos + h, 0.0f, 0.0f},
			{xpos, ypos, 0.0f, 1.0f},
			{xpos + w, ypos, 1.0f, 1.0f},

			{xpos, ypos + h, 0.0f, 0.0f},
			{xpos + w, ypos, 1.0f, 1.0f},
			{xpos + w, ypos + h, 1.0f, 0.0f},
		};


		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, c.textureId);
		glUniform1f(font_samplerUniform, 0);

		glUniform3fv(fontColorUniform, 1, vec3(color[0], color[1], color[2]));


		glBindBuffer(GL_ARRAY_BUFFER, font_vbo_Rect_Position);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glDrawArrays(GL_TRIANGLE_FAN, 0, 6);

		x = x + (c.advance >> 6) * scale;
	}

	glBindVertexArray(0);

}
