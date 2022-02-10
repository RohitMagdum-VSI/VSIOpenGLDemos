// ************************************
// TESSELATION SHADER TERRAIN
// ************************************

//For Shader
GLuint Terrain_gShaderProgramObject;
GLuint Terrain_gVertexShaderObject;
GLuint Terrain_gFragmentShaderObject;


//For Grid
GLuint vao_Grid;
GLuint vbo_Grid_Position;
GLuint vbo_Grid_Texcoord;
GLuint vbo_Grid_Normals;
GLuint vbo_Grid_Index;

const GLfloat PI = 3.1415926535;

const GLuint GRID_WIDTH = 128;
const GLuint GRID_HEIGHT = 128;


GLfloat gGrid_StartX = -8.0;
GLfloat gGrid_StartZ = -8.0;
GLint gGrid_NumOfElements = 0;

GLfloat grid_Position[GRID_WIDTH * GRID_HEIGHT * 3];
GLfloat grid_Texcoord[GRID_WIDTH * GRID_HEIGHT * 2];;
GLfloat grid_Normals[GRID_WIDTH * GRID_HEIGHT * 4];
GLuint grid_Index[(GRID_WIDTH - 1) * (GRID_HEIGHT - 1) * 6];

GLuint Terrain_modelMatUniform;
GLuint Terrain_viewMatUniform;
GLuint Terrain_projMatUniform;

GLuint Terrain_textureHtMap;
GLuint Terrain_samplerHtMapUniform;

GLuint Terrain_texture;
GLuint Terrain_samplerTexUniform;

//For Choice
GLuint Terrain_choiceUniform;
const GLuint TERR_NORMAL = 0;
const GLuint TERR_SINGLE_COLOR = 1;

//For Transform Feedback
GLuint vbo_temp_position;
GLfloat *grid_Transformed_Pos = NULL;


//For Single Light Uniform
GLuint la_Uniform;
GLuint ld_Uniform;
GLuint ls_Uniform;
GLuint lightPosition_Uniform;

GLuint ka_Uniform;
GLuint kd_Uniform;
GLuint ks_Uniform;
GLuint shininess_Uniform;

//For Lights
GLfloat lightAmbient[] = {0.1, 0.1, 0.1};
GLfloat lightDiffuse[] ={1.0, 1.0, 1.0};
GLfloat lightSpecular[] = {1.0, 1.0, 1.0};
GLfloat lightPosition[] = {300.0, 400.0, 300.0, 1.0};
bool bLights = false;


//For Material
GLfloat materialAmbient[] = {0.0, 0.0, 0.0};
GLfloat materialDiffuse[] = {1.0, 1.0, 1.0};
GLfloat materialSpecular[] = {1.0, 1.0, 1.0};
GLfloat materialShininess = 50.0;





void initialize_Terrain(void){

	void uninitialize(void);
	int LoadKTXTexture(const char*, GLuint*);

	/********** VERTEX SHADER **********/
	Terrain_gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
	const char *vertexShaderSourceCode =
		"#version 450" \
		"\n" \

		"in vec3 vPosition;" \
		"in vec4 vNormal;" \
		"in vec2 vTex;" \

		"uniform int u_choice;" \
		"const int NORMAL = 0;" \
		"const int SINGLE_COLOR = 1;" \


		//Common in Lights
		"out vec3 outViewer;" \
		"out vec3 outNormal;" \

		//For Single Light
		"uniform vec4 u_light_position;" \
		"out vec3 outLightDirection;" \


		"uniform mat4 u_model_mat;" \
		"uniform mat4 u_view_mat;" \
		"uniform mat4 u_proj_mat;" \
 
 		"uniform sampler2D u_samplerHtMap;" \

		"out vec2 outTex;" \
 		
 		//For Capturing Pos for Normal Calculation
 		"out vec4 out_pos_normal;" \

		"void main(void) {" \

			"vec4 newPos = vec4(vPosition, 1.0f);" \

			"vec4 tex = texture(u_samplerHtMap, vTex);" \

			"float ht = (tex.x + tex.y + tex.z) / 3.0f;" \

			// "float diff = length(vTex - vec2(0.5f));" \

			// "if(diff < 0.3)"
			// 	"newPos.y = ht * 25.50f;" \
			// "else{" \

			// 	"newPos.y = ht * .5f;" \
			// "}" \
 			
 			"newPos.y = ht * 25.5f;" \

			"outTex = vTex;" \

		
			"if(u_choice == NORMAL){" \
				
				// Common For Lights
				"vec4 worldCoord = u_model_mat * newPos;" \
				"vec4 eyeCoordinate = u_view_mat * worldCoord;" \
				
				"mat3 normalMatrix = mat3(u_view_mat * u_model_mat);" \
				"outNormal = vec3(normalMatrix * vec3(vNormal));" \
				"outViewer = vec3(-eyeCoordinate);" \

				//For Single Light
				"outLightDirection = vec3(u_light_position - eyeCoordinate);" \
			
			"}" \

			"else if(u_choice == SINGLE_COLOR){" \

				//For Transform Feedback
				"out_pos_normal = vec4(newPos.xyz, 0.82f);" \
			
			"}" \

			"gl_Position = u_proj_mat * u_view_mat * u_model_mat * newPos;" \
		"}";

	glShaderSource(Terrain_gVertexShaderObject, 1, (const char**)&vertexShaderSourceCode, NULL);
	glCompileShader(Terrain_gVertexShaderObject);

	int iInfoLogLength;
	int iShaderCompileStatus;
	char *szInfoLog = NULL;

	glGetShaderiv(Terrain_gVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE) {
		glGetShaderiv(Terrain_gVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (char*)malloc(sizeof(char) * iInfoLogLength);
			if (szInfoLog) {
				GLsizei written;

				glGetShaderInfoLog(Terrain_gVertexShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "VERTEX SHADER ERROR: \n %s", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				DestroyWindow(ghwnd);
			}
		}
	}




	/********** FRAGMENT SHADER **********/
	Terrain_gFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);
	const char *fragmentShaderSourceCode =
		"#version 450" \
		"\n" \
		
		"in vec2 outTex;" \

		"uniform int u_choice;" \
		"const int NORMAL = 0;" \
		"const int SINGLE_COLOR = 1;" \

		 //Common in both lights
		"in vec3 outNormal;" \
		"in vec3 outViewer;" \

		//For Single Lights
		"in vec3 outLightDirection;" \

		"uniform vec3 u_la;" \
		"uniform vec3 u_ld;" \
		"uniform vec3 u_ls;" \

		
		"uniform vec3 u_ka;" \
		"uniform vec3 u_kd;" \
		"uniform vec3 u_ks;" \
		"uniform float u_shininess;" \



		"uniform sampler2D u_samplerTex;" \
		"uniform sampler2D u_samplerHtMap;" \

		"out vec4 FragColor;" \


		"void main(void) {" \

			"vec4 tex =  texture(u_samplerTex, outTex * 1.0f);" \
			"vec3 PhongLight;" \
			"vec4 col;" \

			"if(u_choice == NORMAL){" \
				
				// //For Single Light
				// "vec3 normalizeLightDirection = normalize(outLightDirection);" \
				// "vec3 normalizeNormalVector = normalize(outNormal);" \
				// "float S_Dot_N = max(dot(normalizeLightDirection, normalizeNormalVector), 0.0);" \

				// "vec3 normalizeViewer = normalize(outViewer);" \
				// "vec3 reflection = reflect(-normalizeLightDirection, normalizeNormalVector);" \
				// "float R_Dot_V = max(dot(reflection, normalizeViewer), 0.0);" \

				// "vec3 ambient = u_la * u_ka;" \
				// "vec3 diffuse =  u_ld * u_kd * S_Dot_N;" \

				// // "vec3 specular = u_ls * u_ks * pow(R_Dot_V, u_shininess);" \
	
				// "PhongLight = ambient + diffuse;" \

				"col = tex * vec4(0.5f);" \

			"}" \

			"else if(u_choice == SINGLE_COLOR){" \
				"col = vec4(1.0, 1.0, 1.0, 1.0);" \
			"}" \

			"FragColor = col;" \
		
		"}";


	glShaderSource(Terrain_gFragmentShaderObject, 1, 
		(const char**)&fragmentShaderSourceCode, NULL);

	glCompileShader(Terrain_gFragmentShaderObject);

	iInfoLogLength = 0;
	iShaderCompileStatus = 0;
	szInfoLog = NULL;
	
	glGetShaderiv(Terrain_gFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE) {
		glGetShaderiv(Terrain_gFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (char*)malloc(sizeof(char) * iInfoLogLength);
			if (szInfoLog) {
				GLsizei written;
				glGetShaderInfoLog(Terrain_gFragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "FRAGMENT SHADER ERROR: \n %s", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				DestroyWindow(ghwnd);
			}
		}
	}


	/********** SHADER PROGRAM **********/
	Terrain_gShaderProgramObject = glCreateProgram();

	glAttachShader(Terrain_gShaderProgramObject, Terrain_gVertexShaderObject);
	glAttachShader(Terrain_gShaderProgramObject, Terrain_gFragmentShaderObject);

	glBindAttribLocation(Terrain_gShaderProgramObject, ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(Terrain_gShaderProgramObject, ATTRIBUTE_NORMAL, "vNormal");
	glBindAttribLocation(Terrain_gShaderProgramObject, ATTRIBUTE_TEXCOORD0, "vTex");

	glLinkProgram(Terrain_gShaderProgramObject);

	int iProgramLinkStatus;
	iInfoLogLength = 0;
	szInfoLog = NULL;
	glGetProgramiv(Terrain_gShaderProgramObject, GL_LINK_STATUS, &iProgramLinkStatus);
	if (iProgramLinkStatus == GL_FALSE) {
		glGetProgramiv(Terrain_gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (char*)malloc(sizeof(char) * iInfoLogLength);
			if (szInfoLog) {
				GLsizei written;
				glGetProgramInfoLog(Terrain_gShaderProgramObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "SHADER PROGRAM ERROR: %s", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				DestroyWindow(ghwnd);
			}
		}
	}


	// ********** Transform Feedback **********
	const GLchar *varying = {"out_pos_normal"};

	glTransformFeedbackVaryings(Terrain_gShaderProgramObject, 1, &varying, GL_INTERLEAVED_ATTRIBS);

	glLinkProgram(Terrain_gShaderProgramObject);

	iProgramLinkStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;
	glGetProgramiv(Terrain_gShaderProgramObject, GL_LINK_STATUS, &iProgramLinkStatus);
	if (iProgramLinkStatus == GL_FALSE) {
		glGetProgramiv(Terrain_gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (char*)malloc(sizeof(char) * iInfoLogLength);
			if (szInfoLog) {
				GLsizei written;
				glGetProgramInfoLog(Terrain_gShaderProgramObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "SHADER PROGRAM ERROR TRANSFORMED FEEDBACK: %s", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				DestroyWindow(ghwnd);
			}
		}
	}


	Terrain_modelMatUniform = glGetUniformLocation(Terrain_gShaderProgramObject, "u_model_mat");
	Terrain_viewMatUniform = glGetUniformLocation(Terrain_gShaderProgramObject, "u_view_mat");
	Terrain_projMatUniform = glGetUniformLocation(Terrain_gShaderProgramObject, "u_proj_mat");

	Terrain_choiceUniform = glGetUniformLocation(Terrain_gShaderProgramObject, "u_choice");

	Terrain_samplerHtMapUniform = glGetUniformLocation(Terrain_gShaderProgramObject, "u_samplerHtMap");
	Terrain_samplerTexUniform = glGetUniformLocation(Terrain_gShaderProgramObject, "u_samplerTex");

	//For Single Global Light
	la_Uniform = glGetUniformLocation(Terrain_gShaderProgramObject, "u_la");
	ld_Uniform = glGetUniformLocation(Terrain_gShaderProgramObject, "u_ld");
	ls_Uniform = glGetUniformLocation(Terrain_gShaderProgramObject, "u_ls");
	lightPosition_Uniform = glGetUniformLocation(Terrain_gShaderProgramObject, "u_light_position");

	ka_Uniform = glGetUniformLocation(Terrain_gShaderProgramObject, "u_ka");
	kd_Uniform = glGetUniformLocation(Terrain_gShaderProgramObject, "u_kd");
	ks_Uniform = glGetUniformLocation(Terrain_gShaderProgramObject, "u_ks");
	shininess_Uniform = glGetUniformLocation(Terrain_gShaderProgramObject, "u_shininess");


	/********** Grid COORDINATES **********/
	void LoadGrid(void);

	LoadGrid();

	Terrain_textureHtMap = LoadTexture(MAKEINTRESOURCE(ID_TERRAIN_HT), 0);

	LoadKTXTexture("02-Terrain/terragen_color.ktx", &Terrain_texture);

	// Terrain_texture = LoadTexture(MAKEINTRESOURCE(ID_TERRAIN_TEX), 0);
	


	glGenVertexArrays(1, &vao_Grid);
	glBindVertexArray(vao_Grid);

		/********** Position **********/
		glGenBuffers(1, &vbo_Grid_Position);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Grid_Position);
		glBufferData(GL_ARRAY_BUFFER, sizeof(grid_Position), grid_Position, GL_STATIC_DRAW);
		glVertexAttribPointer(ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		/********** Normals **********/
		glGenBuffers(1, &vbo_Grid_Normals);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Grid_Normals);
		glBufferData(GL_ARRAY_BUFFER, sizeof(grid_Normals), NULL, GL_DYNAMIC_COPY);
		glVertexAttribPointer(ATTRIBUTE_NORMAL, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(ATTRIBUTE_NORMAL);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		/********** Texcoord **********/
		glGenBuffers(1, &vbo_Grid_Texcoord);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Grid_Texcoord);
		glBufferData(GL_ARRAY_BUFFER, sizeof(grid_Texcoord), grid_Texcoord, GL_STATIC_DRAW);
		glVertexAttribPointer(ATTRIBUTE_TEXCOORD0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(ATTRIBUTE_TEXCOORD0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		/********** Index **********/
		glGenBuffers(1, &vbo_Grid_Index);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Grid_Index);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(grid_Index), grid_Index, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


	glBindVertexArray(0);
}


void uninitialize_Terrain(void){

	

	if(Terrain_textureHtMap){
		glDeleteTextures(1, &Terrain_textureHtMap);
		Terrain_textureHtMap = 0;
	}

	if (vbo_Grid_Index) {
		glDeleteBuffers(1, &vbo_Grid_Index);
		vbo_Grid_Index = 0;
	}

	if (vbo_Grid_Texcoord) {
		glDeleteBuffers(1, &vbo_Grid_Texcoord);
		vbo_Grid_Texcoord = 0;
	}

	if (vbo_Grid_Normals) {
		glDeleteBuffers(1, &vbo_Grid_Normals);
		vbo_Grid_Normals = 0;
	}

	if (vbo_Grid_Position) {
		glDeleteBuffers(1, &vbo_Grid_Position);
		vbo_Grid_Position = 0;
	}

	if (vao_Grid) {
		glDeleteVertexArrays(1, &vao_Grid);
		vao_Grid = 0;
	}

	if (Terrain_gShaderProgramObject) {

		glUseProgram(Terrain_gShaderProgramObject);
			
		GLint shaderCount;
		GLint shaderNo;

		glGetShaderiv(Terrain_gShaderProgramObject, GL_ATTACHED_SHADERS, &shaderCount);
		fprintf(gpFile, "INFO: ShaderCount: %d\n", shaderCount);
		GLuint *pShaders = (GLuint*)malloc(sizeof(GLuint*) * shaderCount);
		if (pShaders) {
			glGetAttachedShaders(Terrain_gShaderProgramObject, shaderCount, &shaderCount, pShaders);
			for (shaderNo = 0; shaderNo < shaderCount; shaderNo++) {
				glDetachShader(Terrain_gShaderProgramObject, pShaders[shaderNo]);
				glDeleteShader(pShaders[shaderNo]);
				pShaders[shaderNo] = 0;

				fprintf(gpFile, "Terrain: Shader %d Detached and Deleted\n", shaderNo+1);
			}
			free(pShaders);
			pShaders = NULL;
		}


		fprintf(gpFile, "Terrain: All Shader Detached and Deleted\n");

		glUseProgram(0);
		glDeleteProgram(Terrain_gShaderProgramObject);
		Terrain_gShaderProgramObject = 0;
	}

}



void display_Terrain(void){

	void display_GodRays_LightHouse(void);

	mat4 modelMat;
	mat4 viewMat;


	glUseProgram(Terrain_gShaderProgramObject);

	modelMat = mat4::identity();
	viewMat = mat4::identity();

	// modelMat = translate(0.0f, -2.00f, -5.0f) * rotate(10.0f, 0.0f, 0.0f);
	modelMat = scale(10.0f, 1.0f, 10.0f);
	viewMat = lookat(c.cameraPosition, c.cameraPosition + c.cameraFront, c.cameraUp);

	glUniformMatrix4fv(Terrain_modelMatUniform, 1, GL_FALSE, modelMat);
	glUniformMatrix4fv(Terrain_viewMatUniform, 1, GL_FALSE, viewMat);
	glUniformMatrix4fv(Terrain_projMatUniform, 1, GL_FALSE, PerspectiveProjectionMatrix);

	//For Ht Map
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, Terrain_textureHtMap);
	glUniform1i(Terrain_samplerHtMapUniform, 0);

	//For Texture
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, Terrain_texture);
	glUniform1i(Terrain_samplerTexUniform, 1);

	// For Single Light
	glUniform3fv(la_Uniform, 1, lightAmbient);
	glUniform3fv(ld_Uniform, 1, lightDiffuse);
	glUniform3fv(ls_Uniform, 1, lightSpecular);
	glUniform4fv(lightPosition_Uniform, 1, lightPosition);

	glUniform3fv(ka_Uniform, 1, materialAmbient);
	glUniform3fv(kd_Uniform, 1, materialDiffuse);
	glUniform3fv(ks_Uniform, 1, materialSpecular);
	glUniform1f(shininess_Uniform, materialShininess);

	glUniform1i(Terrain_choiceUniform, TERR_NORMAL);	

	glBindVertexArray(vao_Grid);

		//glDrawArrays(GL_POINTS, 0, GRID_WIDTH * 5);

		/********** Normals **********/
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Grid_Normals);
		glBufferData(GL_ARRAY_BUFFER, sizeof(grid_Normals), grid_Normals, GL_STATIC_DRAW);
		glVertexAttribPointer(ATTRIBUTE_NORMAL, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(ATTRIBUTE_NORMAL);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Grid_Index);
		glDrawElements(GL_TRIANGLES, gGrid_NumOfElements, GL_UNSIGNED_INT, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);


	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, 1);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);

	glUseProgram(0);


	modelMat = mat4::identity();
	modelMat = translate(-30.0f, 26.0f, 0.0f) * scale(1.0f, 1.0f, 1.0f);
	display_ModelWithLight(display_Model, gpModelLightHouse, modelMat);

	// display_GodRays_LightHouse();


}


void display_Terrain_TF_Pass(void){

	void CalculateGridNormals(void);

	mat4 modelMat;
	mat4 viewMat;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(Terrain_gShaderProgramObject);

	modelMat = mat4::identity();
	viewMat = mat4::identity();

	modelMat = translate(0.0f, -2.00f, -5.0f) * rotate(10.0f, 0.0f, 0.0f);


	glUniformMatrix4fv(Terrain_modelMatUniform, 1, GL_FALSE, modelMat);
	glUniformMatrix4fv(Terrain_viewMatUniform, 1, GL_FALSE, viewMat);
	glUniformMatrix4fv(Terrain_projMatUniform, 1, GL_FALSE, PerspectiveProjectionMatrix);


	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, Terrain_textureHtMap);
	glUniform1i(Terrain_samplerHtMapUniform, 0);

	glUniform1i(Terrain_choiceUniform, TERR_SINGLE_COLOR);	


	// Start Transformed Feedback
	glEnable(GL_RASTERIZER_DISCARD);
	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbo_Grid_Normals);
	glBeginTransformFeedback(GL_POINTS);

	glBindVertexArray(vao_Grid);
		glDrawArrays(GL_POINTS, 0, GRID_WIDTH * GRID_HEIGHT);
	glBindVertexArray(0);

	glEndTransformFeedback();
	glDisable(GL_RASTERIZER_DISCARD);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);


	glUseProgram(0);

	CalculateGridNormals();

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0);

	SwapBuffers(ghdc);

}


void CalculateGridNormals(void){

	GLuint  size = GRID_WIDTH * GRID_HEIGHT * 4;

	grid_Transformed_Pos = (GLfloat*)malloc(sizeof(GLfloat) * size);

	// memset(grid_Transformed_Pos, 0, size * sizeof(GLfloat));

	glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, size, (void*)grid_Transformed_Pos);

	GLenum err = glGetError();

	fprintf(gpFile, "%d\n", err);

	// For Value Checking
	
	// for(int i = 0; i < size; i = i + 4){

	// 	fprintf(gpFile, "%f/ %f/ %f/ %f\n",  grid_Transformed_Pos[i + 0], grid_Transformed_Pos[i + 1], grid_Transformed_Pos[i + 2], grid_Transformed_Pos[i + 3]);
	
	// }		


	for(int i = 0; i < gGrid_NumOfElements; i = i + 3){

		GLuint x, y, z;

		x = grid_Index[i + 0];
		y = grid_Index[i + 1];
		z = grid_Index[i + 2];

		GLuint xi, yi, zi;
		GLuint xj, yj, zj;

		xi = floor(x / GRID_WIDTH);
		yi = floor(y / GRID_WIDTH);
		zi = floor(z / GRID_WIDTH);

		xj = x % GRID_WIDTH;
		yj = y % GRID_WIDTH;
		zj = z % GRID_WIDTH;

		GLuint index0, index1, index2;

		index0 = (GRID_WIDTH * 4 * xi) + (xj * 4);
		index1 = (GRID_WIDTH * 4 * yi) + (yj * 4);
		index2 = (GRID_WIDTH * 4 * zi) + (zj * 4);

		vec3 v0 = vec3(grid_Transformed_Pos[index0 + 0], grid_Transformed_Pos[index0 + 1], grid_Transformed_Pos[index0 + 2]);		
		vec3 v1 = vec3(grid_Transformed_Pos[index1 + 0], grid_Transformed_Pos[index1 + 1], grid_Transformed_Pos[index1 + 2]);		
		vec3 v2 = vec3(grid_Transformed_Pos[index2 + 0], grid_Transformed_Pos[index2 + 1], grid_Transformed_Pos[index2 + 2]);		

		// fprintf(gpFile, "%f/ %f/ %f\n", v0[0], v0[1],  v0[2]);

		// fprintf(gpFile, "%f / %f/ %f\n", 
		// 	grid_Transformed_Pos[index0 + 0],
		// 	grid_Transformed_Pos[index0 + 1],
		// 	grid_Transformed_Pos[index0 + 2]);



		vec3 v1_v0, v2_v0;

		v1_v0 = v1 - v0;
		v2_v0 = v2 - v0;

		vec3 normal;

		normal = cross(v1_v0, v2_v0);

		normal = normalize(normal);


		fprintf(gpFile, "%f/ %f/ %f\n", normal[0], normal[1],  normal[2]);
		//fflush(gpFile);

		vec3 tempNormal;

		// Vertex 1
		tempNormal = normal + vec3(grid_Normals[index0 + 0], grid_Normals[index0 + 1], grid_Normals[index0 + 2]);
		tempNormal = normalize(tempNormal);

		grid_Normals[index0 + 0] = tempNormal[0];
		grid_Normals[index0 + 1] = tempNormal[1];
		grid_Normals[index0 + 2] = tempNormal[2];
		grid_Normals[index0 + 3] = 1.0f;

		// Vertex 2
		tempNormal = normal + vec3(grid_Normals[index1 + 0], grid_Normals[index1 + 1], grid_Normals[index1 + 2]);
		tempNormal = normalize(tempNormal);

		grid_Normals[index1 + 0] = tempNormal[0];
		grid_Normals[index1 + 1] = tempNormal[1];
		grid_Normals[index1 + 2] = tempNormal[2];
		grid_Normals[index1 + 3] = 1.0f;


		// Vertex 3
		tempNormal = normal + vec3(grid_Normals[index2 + 0], grid_Normals[index2 + 1], grid_Normals[index2 + 2]);
		tempNormal = normalize(tempNormal);

		grid_Normals[index2 + 0] = tempNormal[0];
		grid_Normals[index2 + 1] = tempNormal[1];
		grid_Normals[index2 + 2] = tempNormal[2];
		grid_Normals[index2 + 3] = 1.0f;

	}

	free(grid_Transformed_Pos);
	grid_Transformed_Pos = NULL;
}



void LoadGrid(void){


	GLfloat fX = gGrid_StartX;
	GLfloat fZ = gGrid_StartZ;

	GLfloat tX = 0.0;
	GLfloat tZ = 1.0;

	for(int i = 0; i < GRID_HEIGHT; i++){
	
		fX = gGrid_StartX;
		tX = 0.0;
		

		for(int j = 0; j < GRID_WIDTH; j++){

			grid_Position[(i * 3 * GRID_WIDTH) + (j * 3) + 0] = fX;
			grid_Position[(i * 3 * GRID_WIDTH) + (j * 3) + 1] = 0.0;
			grid_Position[(i * 3 * GRID_WIDTH) + (j * 3) + 2] = fZ;

			grid_Texcoord[(i * 2 * GRID_WIDTH) + (j * 2) + 0] = tX;
			grid_Texcoord[(i * 2 * GRID_WIDTH) + (j * 2) + 1] = tZ;

			// gGrid_Normals[(i * 3 * GRID_WIDTH) + (j * 3) + 0] = 0.0;
			// gGrid_Normals[(i * 3 * GRID_WIDTH) + (j * 3) + 1] = 1.0;
			// gGrid_Normals[(i * 3 * GRID_WIDTH) + (j * 3) + 2] = 0.0;

			fX = fX + ((2.0 * fabs(gGrid_StartX)) / (GRID_WIDTH - 1));
			tX = tX + (1.0 / (GRID_WIDTH - 1));

		}

		fZ = fZ + ((2.0 * fabs(gGrid_StartZ)) / (GRID_HEIGHT - 1));
		tZ = tZ - (1.0 / (GRID_HEIGHT - 1));
	}


	GLuint index = 0;

	for(int i = 0; i < (GRID_HEIGHT - 1); i++){

		for(int j = 0; j < (GRID_WIDTH - 1); j++){

			GLuint topLeft = (i * GRID_WIDTH) + j;
			GLuint bottomLeft = ((i + 1) * GRID_WIDTH) + j;
			GLuint topRight = topLeft + 1;
			GLuint bottomRight = bottomLeft + 1;

			grid_Index[index + 0] = topLeft;
			grid_Index[index + 1] = bottomLeft;
			grid_Index[index + 2] = topRight;

			grid_Index[index + 3] = topRight;
			grid_Index[index + 4] = bottomLeft;
			grid_Index[index + 5] = bottomRight;

			//console.log("Index-:> ", grid_Index[index + 0], grid_Index[index + 1], grid_Index[index + 2], " ", grid_Index[index + 3], grid_Index[index + 4], grid_Index[index + 5]);

			index = index + 6;
			gGrid_NumOfElements = index;
		}
	}

	fprintf(gpFile, "Terrain : gGrid_NumOfElements: %d\n", gGrid_NumOfElements);
}
