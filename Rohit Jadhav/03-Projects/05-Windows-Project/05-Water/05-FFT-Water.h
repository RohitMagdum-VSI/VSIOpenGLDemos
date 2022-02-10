#include<cuda_gl_interop.h>
#include<cuda_runtime.h>
#include<cuda.h>
#include<cufft.h>


#include"05-FFT-Cuda.h"


#pragma comment(lib, "cudart.lib")


//For Shader Program Object;
GLint Water_gShaderProgramObject;


//For Wave
cudaError_t error;
cufftHandle fftPlan;

GLuint vao_Wave;
GLuint vbo_WavePos;
GLuint vbo_WaveHeight;
GLuint vbo_WaveSlope;
GLuint vbo_WaveElements;


//For Cuda Interop
struct cudaGraphicsResource *gHeight_GraphicsResource;
struct cudaGraphicsResource *gSlope_GraphicsResource;


//For Wave Data
float2 *h_h0 = NULL;
float2 *d_h0 = NULL;

float2 *d_ht = NULL;
float2 *d_Slope = NULL;

//For Pointer To Device Object used in display
float *g_hPtr = NULL;
float2 *g_sPtr = NULL;


//For Wave
#define RRJ_PI 3.1415926535f

const GLuint Water_gMeshSize = 512;
const GLuint Water_gSpectrumW = Water_gMeshSize + 4;
const GLuint Water_gSpectrumH = Water_gMeshSize + 1;

const GLfloat Water_gfGravity = 9.81f;
const GLfloat Water_gfAmplitude = 1E-7f;
const GLfloat Water_gfPatchSize = 100.0f;
GLfloat Water_gfWindSpeed = 5.0f;
GLfloat Water_gfWindDirection = RRJ_PI / 3.0f;
GLfloat Water_gfDirDepend = 0.07f;
GLfloat Water_gfAnimationTime = 1000.0f;
GLfloat Water_gfAnimFactor = 0.050f;



//For VS Uniform
GLuint Water_modelMatrixUniform;
GLuint Water_viewMatrixUniform;
GLuint Water_projectionMatrixUniform;

GLuint Water_meshSizeUniform;
GLuint Water_heightScaleUniform;
GLuint Water_chopinessUniform;

// For FS Uniform
GLuint Water_deepColorUniform;
GLuint Water_shallowColorUniform;
GLuint Water_skyColorUniform;
GLuint Water_lightDirectionUniform;

GLfloat deepColor[] = {0.0f, 0.1f, 0.4f, 1.0f};
GLfloat shallowColor[] = {0.1f, 0.3f, 0.3f, 1.0f};
GLfloat skyColor[] = {0.50f, 0.50f, 0.50f, 1.0f};
GLfloat lightDir[] = {-0.50f, 0.50f, 0.0f};
GLfloat heightScale = 8.1f;
GLfloat chopiness = 1.0f;
GLfloat Water_mesh[] = {Water_gMeshSize, Water_gMeshSize};

GLfloat Water_wavePos[Water_gMeshSize * Water_gMeshSize * 4];



void initialize_Water(void){

	void uninitialize(void);

	//Shader Object;
	GLint iVertexShaderObject;
	GLint iFragmentShaderObject;

	/********** Vertex Shader **********/
	iVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *szVertexShaderSourceCode =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in float vHeight;" \
		"in vec2 vSlope;" \

		"uniform mat4 u_modelMatrix;" \
		"uniform mat4 u_viewMatrix;" \
		"uniform mat4 u_projectionMatrix;" \

		"uniform float u_heightScale;" \
		"uniform float u_chopiness;" \
		"uniform vec2 u_meshSize;" \


		"out vec3 eyeSpacePos_VS;" \
		"out vec3 worldSpaceNormal_VS;" \
		"out vec3 eyeSpaceNormal_VS;" \

		// "const vec4 newPos[3] = { " \
		// 		"vec4(1.0f, 1.0f, 0.0f, 1.0f)," \
		// 		"vec4(-1.0f, -1.0f, 0.0f, 1.0f)," \
		// 		"vec4(1.0f, -1.0f, 0.0f, 1.0f)," \
		// 	"};" \




		"void main(void)" \
		"{" \
			"mat3 normalMatrix = mat3(u_viewMatrix * u_modelMatrix); " \

			"vec3 normal = normalize(cross(vec3(0.0f, vSlope.y * u_heightScale, 2.0f / u_meshSize.x), vec3(2.0f / u_meshSize.y, vSlope.x * u_heightScale, 0.0f)));" \

			"worldSpaceNormal_VS = normal;" \

			"vec4 pos = vec4(vPosition.x, vHeight * u_heightScale, vPosition.z, 1.0f);" \

			// "gl_Position = u_projectionMatrix * u_viewMatrix * u_modelMatrix * newPos[gl_VertexID];" \
			
			"gl_Position = u_projectionMatrix * u_viewMatrix * u_modelMatrix * pos;" \

			"eyeSpacePos_VS = vec3(u_viewMatrix * u_modelMatrix * pos);" \
			"eyeSpaceNormal_VS = vec3(normalMatrix * normal);" \

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
				fprintf(gpFile, "Water : Vertex Shader Compilation Error: %s\n", szInfoLog);
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

		"in vec3 eyeSpacePos_VS;" \
		"in vec3 worldSpaceNormal_VS;" \
		"in vec3 eyeSpaceNormal_VS;" \

		"uniform vec4 u_deepColor;" \
		"uniform vec4 u_shallowColor;" \
		"uniform vec4 u_skyColor;" \
		"uniform vec3 u_lightDirection;" \


		"out vec4 FragColor;" \

		"void main(void)" \
		"{" \
			
			"vec3 eyeSpacePosVec = normalize(eyeSpacePos_VS);" \
			"vec3 eyeSpaceNorVec = normalize(eyeSpaceNormal_VS);" \
			"vec3 worldSpaceNorVec = normalize(worldSpaceNormal_VS);" \


			"float facing = max(0.0f, dot(eyeSpaceNorVec, -eyeSpacePosVec));" \
			"float fresnel = pow(1.0f - facing, 5.0f);" \
			"float diffuse = max(0.0f, dot(worldSpaceNorVec, u_lightDirection));" \

			"vec4 waterColor = u_deepColor;" \

			// "vec4 waterColor = mix(vec4(1.0f, 0.0f, 0.0f, 1.0f), vec4(0.0f, 0.0f, 1.0f, 1.0f), facing);"

			"FragColor = waterColor * diffuse + u_skyColor * fresnel;" \

			   // "gl_FragColor = vec4(fresnel);" \
			 // "FragColor = vec4(diffuse); " \

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
				fprintf(gpFile, "Water : Fragment Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				DestroyWindow(ghwnd);
				exit(0);
			}
		}
	}

	/********** Shader Program Object **********/
	Water_gShaderProgramObject = glCreateProgram();

	glAttachShader(Water_gShaderProgramObject, iVertexShaderObject);
	glAttachShader(Water_gShaderProgramObject, iFragmentShaderObject);

	glBindAttribLocation(Water_gShaderProgramObject, ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(Water_gShaderProgramObject, ATTRIBUTE_HEIGHTMAP, "vHeight");
	glBindAttribLocation(Water_gShaderProgramObject, ATTRIBUTE_SLOPE, "vSlope");

	glLinkProgram(Water_gShaderProgramObject);

	GLint iProgramLinkingStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;
	glGetProgramiv(Water_gShaderProgramObject, GL_LINK_STATUS, &iProgramLinkingStatus);
	if (iProgramLinkingStatus == GL_FALSE) {
		glGetProgramiv(Water_gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetProgramInfoLog(Water_gShaderProgramObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Water : Shader Program Object Linking Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				DestroyWindow(ghwnd);
				exit(0);
			}
		}
	}


	Water_modelMatrixUniform = glGetUniformLocation(Water_gShaderProgramObject, "u_modelMatrix");
	Water_viewMatrixUniform = glGetUniformLocation(Water_gShaderProgramObject, "u_viewMatrix");
	Water_projectionMatrixUniform = glGetUniformLocation(Water_gShaderProgramObject, "u_projectionMatrix");

	Water_heightScaleUniform = glGetUniformLocation(Water_gShaderProgramObject, "u_heightScale");
	Water_chopinessUniform = glGetUniformLocation(Water_gShaderProgramObject, "u_chopiness");
	Water_meshSizeUniform = glGetUniformLocation(Water_gShaderProgramObject, "u_meshSize");

	Water_deepColorUniform = glGetUniformLocation(Water_gShaderProgramObject, "u_deepColor");
	Water_shallowColorUniform = glGetUniformLocation(Water_gShaderProgramObject, "u_shallowColor");
	Water_skyColorUniform = glGetUniformLocation(Water_gShaderProgramObject, "u_skyColor");
	Water_lightDirectionUniform = glGetUniformLocation(Water_gShaderProgramObject, "u_lightDirection");





	// ***** Wave Position *****
	memset(Water_wavePos, 0.0f, sizeof(GLfloat) * Water_gMeshSize * Water_gMeshSize * 4);

	int index = 0;

	for(int y = 0; y < Water_gMeshSize; y++){

		for(int x = 0; x < Water_gMeshSize; x++){

			index = (y * Water_gMeshSize * 4) + (x * 4);	

			GLfloat u = x / (GLfloat)(Water_gMeshSize - 1);
			GLfloat v = y / (GLfloat)(Water_gMeshSize - 1);

			//fprintf(gpFile, "Index : %d\n", index);

			Water_wavePos[index + 0] = u * 2.0f - 1.0f;
			Water_wavePos[index + 1] = 0.0f;
			Water_wavePos[index + 2] = v * 2.0f - 1.0f;
			Water_wavePos[index + 3] = 1.0f;

		}
	}

	GLuint size = ((Water_gMeshSize * 2) + 2) * (Water_gMeshSize -1) * sizeof(GLuint);
	//For Information ((w * 2) + 2) * (h - 1) * size(GLuint);


	// ***** For Wave *****
	glGenVertexArrays(1, &vao_Wave);
	glBindVertexArray(vao_Wave);

		// ***** Position *****
		glGenBuffers(1, &vbo_WavePos);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_WavePos);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 4 * Water_gMeshSize * Water_gMeshSize, Water_wavePos, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		// ***** Slope *****
		glGenBuffers(1, &vbo_WaveSlope);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_WaveSlope);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * Water_gMeshSize * Water_gMeshSize, NULL, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(ATTRIBUTE_SLOPE, 2, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(ATTRIBUTE_SLOPE);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		// ***** Height *****
		glGenBuffers(1, &vbo_WaveHeight);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_WaveHeight);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * Water_gMeshSize * Water_gMeshSize, NULL, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(ATTRIBUTE_HEIGHTMAP, 1, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(ATTRIBUTE_HEIGHTMAP);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		// ***** Index *****
		glGenBuffers(1, &vbo_WaveElements);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_WaveElements);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);

		GLuint *indices = (GLuint*)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);

		if(indices == NULL){
			fprintf(gpFile, "Water ERROR: glMapBuffer() For Elements Failed\n");
			uninitialize();
			DestroyWindow(ghwnd);
		}


		for(int y = 0; y < Water_gMeshSize - 1; y++){

			for(int x = 0; x < Water_gMeshSize; x++){

				*indices++ = y * Water_gMeshSize + x;
				*indices++ = (y + 1) * Water_gMeshSize + x;

			}

			*indices++ = (y + 1) * Water_gMeshSize + (Water_gMeshSize - 1);
			*indices++ = (y + 1) * Water_gMeshSize;
		}

		glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


	glBindVertexArray(0);



	// ***** Graphics Resource For Height Map *****
	error = cudaGraphicsGLRegisterBuffer(
				&gHeight_GraphicsResource, 
				vbo_WaveHeight, 
				cudaGraphicsMapFlagsWriteDiscard);

	if(error != cudaSuccess){
		fprintf(gpFile, "Water ERROR: cudaGraphicsGLRegisterBuffer() For HeightMap Failed\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}


	// ***** Graphics Resource For Slope *****
	error = cudaGraphicsGLRegisterBuffer(
				&gSlope_GraphicsResource,
				vbo_WaveSlope,
				cudaGraphicsMapFlagsWriteDiscard);

	if(error != cudaSuccess){
		fprintf(gpFile, "Water ERROR: cudaGraphicsGLRegisterBuffer() For Slope Failed\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}



	// ***** Memory For Arrays *****
	GLuint spectrumSize = Water_gSpectrumW * Water_gSpectrumH * sizeof(float2);
	GLuint meshSize = Water_gMeshSize * Water_gMeshSize * sizeof(float);
	

	// *** Host Memory ***
	h_h0 = (float2*)malloc(spectrumSize);
	if(h_h0 == NULL){
		fprintf(gpFile, "Water ERROR: malloc() for h_h0 Failed\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}

	// *** Device Memory ***
	error = cudaMalloc((void**)&d_h0, spectrumSize);
	if(error != cudaSuccess){
		fprintf(gpFile, "Water ERROR: cudaMalloc() Failed for d_h0 with : %s\n", cudaGetErrorString(error));
		uninitialize();
		DestroyWindow(ghwnd);
	}

	error = cudaMalloc((void**)&d_ht, meshSize);
	if(error != cudaSuccess){
		fprintf(gpFile, "Water ERROR: cudaMalloc() failed for d_ht with : %s\n", cudaGetErrorString(error));
		uninitialize();
		DestroyWindow(ghwnd);
	}


	error = cudaMalloc((void**)&d_Slope, meshSize);
	if(error != cudaSuccess){
		fprintf(gpFile, "Water ERROR: cudaMalloc() failed for d_Slope with : %s\n", cudaGetErrorString(error));
		uninitialize();
		DestroyWindow(ghwnd);
	}



	// ***** Genenrate Initial Height Field *****
	Generate_H0(h_h0);

	error = cudaMemcpy(d_h0, h_h0, spectrumSize, cudaMemcpyHostToDevice);
	if(error != cudaSuccess){
		fprintf(gpFile, "Water ERROR: cudaMemcpy() failed for h_h0 -> d_h0 with : %s\n", cudaGetErrorString(error));
		uninitialize();
		DestroyWindow(ghwnd);
	}


	cufftResult fftResult;

	fftResult = cufftPlan2d(&fftPlan, Water_gMeshSize, Water_gMeshSize, CUFFT_C2C);
	if(fftResult != CUFFT_SUCCESS){
		fprintf(gpFile, "Water ERROR: cufftPlan2d");
		uninitialize();
		DestroyWindow(ghwnd);
	}

	fprintf(gpFile, "Water : initialize_Water() Done\n");
}


void uninitialize_Water(void){
	
	cudaGraphicsUnregisterResource(gHeight_GraphicsResource);
	cudaGraphicsUnregisterResource(gSlope_GraphicsResource);


	cufftDestroy(fftPlan);

	if(d_Slope){
		cudaFree(d_Slope);
		d_Slope = NULL;
	}

	if(d_ht){
		cudaFree(d_ht);
		d_ht = NULL;
	}

	if(d_h0){
		cudaFree(d_h0);
		d_h0 = NULL;
	}

	if(h_h0){
		free(h_h0);
		h_h0 = NULL;
	}


	if(vbo_WaveElements){
		glDeleteBuffers(1, &vbo_WaveElements);
		vbo_WaveElements = 0;
	}

	if(vbo_WaveHeight){
		glDeleteBuffers(1, &vbo_WaveHeight);
		vbo_WaveHeight = 0;
	}

	if(vbo_WaveSlope){
		glDeleteBuffers(1, &vbo_WaveSlope);
		vbo_WaveSlope = 0;
	}

	if(vbo_WavePos){
		glDeleteBuffers(1, &vbo_WavePos);
		vbo_WavePos = NULL;
	}

	if(vao_Wave){
		glDeleteVertexArrays(1, &vao_Wave);
		vao_Wave = NULL;
	}




	GLsizei ShaderCount;
	GLsizei ShaderNumber;

	if (Water_gShaderProgramObject) {
		glUseProgram(Water_gShaderProgramObject);

		glGetProgramiv(Water_gShaderProgramObject, GL_ATTACHED_SHADERS, &ShaderCount);

		GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * ShaderCount);
		if (pShader) {
			glGetAttachedShaders(Water_gShaderProgramObject, ShaderCount,
				&ShaderCount, pShader);
			for (ShaderNumber = 0; ShaderNumber < ShaderCount; ShaderNumber++) {
				glDetachShader(Water_gShaderProgramObject, pShader[ShaderNumber]);
				glDeleteShader(pShader[ShaderNumber]);
				pShader[ShaderNumber] = 0;
			}
			free(pShader);
			pShader = NULL;
		}
		glDeleteProgram(Water_gShaderProgramObject);
		Water_gShaderProgramObject = 0;
		glUseProgram(0);
	}


	fprintf(gpFile, "Water : uninitialize_Water() Done\n");
}


GLfloat urand(void){
	return(rand() / RAND_MAX);
}


GLfloat Gauss(void){

	GLfloat u1 = urand();
	GLfloat u2 = urand();

	if(u1 < 1e-6)
		u1 = 1e-6;


	GLfloat ret = sqrt(-2 * logf(u1)) * cosf(-2 * RRJ_PI * u2);
	return(ret);
}



// Phillips spectrum
GLfloat Philips(float Kx, float Ky, float windDir, float windSpeed, float A, float dirDepend){

	float k_square = Kx * Kx + Ky * Ky;

	if(k_square == 0.0f)
		return(0.0f);


	float L = windSpeed * windSpeed / Water_gfGravity;

	float k_x = Kx / sqrt(k_square);
	float k_y = Ky / sqrt(k_square);

	float w_dot_k = k_x * cosf(windDir) + k_y * sinf(windDir);

	float philips = A * exp(-1.0f /(k_square * L * L)) / (k_square * k_square) * w_dot_k * w_dot_k;

	if(w_dot_k < 0.0f){
		philips = philips * dirDepend;
	}

	return(philips);
}


void Generate_H0(float2 *h0){


	for(unsigned int y = 0; y <= Water_gMeshSize; y++){

		for(unsigned int x = 0; x <= Water_gMeshSize; x++){

			float kx = (-(int)Water_gMeshSize / 2 + x) * (2.0f * RRJ_PI / Water_gfPatchSize);
			float ky = (-(int)Water_gMeshSize / 2 + y) * (2.0f * RRJ_PI / Water_gfPatchSize);

			float p = sqrt(Philips(kx, ky, Water_gfWindDirection, Water_gfWindSpeed, Water_gfAmplitude, Water_gfDirDepend));

			if(kx == 0.0f && ky == 0.0f)
				p = 0.0f;

			float Er = Gauss();
			float Ei = Gauss();

			float h0_r = Er * p * sqrt(0.5f);
			float h0_i = Ei * p * sqrt(0.5f);

			int i = y * Water_gSpectrumW + x;
			h0[i].x = h0_r;
			h0[i].y = h0_i;
 		}
	}
}



void display_Water(void){

	mat4 water_TranslateMatrix;
	mat4 water_ModelMatrix;
	mat4 water_ViewMatrix;

	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(Water_gShaderProgramObject);

	water_TranslateMatrix = mat4::identity();
	water_ModelMatrix = mat4::identity();
	water_ViewMatrix = mat4::identity();


	// water_TranslateMatrix = translate(0.0f, 0.0f, -1.0f) ; //* scale(2.0f, 1.0f, 2.0f);
	// water_ModelMatrix = water_ModelMatrix * water_TranslateMatrix * rotate(10.0f, 1.0f, 0.0f, 0.0f);
	
	water_ModelMatrix = water_ModelMatrix *  translate(0.0f, 20.0f, 0.0f) * scale(100.0f, 1.0f, 100.0f);	
	water_ViewMatrix = lookat(c.cameraPosition, c.cameraPosition + c.cameraFront, c.cameraUp);	


	glUniformMatrix4fv(Water_modelMatrixUniform, 1, GL_FALSE, water_ModelMatrix);
	glUniformMatrix4fv(Water_viewMatrixUniform, 1, GL_FALSE, water_ViewMatrix);
	glUniformMatrix4fv(Water_projectionMatrixUniform, 1, GL_FALSE, PerspectiveProjectionMatrix);
	
	cudaGenerateSpectrumKernel(d_h0, d_ht, Water_gSpectrumW, Water_gMeshSize, Water_gMeshSize, Water_gfAnimationTime, Water_gfPatchSize);

	//FFT
	cufftResult fftResult;
	fftResult = cufftExecC2C(fftPlan, d_ht, d_ht, CUFFT_INVERSE);
	if(fftResult != CUFFT_SUCCESS){
		fprintf(gpFile, "Water ERROR: cufftExecC2C() failed");
		uninitialize();
		DestroyWindow(ghwnd);
	}


	// *** HeightMap ***
	size_t numOfBytes = 0;
	error = cudaGraphicsMapResources(1, &gHeight_GraphicsResource, 0);
	if(error != cudaSuccess){
		fprintf(gpFile, "ERROR: cudaGraphicsMapResource() failed for gHeight_GraphicsResource\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}


	error = cudaGraphicsResourceGetMappedPointer((void**)&g_hPtr, &numOfBytes, gHeight_GraphicsResource);
	if(error != cudaSuccess){
		fprintf(gpFile, "ERROR: cudaGraphicsResourceGetMappedPointer() failed for gHeight_GraphicsResource\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}


	cudaUpdateHeightMapKernel(g_hPtr, d_ht, Water_gMeshSize, Water_gMeshSize);



	// *** Slope ***
	error = cudaGraphicsMapResources(1, &gSlope_GraphicsResource, 0);
	if(error != cudaSuccess){
		fprintf(gpFile, "ERROR: cudaGraphicsMapResource() failed for gSlope_GraphicsResource\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}


	error = cudaGraphicsResourceGetMappedPointer((void**)&g_sPtr, &numOfBytes, gSlope_GraphicsResource);
	if(error != cudaSuccess){
		fprintf(gpFile, "ERROR: cudaGraphicsResourceGetMappedPointer() failed for gSlope_GraphicsResource\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}

	cudaCalculateSlopeKernel(g_hPtr, g_sPtr, Water_gMeshSize, Water_gMeshSize);


	// *** Unmap ***
	error = cudaGraphicsUnmapResources(1, &gSlope_GraphicsResource, 0);
	if(error != cudaSuccess){
		fprintf(gpFile, "ERROR: cudaGraphicsUnmapResource() failed for gSlope_GraphicsResource\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}

	error = cudaGraphicsUnmapResources(1, &gHeight_GraphicsResource, 0);
	if(error != cudaSuccess){
		fprintf(gpFile, "ERROR: cudaGraphicsUnmapResource() failed for gHeight_GraphicsResource\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}


	glUniform4fv(Water_deepColorUniform, 1, deepColor);
	glUniform4fv(Water_shallowColorUniform, 1, shallowColor);
	glUniform4fv(Water_skyColorUniform, 1, skyColor);
	glUniform3fv(Water_lightDirectionUniform, 1, lightDir);

	glUniform1f(Water_heightScaleUniform, heightScale);
	glUniform1f(Water_chopinessUniform, chopiness);
	glUniform2fv(Water_meshSizeUniform, 1, Water_mesh);

	glBindVertexArray(vao_Wave);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_WaveElements);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDrawElements(GL_TRIANGLE_STRIP, ((Water_gMeshSize * 2) + 2) * (Water_gMeshSize - 1),  GL_UNSIGNED_INT, 0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	
	// glDrawArrays(GL_TRIANGLES, 0, 3);


	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);


	glUseProgram(0);

	//animationTime = 1.0;
	Water_gfAnimationTime = Water_gfAnimationTime + Water_gfAnimFactor;

}
