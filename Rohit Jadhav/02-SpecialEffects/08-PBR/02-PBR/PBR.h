//For Shader Object
GLint PBR_gShaderProgramObject = 0;



//For Texture
GLuint PBR_texture_Moon;
GLuint PBR_texture_albedo;
GLuint PBR_texture_ao;
GLuint PBR_texture_roughness;
GLuint PBR_texture_metalic;
GLuint PBR_texture_normal;


GLfloat PBR_pLightPosition[] = {
	0.0f, 0.0f, 1.0f
};

GLfloat PBR_pLightColor[] = {
	1.0f, 1.0f, 1.0f
};

GLfloat PBR_lightCount = 1;

GLfloat PBR_cameraPosition[] = {0.0f, 0.0f, 0.0f};


// Uniform
GLuint PBR_samplerMoonUniform;
GLuint PBR_samplerAlbedoUniform;
GLuint PBR_samplerAOUniform;
GLuint PBR_samplerRoughnessUniform;
GLuint PBR_samplerMetalicUniform;
GLuint PBR_samplerNormalUniform;

GLuint PBR_lightPositionUniform;
GLuint PBR_lightColorUniform;
GLuint PBR_lightCountUniform;

GLuint PBR_cameraPositionUniform;

GLuint PBR_modelMatrixUniform;
GLuint PBR_viewMatrixUniform;
GLuint PBR_ProjectionMatrixUniform;


void initialize_PBR(void){

	void uninitialize(void);

	GLuint PBR_iVertexShaderObject;
	GLuint PBR_iFragmentShaderObject;



	/*********** Vertex Shader **********/
	PBR_iVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

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
		
		"out vec3 outWorldPos;" \
		"out vec3 outNormal;" \

		"void main(void)" \
		"{" \

			"outTex = vTex * 2.0f;" \
			"outWorldPos = vec3(u_model_matrix * vec4(vPosition.xyz, 1.0f));" \
			"outNormal = mat3(u_view_matrix * u_model_matrix) * vNormal;" \
			
			"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \

		"}";

	glShaderSource(PBR_iVertexShaderObject, 1, (const GLchar**)&szVertexShaderSourceCode, NULL);

	glCompileShader(PBR_iVertexShaderObject);

	GLint iShaderCompileStatus = 0;
	GLint iInfoLogLength = 0;
	GLchar *szInfoLog = NULL;

	glGetShaderiv(PBR_iVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE) {
		//Error ahe
		glGetShaderiv(PBR_iVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetShaderInfoLog(PBR_iVertexShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Vertex Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				DestroyWindow(ghwnd);

			}
		}
	}

	/********** Fragment Shader **********/
	PBR_iFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *szFragmentShaderSourceCode =
		"#version 450 core" \
		"\n" \

		"in vec2 outTex;" \
		"in vec3 outWorldPos;" \
		"in vec3 outNormal;" \


		"uniform sampler2D u_sampler;" \
		"uniform sampler2D u_samplerAlbedo;" \
		"uniform sampler2D u_samplerAO;" \
		"uniform sampler2D u_samplerRoughness;" \
		"uniform sampler2D u_samplerMetalic;" \
		"uniform sampler2D u_samplerNormals;" \

		
		"out vec4 FragColor;" \

		"const float PI = 3.14159265359f;" \

		// -----------------------------------------------------------------------------------------------
		// Normal Distribution Function (D)
		"float NormalDistributionFunctionGGX(vec3 normal, vec3 halfVector, float roughness){" \

			"float D;" \

			"float a = roughness * roughness;" \
			"float a2 = a * a;" \

			"float NdotH = max(dot(normal, halfVector), 0.0f);" \
			"float NdotH2 = NdotH * NdotH;" \

			"float numerator = a2;" \
			"float denominator = (NdotH2 * (a2 - 1.0f) + 1.0f);" \
			"denominator = PI * denominator *  denominator;" \

			"D = numerator / denominator;" \
 
			"return(D);" \
		"}" \


		// -----------------------------------------------------------------------------------------------
		//Geometry Function (G)
		"float GeometrySchlickGGX(float NdotV, float k){" \

			"float numerator = NdotV;" \
			"float denominator = NdotV * (1.0f - k) + k;" \

			"return(numerator / denominator);" \

		"}" \


		"float GeometrySmith(vec3 normal, vec3 viewVector, vec3 lightDir, float k){" \

			"float G;" \

			"float NdotV = max(dot(normal, viewVector), 0.0f);" \
			"float NdotL = max(dot(normal, lightDir), 0.0f);" \

			"float ggx1 = GeometrySchlickGGX(NdotV, k);" \
			"float ggx2 = GeometrySchlickGGX(NdotL, k);" \

			"G = ggx1 * ggx2;" \

			"return(G);" \
 
		"}" \



		// -----------------------------------------------------------------------------------------------
		//Fresnel Equation (F)
		"vec3 fresnelSchlick(float cosTheta, vec3 F0){" \

			"vec3 F;" \

			"F = F0 + (1.0f - F0) * pow(clamp(1.0f - cosTheta, 0.0f, 1.0f), 5);" \

			"return(F);" \

		"}" \


		// -----------------------------------------------------------------------------------------------
		// Calculation of Tangent and BiTangent
		"vec3 getNormalFromMap(vec3 normal){" \

			"vec3 tangentNormal = texture(u_samplerNormals, outTex).xyz * 2.0f - 1.0f;" \

			"vec3 Q1 = dFdx(outWorldPos);" \
			"vec3 Q2 = dFdy(outWorldPos);" \
			"vec2 st1 = dFdx(outTex);" \
			"vec2 st2 = dFdy(outTex);" \

			"vec3 N = normalize(normal);" \
			"vec3 T = normalize(Q1 * st2.t - Q2 * st1.t);" \
			"vec3 B = -normalize(cross(N, T));" \
			"mat3 TBN = mat3(T, B, N);" \

			"return(normalize(TBN * tangentNormal));" \

		"}" \



		// -----------------------------------------------------------------------------------------------
		
		"uniform vec3 u_camPos;" \
		"uniform vec3 u_lightPosition[1];" \
		"uniform vec3 u_lightColor[1];" \
		"uniform int u_lightCount;" \



		"void main(void)" \
		"{" \

			"vec3 color;" \
			"vec3 color1;" \


			"vec3 albedo = texture(u_samplerAlbedo, outTex).xyz;" \
			"float roughness = texture(u_samplerRoughness, outTex).r;" \
			"roughness = 1.0f - roughness;" \
			
			"float metalic = texture(u_samplerMetalic, outTex).r;" \
			
			// "float metalic = 0.0f;" \

			"float ao = texture(u_samplerAO, outTex).r;" \


			//"vec3 normal = normalize(outNormal);" \
			
			"vec3 normal = getNormalFromMap(outNormal);" \


			"vec3 viewVector = normalize(u_camPos - outWorldPos);" \

			// For Fresnel Equation
			"vec3 F0 = vec3(0.04f);" \
			"F0 = mix(F0, albedo, metalic);" \


			// Calculate Reflectance
			"vec3 L0 = vec3(0.0f);" \
			"for(int i = 0; i < u_lightCount; i++){" \

				
			        // Calculate Per Light Radiance
			        "vec3 lightDir = normalize(u_lightPosition[i] - outWorldPos);" \
			        "vec3 halfVector = normalize(viewVector + lightDir);" \
			        "float distance = length(u_lightPosition[i] - outWorldPos);" \
			        "float attenuation = 1.0f / (distance * distance);" \
			        "vec3 radiance = u_lightColor[i];" \


			        // Cook-Torrance BRDF
			        "float NDF = NormalDistributionFunctionGGX(normal, halfVector, roughness);" \
			        "float G = GeometrySmith(normal, viewVector, lightDir, roughness);" \
			        "vec3 F = fresnelSchlick(clamp(dot(halfVector, viewVector), 0.0f, 1.0f), F0);" \

			        "vec3 numerator = NDF * G * F;" \

			        // + 0.0001 to prevent divide by zero
			        "float denominator = 4.0f * max(dot(normal, viewVector), 0.0f) * max(dot(normal, lightDir), 0.0f) + 0.0001f;" \
			        "vec3 specular = numerator / denominator;" \

			        // For Diffuse and Specular Intensity
			        "vec3 kS = F;" \
			        "vec3 kD = 1.0f - kS;" \

			        // Jr pure metal asel tr diffuse ha 0.0f asnar because metal only reflect krto refract nahi
			        // jr partial asel tr thoda diffuse disel tymule tyla 1.0f - metalic ni multiply kela ahe
			        "kD = kD * (1.0f - metalic);" \

			        // For (n . wi) from the Reflectance Equation
			        "float NdotL = max(dot(normal, lightDir), 0.0f);" \


 				"L0 += (kD * (albedo / PI) + specular) * radiance * NdotL;" \

 				"color1 = vec3(L0);" \

			"}" \


			"vec3 ambiant = albedo * ao;" \

			"color = ambiant + L0;" \

			// // HDR tonemapping
			// "color = color / (color + vec3(1.0f));" \

			// // Gamma Correct	
			// "color = pow(color, vec3(1.0f / 2.2f));" \

			"FragColor = vec4(vec3(color), 1.0f);" \
		"}";

	glShaderSource(PBR_iFragmentShaderObject, 1,
		(const GLchar**)&szFragmentShaderSourceCode, NULL);

	glCompileShader(PBR_iFragmentShaderObject);

	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;
	glGetShaderiv(PBR_iFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE) {
		//Error ahe
		glGetShaderiv(PBR_iFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetShaderInfoLog(PBR_iFragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Fragment Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				DestroyWindow(ghwnd);
			}
		}
	}

	

	/********** Shader Program Object **********/
	PBR_gShaderProgramObject = glCreateProgram();

	glAttachShader(PBR_gShaderProgramObject, PBR_iVertexShaderObject);
	glAttachShader(PBR_gShaderProgramObject, PBR_iFragmentShaderObject);

	/********** Bind Vertex Attribute **********/
	glBindAttribLocation(PBR_gShaderProgramObject, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(PBR_gShaderProgramObject, AMC_ATTRIBUTE_NORMAL, "vNormal");
	glBindAttribLocation(PBR_gShaderProgramObject, AMC_ATTRIBUTE_TEXCOORD0, "vTex");

	glLinkProgram(PBR_gShaderProgramObject);

	int iProgramLinkStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;
	glGetProgramiv(PBR_gShaderProgramObject, GL_LINK_STATUS, &iProgramLinkStatus);
	if (iProgramLinkStatus == GL_FALSE) {
		glGetProgramiv(PBR_gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetProgramInfoLog(PBR_gShaderProgramObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Shader Progame Object Linking Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				DestroyWindow(ghwnd);
			}
		}
	}

	/********** Getting Uniforms Location **********/
	PBR_modelMatrixUniform = glGetUniformLocation(PBR_gShaderProgramObject, "u_model_matrix");
	PBR_viewMatrixUniform = glGetUniformLocation(PBR_gShaderProgramObject, "u_view_matrix");
	PBR_ProjectionMatrixUniform = glGetUniformLocation(PBR_gShaderProgramObject, "u_projection_matrix");
	
	PBR_samplerMoonUniform = glGetUniformLocation(PBR_gShaderProgramObject, "u_sampler");
	PBR_samplerAlbedoUniform = glGetUniformLocation(PBR_gShaderProgramObject, "u_samplerAlbedo");
	PBR_samplerAOUniform = glGetUniformLocation(PBR_gShaderProgramObject, "u_samplerAO");
	PBR_samplerRoughnessUniform = glGetUniformLocation(PBR_gShaderProgramObject, "u_samplerRoughness");
	PBR_samplerMetalicUniform = glGetUniformLocation(PBR_gShaderProgramObject, "u_samplerMetalic");
	PBR_samplerNormalUniform = glGetUniformLocation(PBR_gShaderProgramObject, "u_samplerNormals");

	PBR_lightPositionUniform = glGetUniformLocation(PBR_gShaderProgramObject, "u_lightPosition");
	PBR_lightColorUniform = glGetUniformLocation(PBR_gShaderProgramObject, "u_lightColor");
	PBR_lightCountUniform = glGetUniformLocation(PBR_gShaderProgramObject, "u_lightCount");

	PBR_cameraPositionUniform = glGetUniformLocation(PBR_gShaderProgramObject, "u_camPos");


	glEnable(GL_TEXTURE_2D);
	
	PBR_texture_Moon = stbLoadTexture("02-PBR/01-tex/displacement.png");
	PBR_texture_albedo = stbLoadTexture("02-PBR/01-tex/albedo.png");
	PBR_texture_ao =  stbLoadTexture("02-PBR/01-tex/ao.png");
	PBR_texture_roughness =  stbLoadTexture("02-PBR/01-tex/roughness.png");
	PBR_texture_metalic =  stbLoadTexture("02-PBR/01-tex/metalic.png");
	PBR_texture_normal = stbLoadTexture("02-PBR/01-tex/normal.png");



	
}


void uninitialize_PBR(void){

	if(PBR_texture_albedo){
		glDeleteTextures(1, &PBR_texture_albedo);
		PBR_texture_albedo = 0;
	}

	if(PBR_texture_ao){
		glDeleteTextures(1, &PBR_texture_ao);
		PBR_texture_ao = 0;
	}

	if(PBR_texture_roughness){
		glDeleteTextures(1, &PBR_texture_roughness);
		PBR_texture_roughness = 0;
	}

	if(PBR_texture_metalic){
		glDeleteTextures(1, &PBR_texture_metalic);
		PBR_texture_metalic = 0;
	}	

	if(PBR_texture_normal){
		glDeleteTextures(1, &PBR_texture_normal);
		PBR_texture_normal = 0;
	}	

	if(PBR_texture_Moon){
		glDeleteTextures(1, &PBR_texture_Moon);
		PBR_texture_Moon = 0;
	}


	if(PBR_gShaderProgramObject){

		glUseProgram(PBR_gShaderProgramObject);

		GLint iShaderCount;
		GLint iShaderNumber;

		glGetProgramiv(PBR_gShaderProgramObject, GL_ATTACHED_SHADERS, &iShaderCount);

		GLuint *pShaders = (GLuint*)malloc(sizeof(GLuint) * iShaderCount);

		if(pShaders){

			glGetAttachedShaders(PBR_gShaderProgramObject, iShaderCount, &iShaderCount, pShaders);

			for(iShaderNumber = 0; iShaderNumber < iShaderCount; iShaderNumber++){

				glDetachShader(PBR_gShaderProgramObject, pShaders[iShaderNumber]);
				glDeleteShader(pShaders[iShaderNumber]);

				fprintf(gpFile, "\nShader %d Detached and Deleted", iShaderNumber+1);
			}

			free(pShaders);
			pShaders = NULL;

		}

		glUseProgram(0);

		glDeleteProgram(PBR_gShaderProgramObject);
		PBR_gShaderProgramObject = 0;

	}

	
}



void display_PBR(void) {

	void update_PBR(void);

	mat4 translateMatrix;
	mat4 rotateMatrix;
	mat4 modelMatrix;
	mat4 viewMatrix;



	glUseProgram(PBR_gShaderProgramObject);
	

		translateMatrix = mat4::identity();
		rotateMatrix = mat4::identity();
		modelMatrix = mat4::identity();
		viewMatrix = mat4::identity();
		

		translateMatrix = translate(0.0f, 0.0f, -3.0f);
		// rotateMatrix = rotate(60.0f, 1.0f, 0.0f, 0.0f);
		modelMatrix = modelMatrix * translateMatrix * rotateMatrix;

		viewMatrix = lookat(c.cameraPosition, c.cameraPosition + c.cameraFront, c.cameraUp);
		PBR_cameraPosition[0] = c.cameraPosition[0];
		PBR_cameraPosition[1] = c.cameraPosition[1];
		PBR_cameraPosition[2] = c.cameraPosition[2];


		glUniformMatrix4fv(PBR_modelMatrixUniform, 1, GL_FALSE, modelMatrix);
		glUniformMatrix4fv(PBR_viewMatrixUniform, 1, GL_FALSE, viewMatrix);
		glUniformMatrix4fv(PBR_ProjectionMatrixUniform, 1, GL_FALSE, global_PerspectiveProjectionMatrix);


		glUniform3fv(PBR_lightPositionUniform, 1, PBR_pLightPosition);
		glUniform3fv(PBR_lightColorUniform, 1, PBR_pLightColor);
		glUniform1i(PBR_lightCountUniform, PBR_lightCount);
		
		glUniform3fv(PBR_cameraPositionUniform, 1, PBR_cameraPosition);


		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, PBR_texture_Moon);
		glUniform1i(PBR_samplerMoonUniform, 0);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, PBR_texture_albedo);
		glUniform1i(PBR_samplerAlbedoUniform, 1);

		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, PBR_texture_ao);
		glUniform1i(PBR_samplerAOUniform, 2);

		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, PBR_texture_roughness);
		glUniform1i(PBR_samplerRoughnessUniform, 3);

		glActiveTexture(GL_TEXTURE4);
		glBindTexture(GL_TEXTURE_2D, PBR_texture_metalic);
		glUniform1i(PBR_samplerMetalicUniform, 4);

		glActiveTexture(GL_TEXTURE5);
		glBindTexture(GL_TEXTURE_2D, PBR_texture_normal);
		glUniform1i(PBR_samplerNormalUniform, 5);

		// draw_Sphere();
		draw_Rect(0);

		glActiveTexture(GL_TEXTURE5);
		glBindTexture(GL_TEXTURE_2D, 0);

		glActiveTexture(GL_TEXTURE4);
		glBindTexture(GL_TEXTURE_2D, 0);

		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, 0);

		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, 0);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, 0);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, 0);

	glUseProgram(0);

	update_PBR();

}

void update_PBR(void){

	static GLfloat PBR_angle_Sphere = 0.0f;

	PBR_angle_Sphere += 0.050f;
	if(PBR_angle_Sphere > 360.0f)
		PBR_angle_Sphere = 0.0f;


	// PBR_pLightPosition[0] = (10.0f * cos(PBR_angle_Sphere));  
	// PBR_pLightPosition[2] = (10.0f * sin(PBR_angle_Sphere));  

}


