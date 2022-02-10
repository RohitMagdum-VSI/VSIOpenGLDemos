//For External
var canvas;
var gl;



//For Shader
var vertexShaderObject_Grid;
var fragmentShaderObject_Grid;
var shaderProgramObject_Grid;

//For Uniform
var modelMatUniform;
var viewMatUniform;
var projMatUniform;
var grid_HeightMap_Sampler;
var grid_choiceUniform;
var shadowChoiceUniform;
var shadowMatUniform;
var depthMapSamplerUniform;
var clipPlaneUniform;


//For Grid
var vao_Grid;
var vbo_Grid_Pos_Memory;
var vbo_Grid_Pos_Reality;
var vbo_Grid_Normals_Memory;
var vbo_Grid_Normals_Reality;
var vbo_Grid_Texcoord;
var vbo_Grid_Texcoord_Terrain;
var vbo_Grid_Index;

const RRJ_PI = 3.1415926535;

const GRID_WIDTH = 256;
const GRID_HEIGHT = 256;
const gWater_Level = 35.0;
var gGrid_StartX = -1024.0;
var gGrid_StartZ = -1024.0;
var gGrid_NumOfElements = 0;
var gGrid_TexX = 0.0;
var gGrid_TexZ = 30.00;


var grid_Pos_Memory = new Float32Array(GRID_WIDTH * GRID_HEIGHT * 3);
var grid_Pos_Reality = new Float32Array(GRID_WIDTH * GRID_HEIGHT * 3);
var grid_Nor_Memory = new Float32Array(GRID_WIDTH * GRID_HEIGHT * 4);
var grid_Nor_Reality = new Float32Array(GRID_WIDTH * GRID_HEIGHT * 4);
var grid_Texcoord = new Float32Array(GRID_WIDTH * GRID_HEIGHT * 2);
var grid_Texcoord_Terrain = new Float32Array(GRID_WIDTH * GRID_HEIGHT * 2);
var grid_Normals = new Float32Array(GRID_WIDTH * GRID_HEIGHT * 3);
var grid_Index = new Uint16Array((GRID_WIDTH - 1) * (GRID_HEIGHT - 1) * 6);


var grid_Texture1;
var grid_Texture2;
var grid_Texture3;
var grid_Tex1_Sampler;
var grid_Tex2_Sampler;
var grid_Tex3_Sampler;


//For Transform Feedback
var vbo_temp_position;


//For Single Light Uniform
var la_Uniform;
var ld_Uniform;
var ls_Uniform;
var lightPosition_Uniform;

var ka_Uniform;
var kd_Uniform;
var ks_Uniform;
var shininess_Uniform;

//For Lights
var lightAmbient = [0.1, 0.1, 0.1];
var lightDiffuse =[1.0, 1.0, 1.0];
var lightSpecular = [1.0, 1.0, 1.0];
var lightPosition = [300.0, 400.0, 300.0, 1.0];
var bLights = false;


//For Material
var materialAmbient = [0.0, 0.0, 0.0];
var materialDiffuse = [1.0, 1.0, 1.0];
var materialSpecular = [1.0, 1.0, 1.0];
var materialShininess = 50.0;


//For Multiple Point Light
const TOTAL_POINT_LIGHT = 10;
var pointLight_Position =  new Float32Array(TOTAL_POINT_LIGHT * 4);
var pointLight_Ld = new Float32Array(TOTAL_POINT_LIGHT * 3);
var pointLight_Ls = new Float32Array(TOTAL_POINT_LIGHT * 3);

var pointLightPosUniform;
var pointLightLdUniform;
var pointLightLsUniform;



//For Tree Height
var gMemoryModelPos;
var gRealityModelPos;


function initialize_Terrain(){


	
	vertexShaderObject_Grid = gl.createShader(gl.VERTEX_SHADER);

	var szVertexShaderSourceCode = 
		"#version 300 es" +
		"\n" +


		"precision highp int;" +


		"in vec4 vPosition;" +
		"in vec4 vNormal;" +
		"in vec2 vTexcoord;" +
		"in vec2 vTexcoord_Terrain;" +


		"uniform mat4 u_model_matrix;" +
		"uniform mat4 u_view_matrix;" +
		"uniform mat4 u_projection_matrix;" +


		//Common in Lights
		"out vec3 outViewer;" +
		"out vec3 outNormal;" +

		//For Single Light
		"uniform vec4 u_light_position;" +
		"out vec3 outLightDirection;" +
		 

		//For Multiple Point Light
		"uniform vec4 u_point_light_pos[10];" +
		"out vec3 out_pointLightDir[10];" +
		"out vec3 out_pointLight_Normal;" +
		"out vec3 out_pointLight_viewerVec;" +
		// "uniform vec3 u_cam_pos;" +
		 

		
		//For HeightMap
		"out vec2 outTexcoord;" +

		//For Multitexture
		"out vec2 outTexcoord_Terrain;" +
		"uniform sampler2D u_heightmap_sampler;" +

		"out vec4 outHeight;" +
		"out float out_fHeight;" +


		//For Terrain 
		"uniform int u_choice;" +
		"flat out int out_u_choice;" +
		"const int MEMORY = 3;" +
		"const int REALITY = 4;" +


		//For Light & Shadow
		"uniform int u_shadow;" +
		"flat out int out_u_shadow;" +
		"const int DEPTH_MAP = 0;" +
		"const int SHADOW = 1;" +
		"const int SINGLE_LIGHT = 2;" +
		"const int FOG = 3;" +
		"const int POINT_LIGHT = 4;" +
		

		"out vec4 out_pos_normal;" +


		
		//For Shadow
		"uniform mat4 u_shadowMatrix;"+
		"out vec4 out_shadowCoord;" +


		//For Fog
		"out float out_fogCoord;" +



		"void main(){" +

			"vec4 newPos = vPosition;" +


			"if(u_choice == MEMORY){" +
				"vec2 v2center = vec2(0.5f);" +

				"vec4 pos = texture(u_heightmap_sampler, vTexcoord);" +
				"float height = pos.x;" +

				"newPos = vec4(vPosition.xyz, 1.0f);" +
				"float distance = length(vTexcoord - v2center);" +



				"float scale = distance;" +
				"if(distance > 0.5)" +
					"scale = 0.5;" +

				"height *= 1000.0f * scale;" +
			


				"newPos.y = height;" +
				"outHeight = u_model_matrix * newPos;" +
				"out_fHeight = pos.x;" +

			"}" +

			"else if(u_choice == REALITY){" +

				"vec4 pos = texture(u_heightmap_sampler, vTexcoord);" +
				"float height = (pos.x + pos.y + pos.z) / 3.0;" +

				"newPos = vec4(vPosition.xyz, 1.0f);" +
				"newPos.y = height * 3.80f;" +

				"outHeight = u_model_matrix * newPos;" +
				"out_fHeight = pos.x;" +

			"}" +


	




			"vec4 worldCoord = u_model_matrix * newPos;" +
			"vec4 eyeCoordinate = u_view_matrix * worldCoord;" +
			
			"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" +
			"outNormal = vec3(normalMatrix * vec3(vNormal));" +
			"outViewer = vec3(-eyeCoordinate);" +


			//For Shadow
			"if(u_shadow == SHADOW){" +
				"outLightDirection = vec3(u_light_position - eyeCoordinate);" +
				"out_shadowCoord = u_shadowMatrix * u_model_matrix * newPos;"+
			"}" +
			
			//For Single Light
			"else if(u_shadow == SINGLE_LIGHT){" +
				"outLightDirection = vec3(u_light_position - eyeCoordinate);" +
			"}" +


			//For Multiple Point Light
			"else if(u_shadow == POINT_LIGHT){" +


				"out_pointLight_Normal = vec3(mat3(u_model_matrix) * vNormal.xyz);" +
				"out_pointLight_viewerVec = vec3(-eyeCoordinate);" + 
 
				"for(int i = 0; i < 10; i++){" +
					"out_pointLightDir[i] = vec3(u_point_light_pos[i] - worldCoord);" +
				"}" +

				//For Single Light
				"outLightDirection = vec3(u_light_position - eyeCoordinate);" +

			"}" +
			
			

			//For Transform Feedback
			"out_pos_normal = vec4(newPos.xyz, 0.82f);" +

			"out_u_choice = u_choice;" +
			"out_u_shadow = u_shadow;" +

			//For Fog
			"out_fogCoord = abs(eyeCoordinate.z);" +

			//For Texture
			"outTexcoord_Terrain = vTexcoord_Terrain;" +
			"outTexcoord = vTexcoord;" +	

			"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * newPos;" +
			// "gl_PointSize = 10.0f;" +

		"}";

	gl.shaderSource(vertexShaderObject_Grid, szVertexShaderSourceCode);
	gl.compileShader(vertexShaderObject_Grid);

	var shaderCompileStatus = gl.getShaderParameter(vertexShaderObject_Grid, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(vertexShaderObject_Grid);
		if(error.length > 0){
			alert("03-Terrain : VertexShader Compilation Error : " + error);
			uninitialize();
			window.close();

		}
	}


	fragmentShaderObject_Grid = gl.createShader(gl.FRAGMENT_SHADER);
	var szFragmentShaderSourceCode = 
		"#version 300 es" +
		"\n" +
		"precision highp float;" +
		"precision highp sampler2D;" +
		"precision highp sampler2DShadow;" +

		"in vec2 outTexcoord;" +
		"in vec2 outTexcoord_Terrain;" +
		"in float out_fHeight;" +
		"in vec4 outHeight;" +
		

		//For Terrain Multitexture
		"uniform sampler2D u_heightmap_sampler;" +
		"uniform sampler2D u_grid_tex1;" +
		"uniform sampler2D u_grid_tex2;" +
		"uniform sampler2D u_grid_tex3;" +

		
		//Toggle for Reflection and Refraction Texture
		"uniform int u_Reflection_Refraction_Choice;" +
		"uniform float u_waterLevel;" +
		"uniform vec4 u_clipPlane;" +


		"flat in int out_u_choice;" +
		"const int MEMORY = 3;" +
		"const int REALITY = 4;" +

		"flat in int out_u_shadow;" +
		"const int DEPTH_MAP = 0;" +
		"const int SHADOW = 1;" +
		"const int SINGLE_LIGHT = 2;" +
		"const int FOG = 3;" +
		"const int POINT_LIGHT = 4;" +
  

		//For Shadow
		 "uniform sampler2DShadow u_depthTextureSampler;"+
		 "in vec4 out_shadowCoord;"+

		 //Common in both lights
		"in vec3 outNormal;" +
		"in vec3 outViewer;" +

		//For Single Lights
		"in vec3 outLightDirection;" +

		"uniform vec3 u_la;" +
		"uniform vec3 u_ld;" +
		"uniform vec3 u_ls;" +

		
		"uniform vec3 u_ka;" +
		"uniform vec3 u_kd;" +
		"uniform vec3 u_ks;" +
		"uniform float u_shininess;" +

		
		//For Multiple Point Light
		"in vec3 out_pointLightDir[10];" +
		"uniform vec3 u_pl_ld[10];" +
		"uniform vec3 u_pl_ls[10];" +
		"in vec3 out_pointLight_Normal;" +
		"in vec3 out_pointLight_viewerVec;" +





		//For Fog
		"in float out_fogCoord;" +
 
		"out vec4 FragColor;" +



		"void main() {" +

			"vec4 texColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);" +
			"vec3 PhongLight = vec3(0.0f);" +


			"if(out_u_choice == MEMORY){" +


				//Water Level
				"const float fRange1 = 0.02f;" +
				"const float fRange2 = 0.08f;" +
				"const float fRange3 = 0.12f;" +

				//Above Water Level	
				"const float fRange4 = 0.55f;" +	


				"vec2 outTex = outTexcoord_Terrain;" +

				// "float fScale = outHeight.y;" +
				"float fScale = out_fHeight;" +


				"switch(u_Reflection_Refraction_Choice){" +

					//For Reflection Tex
					"case 1:" +
					"case 2:" +

						"if(dot(outHeight, u_clipPlane) < 0.0f){" +
							"discard;" +
						"}" +	

						"break;" +


				"}" +


				//For Water Depth
				"if(fScale <= fRange2){" +

					"fScale = fScale - fRange1;" +
					"fScale /= (fRange2 - fRange1);" +

					"float fScale2 = fScale;" +
					"fScale = 1.0f - fScale;" +

					"texColor += texture(u_grid_tex3, outTex) * fScale2;" +

				"}" +


				"else if(fScale <= fRange3){" +

					"fScale = fScale - fRange2;" +
					"fScale /= (fRange3 - fRange2);" +

					"float fScale2 = fScale;" +
					"fScale = 1.0f - fScale;" +	

					"texColor = texture(u_grid_tex2, outTex) * fScale2;" +
					"texColor += texture(u_grid_tex3, outTex) * fScale;" +
				"}" +


				//For Mountain above Water
				"else if(fScale <= fRange4){" +

					"fScale = fScale - fRange3;" +
					"fScale /= (fRange4 - fRange3);" +

					"float fScale2 = fScale;" +
					"fScale = 1.0f - fScale;" +

					"texColor = texture(u_grid_tex3, outTex) * fScale2;" +
					"texColor += texture(u_grid_tex2, outTex) * fScale;" +

				"}" +

				"else{" +
					"texColor = texture(u_grid_tex3, outTex);" +
				"}" +	
				
			"}" +


			"else if(out_u_choice == REALITY){" +


				
				"const float fRange3 = 1.6f;" +
				"const float fRange4 = 1.7f;" +

				"vec2 outTex = outTexcoord_Terrain;" +

				"float fScale = outHeight.y;" +

			

				"if(fScale <= fRange3){" +
					"texColor = texture(u_grid_tex1, outTex);" +
				"}" +
				
				"else if(fScale <= fRange4){" +

					"fScale = fScale - fRange3;" +
					"fScale /= (fRange4 - fRange3);" +

					"float fScale2 = fScale;" +
					"fScale = 1.0f - fScale;" +

					"texColor = texture(u_grid_tex1, outTex) * fScale;" +
					"texColor += texture(u_grid_tex3, outTex) * fScale2;" +

				"}" +

				"else{" +
					"texColor = texture(u_grid_tex3, outTex);" +
				"}" +


			"}" +

			
			//For Depth Map
			"if(out_u_shadow == DEPTH_MAP){" +
				"FragColor = vec4(1.0f);" +
			"}" +



			//For Shadow
			"else if(out_u_shadow == SHADOW){" +
				"vec3 normalizeLightDirection = normalize(outLightDirection);" +
				"vec3 normalizeNormalVector = normalize(outNormal);" +
				"float S_Dot_N = max(dot(normalizeLightDirection, normalizeNormalVector), 0.0);" +

				"vec3 normalizeViewer = normalize(outViewer);" +
				"vec3 reflection = reflect(-normalizeLightDirection, normalizeNormalVector);" +
				"float R_Dot_V = max(dot(reflection, normalizeViewer), 0.0);" +

				"float f = textureProj(u_depthTextureSampler, out_shadowCoord);"+
				
				"vec3 ambient = f * u_la * u_ka;" +
				"vec3 diffuse = f * u_ld * u_kd * S_Dot_N;" +
				"vec3 specular = f * u_ls * u_ks * pow(R_Dot_V, u_shininess);" +
				"PhongLight = ambient + diffuse;" +
				"FragColor = texColor * vec4(PhongLight, 1.0f);" +

				// // *** For Fog ***
				// "float fog = exp(-0.001 * out_fogCoord);" +

				// "vec3 color = mix(vec3(1.0, 1.0f, 1.0f), FragColor.xyz, fog);" +

				// "FragColor = vec4(color, 1.0);" +

			"}" +

			
			//For Normal Single Light
			"else if(out_u_shadow == SINGLE_LIGHT){" +

				"vec3 normalizeLightDirection = normalize(outLightDirection);" +
				"vec3 normalizeNormalVector = normalize(outNormal);" +
				"float S_Dot_N = max(dot(normalizeLightDirection, normalizeNormalVector), 0.0);" +

				"vec3 normalizeViewer = normalize(outViewer);" +
				"vec3 reflection = reflect(-normalizeLightDirection, normalizeNormalVector);" +
				"float R_Dot_V = max(dot(reflection, normalizeViewer), 0.0);" +

				"vec3 ambient = u_la * u_ka;" +
				"vec3 diffuse =  u_ld * u_kd * S_Dot_N;" +
				"vec3 specular = u_ls * u_ks * pow(R_Dot_V, u_shininess);" +
				"PhongLight = ambient + diffuse ;" +
				"FragColor = texColor * vec4(PhongLight, 0.0f);" +

			"}" +

			//For Fog
			"else if(out_u_shadow == FOG){" +

				"vec3 normalizeLightDirection = normalize(outLightDirection);" +
				"vec3 normalizeNormalVector = normalize(outNormal);" +
				"float S_Dot_N = max(dot(normalizeLightDirection, normalizeNormalVector), 0.0);" +

				"vec3 normalizeViewer = normalize(outViewer);" +
				"vec3 reflection = reflect(-normalizeLightDirection, normalizeNormalVector);" +
				"float R_Dot_V = max(dot(reflection, normalizeViewer), 0.0);" +

				"vec3 ambient = u_la * u_ka;" +
				"vec3 diffuse =  u_ld * u_kd * S_Dot_N;" +
				"vec3 specular = u_ls * u_ks * pow(R_Dot_V, u_shininess);" +
				"PhongLight = ambient + diffuse ;" +
				"FragColor = texColor * vec4(PhongLight, 0.0f);" +

				// *** For Fog ***
				"float f = exp(-0.001 * out_fogCoord);" +

				"vec3 color = mix(vec3(1.0, 1.0f, 1.0f), FragColor.xyz, f);" +

				"FragColor = vec4(color, 1.0);" +

			"}" +


			//For Normal Single Light
			"else if(out_u_shadow == POINT_LIGHT){" +

				"PhongLight = vec3(0.0f);" +
				"vec3 normalizeNormalVector = normalize(out_pointLight_Normal);" +
				"vec3 normalizeViewer = normalize(out_pointLight_viewerVec);" +

				"const float ConstAtt = 1.0f * 0.10;" +
				"const float LinAtt = 0.01f * 0.10;" +
				"const float QuadAtt = 0.002f * 0.10f;" +
				"float attenuation;" +

				"for(int i = 0; i < 10; i++){" +

					"vec3 normalizeLightDirection = normalize(out_pointLightDir[i]);" +
					"float S_Dot_N = max(dot(normalizeLightDirection, normalizeNormalVector), 0.0);" +

					"vec3 reflection = reflect(-normalizeLightDirection, normalizeNormalVector);" +
					"float R_Dot_V = max(dot(reflection, normalizeViewer), 0.0);" +

					"float distance = length(out_pointLightDir[i]);" +
					"attenuation = 1.0f / (ConstAtt + LinAtt * distance + QuadAtt * distance * distance);" +


					"vec3 ambient = u_la * u_ka * attenuation;" +
					"vec3 diffuse =  u_pl_ld[i] * u_kd * S_Dot_N * attenuation;" +
					"vec3 specular = u_pl_ls[i] * u_ks * pow(R_Dot_V, u_shininess) * attenuation;" +
					"PhongLight += ambient + diffuse;" +

					// "FragColor = vec4(diffuse, 1.0f);" +

				"}" +



				// Global Single Light
				"vec3 normalizeLightDirection = normalize(outLightDirection);" +
				"normalizeNormalVector = normalize(outNormal);" +
				"float S_Dot_N = max(dot(normalizeLightDirection, normalizeNormalVector), 0.0);" +

				"normalizeViewer = normalize(outViewer);" +
				"vec3 reflection = reflect(-normalizeLightDirection, normalizeNormalVector);" +
				"float R_Dot_V = max(dot(reflection, normalizeViewer), 0.0);" +

				"vec3 ambient = u_la * u_ka;" +
				"vec3 diffuse =  u_ld * u_kd * S_Dot_N;" +
				"vec3 specular = u_ls * u_ks * pow(R_Dot_V, u_shininess);" +
				"PhongLight += ambient + diffuse;" +





				"FragColor = texColor * vec4(PhongLight, 1.0f);" +





				// "FragColor = vec4(u_pl_ld[0], 1.0f);" +
				// "FragColor = vec4(attenuation);" +

			"}" +

			
	
		
		"}";

	gl.shaderSource(fragmentShaderObject_Grid, szFragmentShaderSourceCode);
	gl.compileShader(fragmentShaderObject_Grid);

	shaderCompileStatus = gl.getShaderParameter(fragmentShaderObject_Grid,  gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(fragmentShaderObject_Grid);
		if(error.length > 0){
			alert("03-Terrain : Fragment Shader Compilation Error : " + error);
			uninitialize();
			window.close();

		}
	}



	shaderProgramObject_Grid = gl.createProgram();

	gl.attachShader(shaderProgramObject_Grid, vertexShaderObject_Grid);
	gl.attachShader(shaderProgramObject_Grid, fragmentShaderObject_Grid);

	gl.bindAttribLocation(shaderProgramObject_Grid, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
	gl.bindAttribLocation(shaderProgramObject_Grid, WebGLMacros.AMC_ATTRIBUTE_NORMAL, "vNormal");
	gl.bindAttribLocation(shaderProgramObject_Grid, WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0, "vTexcoord");
	gl.bindAttribLocation(shaderProgramObject_Grid, WebGLMacros.AMC_ATTRIBUTE_TEXCOORD1, "vTexcoord_Terrain");

	gl.linkProgram(shaderProgramObject_Grid);

	var programLinkStatus = gl.getProgramParameter(shaderProgramObject_Grid, gl.LINK_STATUS);

	if(programLinkStatus == false){
		var error = gl.getProgramInfoLog(shaderProgramObject_Grid);
		if(error.length > 0){
			alert("03-Terrain : Program Linking Error : " + error);
			uninitialize();
			window.close();
		}
	}




	// ********** Transform Feedback **********
	var varying = ['out_pos_normal'];

	gl.transformFeedbackVaryings(shaderProgramObject_Grid, varying, gl.INTERLEAVED_ATTRIBS);


	gl.linkProgram(shaderProgramObject_Grid);

	programLinkStatus = gl.getProgramParameter(shaderProgramObject_Grid, gl.LINK_STATUS);

	if(programLinkStatus == false){
		var error = gl.getProgramInfoLog(shaderProgramObject_Grid);
		if(error.length > 0){
			alert("03-Terrain : Program Linking Error Transform Feedback : " + error);
			uninitialize();
			window.close();
		}
	}



	modelMatUniform = gl.getUniformLocation(shaderProgramObject_Grid, "u_model_matrix");
	viewMatUniform = gl.getUniformLocation(shaderProgramObject_Grid, "u_view_matrix");
	projMatUniform = gl.getUniformLocation(shaderProgramObject_Grid, "u_projection_matrix");
	grid_choiceUniform = gl.getUniformLocation(shaderProgramObject_Grid, "u_choice");

	grid_HeightMap_Sampler = gl.getUniformLocation(shaderProgramObject_Grid, "u_heightmap_sampler");
	grid_Tex1_Sampler = gl.getUniformLocation(shaderProgramObject_Grid, "u_grid_tex1");
	grid_Tex2_Sampler = gl.getUniformLocation(shaderProgramObject_Grid, "u_grid_tex2");
	grid_Tex3_Sampler = gl.getUniformLocation(shaderProgramObject_Grid, "u_grid_tex3");

	gChoice_Reflection_Refraction_Uniform = gl.getUniformLocation(shaderProgramObject_Grid, "u_Reflection_Refraction_Choice");
	gWaterLevelUniform = gl.getUniformLocation(shaderProgramObject_Grid, "u_waterLevel");
	clipPlaneUniform = gl.getUniformLocation(shaderProgramObject_Grid, "u_clipPlane");

	//For Single Global Light
	la_Uniform = gl.getUniformLocation(shaderProgramObject_Grid, "u_la");
	ld_Uniform = gl.getUniformLocation(shaderProgramObject_Grid, "u_ld");
	ls_Uniform = gl.getUniformLocation(shaderProgramObject_Grid, "u_ls");
	lightPosition_Uniform = gl.getUniformLocation(shaderProgramObject_Grid, "u_light_position");

	ka_Uniform = gl.getUniformLocation(shaderProgramObject_Grid, "u_ka");
	kd_Uniform = gl.getUniformLocation(shaderProgramObject_Grid, "u_kd");
	ks_Uniform = gl.getUniformLocation(shaderProgramObject_Grid, "u_ks");
	shininess_Uniform = gl.getUniformLocation(shaderProgramObject_Grid, "u_shininess");

	//For Shadow
	shadowChoiceUniform = gl.getUniformLocation(shaderProgramObject_Grid, "u_shadow");
	shadowMatUniform = gl.getUniformLocation(shaderProgramObject_Grid, "u_shadowMatrix");
	depthMapSamplerUniform = gl.getUniformLocation(shaderProgramObject_Grid, "u_depthTextureSampler");

	//For Point Light
	pointLightPosUniform = gl.getUniformLocation(shaderProgramObject_Grid, "u_point_light_pos");
	pointLightLdUniform = gl.getUniformLocation(shaderProgramObject_Grid, "u_pl_ld");
	pointLightLsUniform = gl.getUniformLocation(shaderProgramObject_Grid, "u_pl_ls");


	LoadGrid(grid_Pos_Memory, -1024.0, -1024.0);
	LoadGrid(grid_Pos_Reality, -900.0, -900.0);

	vao_Grid = gl.createVertexArray();
	gl.bindVertexArray(vao_Grid);

		/********** Memory Position **********/
		vbo_Grid_Pos_Memory = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Grid_Pos_Memory);
		gl.bufferData(gl.ARRAY_BUFFER, grid_Pos_Memory, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,  3, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Reality Position **********/
		vbo_Grid_Pos_Reality = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Grid_Pos_Reality);
		gl.bufferData(gl.ARRAY_BUFFER, grid_Pos_Reality, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,  3, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Memory Normals **********/
		vbo_Grid_Normals_Memory = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Grid_Normals_Memory);
		gl.bufferData(gl.ARRAY_BUFFER, grid_Nor_Memory, gl.DYNAMIC_COPY);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_NORMAL, 4, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_NORMAL);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Reality Normals **********/
		vbo_Grid_Normals_Reality = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Grid_Normals_Reality);
		gl.bufferData(gl.ARRAY_BUFFER, grid_Nor_Reality, gl.DYNAMIC_COPY);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_NORMAL, 4, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_NORMAL);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Texcoord **********/
		vbo_Grid_Texcoord = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Grid_Texcoord);
		gl.bufferData(gl.ARRAY_BUFFER, grid_Texcoord, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0, 2, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Texcoord For Terrain **********/
		vbo_Grid_Texcoord_Terrain = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Grid_Texcoord_Terrain);
		gl.bufferData(gl.ARRAY_BUFFER, grid_Texcoord_Terrain, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD1, 2, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD1);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Elements **********/
		vbo_Grid_Index = gl.createBuffer();
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vbo_Grid_Index);
		gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, grid_Index, gl.STATIC_DRAW);
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);


	gl.bindVertexArray(null);






	//Load Texture
	grid_Texture1 = gl.createTexture();
	grid_Texture1.image = new Image();
	grid_Texture1.image.src = "00-Textures/grass_wet.png";
	grid_Texture1.image.onload = function(){
		gl.bindTexture(gl.TEXTURE_2D, grid_Texture1);
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
		gl.texImage2D(gl.TEXTURE_2D,
					0,
					gl.RGBA,
					gl.RGBA,
					gl.UNSIGNED_BYTE,
					grid_Texture1.image);
		gl.generateMipmap(gl.TEXTURE_2D);
		gl.bindTexture(gl.TEXTURE_2D, null);
	}


	//Load Texture
	grid_Texture2 = gl.createTexture();
	grid_Texture2.image = new Image();
	grid_Texture2.image.src = "00-Textures/grass_evening128.png";
	grid_Texture2.image.onload = function(){
		gl.bindTexture(gl.TEXTURE_2D, grid_Texture2);
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
		gl.texImage2D(gl.TEXTURE_2D,
					0,
					gl.RGBA,
					gl.RGBA,
					gl.UNSIGNED_BYTE,
					grid_Texture2.image);
		gl.generateMipmap(gl.TEXTURE_2D);
		gl.bindTexture(gl.TEXTURE_2D, null);
	}

	//Load Texture
	grid_Texture3 = gl.createTexture();
	grid_Texture3.image = new Image();
	grid_Texture3.image.src = "00-Textures/rock128.png";
	grid_Texture3.image.onload = function(){
		gl.bindTexture(gl.TEXTURE_2D, grid_Texture3);
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
		gl.texImage2D(gl.TEXTURE_2D,
					0,
					gl.RGBA,
					gl.RGBA,
					gl.UNSIGNED_BYTE,
					grid_Texture3.image);
		gl.generateMipmap(gl.TEXTURE_2D);
		gl.bindTexture(gl.TEXTURE_2D, null);
	}


	console.log("03-Terrain: Initialize Complete");
}


function uninitialize_Terrain(){

	if(grid_Texture1){
		gl.deleteTexture(grid_Texture1);
		grid_Texture1 = null;
	}

	if(grid_Texture2){
		gl.deleteTexture(grid_Texture2);
		grid_Texture2 = null;
	}

	if(grid_Texture3){
		gl.deleteTexture(grid_Texture3);
		grid_Texture3 = null;
	}

	if(vbo_Grid_Index){
		gl.deleteBuffer(vbo_Grid_Index);
		vbo_Grid_Index = null;
	}

	if(vbo_Grid_Texcoord_Terrain){
		gl.deleteBuffer(vbo_Grid_Texcoord_Terrain);
		vbo_Grid_Texcoord_Terrain = null;
	}

	if(vbo_Grid_Texcoord){
		gl.deleteBuffer(vbo_Grid_Texcoord);
		vbo_Grid_Texcoord = null;
	}


	if(vbo_Grid_Pos_Reality){
		gl.deleteBuffer(vbo_Grid_Pos_Reality);
		vbo_Grid_Pos_Reality = null;
	}

	if(vbo_Grid_Pos_Memory){
		gl.deleteBuffer(vbo_Grid_Pos_Memory);
		vbo_Grid_Pos_Memory = null;
	}

	if(vao_Grid){
		gl.deleteVertexArray(vao_Grid);
		vao_Grid = null;
	}

	if(shaderProgramObject_Grid){

		gl.useProgram(shaderProgramObject_Grid);

			if(fragmentShaderObject_Grid){
				gl.detachShader(shaderProgramObject_Grid, fragmentShaderObject_Grid);
				gl.deleteShader(fragmentShaderObject_Grid);
				fragmentShaderObject_Grid = null;
			}

			if(vertexShaderObject_Grid){
				gl.detachShader(shaderProgramObject_Grid, vertexShaderObject_Grid);
				gl.deleteShader(vertexShaderObject_Grid);
				vertexShaderObject_Grid = null;
			}

		gl.useProgram(null);
		gl.deleteProgram(shaderProgramObject_Grid);
		shaderProgramObject_Grid = null;
	}

	console.log("03-Terrain: Uninitialize Complete");
}



var FlagForLoadGrid = false;
function LoadGrid(pos, fStartX, fStartZ){


	var fX = fStartX;
	var fZ = fStartZ;

	var tX = 0.0;
	var tZ = 1.0;

	var tX1 = gGrid_TexX;
	var tZ1 = gGrid_TexZ;

	
	//For Position
	for(var i = 0; i < GRID_HEIGHT; i++){
	
		fX = fStartX;
		for(var j = 0; j < GRID_WIDTH; j++){

			pos[(i * 3 * GRID_WIDTH) + (j * 3) + 0] = fX;
			pos[(i * 3 * GRID_WIDTH) + (j * 3) + 1] = 0.0;
			pos[(i * 3 * GRID_WIDTH) + (j * 3) + 2] = fZ;

			fX = fX + ((2.0 * Math.abs(fStartX)) / (GRID_WIDTH - 1));
		}
		fZ = fZ + ((2.0 * Math.abs(fStartZ)) / (GRID_HEIGHT - 1));
	}


	if(FlagForLoadGrid == false){

		//For Texture Loop
		for(var i = 0; i < GRID_HEIGHT; i++){
	
			tX = 0.0;
			tX1 = gGrid_TexX;

			for(var j = 0; j < GRID_WIDTH; j++){
				

				grid_Texcoord[(i * 2 * GRID_WIDTH) + (j * 2) + 0] = tX;
				grid_Texcoord[(i * 2 * GRID_WIDTH) + (j * 2) + 1] = tZ;

				grid_Texcoord_Terrain[(i * 2 * GRID_WIDTH) + (j * 2) + 0] = tX1;
				grid_Texcoord_Terrain[(i * 2 * GRID_WIDTH) + (j * 2) + 1] = tZ1;

		
				tX = tX + (1.0 / (GRID_WIDTH - 1));
				tX1 = tX1 + (gGrid_TexZ / (GRID_WIDTH - 1));

			}

			tZ = tZ - (1.0 / (GRID_HEIGHT - 1));
			tZ1 = tZ1 - (gGrid_TexZ / (GRID_HEIGHT - 1));
		}



		var index = 0;

		for(var i = 0; i < (GRID_HEIGHT - 1); i++){

			for(var j = 0; j < (GRID_WIDTH - 1); j++){

				var topLeft = (i * GRID_WIDTH) + j;
				var bottomLeft = ((i + 1) * GRID_WIDTH) + j;
				var topRight = topLeft + 1;
				var bottomRight = bottomLeft + 1;

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

	}

	console.log("gGrid_NumOfElements: ", gGrid_NumOfElements);
	FlagForLoadGrid = true;
}




function draw_terrain(mode, choice, texture, lightChoice, clipPlane, points){

	var translateMatrix = mat4.create();
	var modelMatrix = mat4.create();

	gl.useProgram(shaderProgramObject_Grid);


	mat4.identity(modelMatrix);
	mat4.identity(translateMatrix);
	mat4.identity(global_viewMatrix);

	
	switch(choice){

		//Reflection	
		case SceneConst.REFLECT_TEX:

			var ht = (gCameraPosition[1] - gWater_Level);

			// console.log("1: ", gCameraFront);
			gCameraPosition[1] -= (2.0 * ht);
			invertPitchAngle(-gPitchAngle);

			// console.log("2: ", gCameraFront);
			
			vec3.add(gCameraLookingAt, gCameraPosition, gCameraFront);
			mat4.lookAt(global_viewMatrix, gCameraPosition, gCameraLookingAt, gCameraUp);
			
			gCameraPosition[1] += (2.0 * ht);
			invertPitchAngle(gPitchAngle);

			break;


		default:
			vec3.add(gCameraLookingAt, gCameraPosition, gCameraFront);
			mat4.lookAt(global_viewMatrix, gCameraPosition, gCameraLookingAt, gCameraUp);
			break;


	}
	

	//console.log(global_viewMatrix)

	gl.uniformMatrix4fv(modelMatUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatUniform, false, global_viewMatrix);
	gl.uniformMatrix4fv(projMatUniform, false, gPerspectiveProjectionMatrix);


	//For Reflection Reflection and Normal mode Tooggleing
	gl.uniform1i(gChoice_Reflection_Refraction_Uniform, choice);
	gl.uniform4fv(clipPlaneUniform, clipPlane);

	gl.uniform1f(gWaterLevelUniform, gWater_Level);
	gl.uniform1i(grid_choiceUniform, mode);
	gl.uniform1i(shadowChoiceUniform, lightChoice);


	//For Multitexture and HeightMap
	gl.activeTexture(gl.TEXTURE0);
	gl.bindTexture(gl.TEXTURE_2D, texture);
	gl.uniform1i(grid_HeightMap_Sampler, 0);

	gl.activeTexture(gl.TEXTURE1);
	gl.bindTexture(gl.TEXTURE_2D, grid_Texture1);
	gl.uniform1i(grid_Tex1_Sampler, 1);

	gl.activeTexture(gl.TEXTURE2);
	gl.bindTexture(gl.TEXTURE_2D, grid_Texture2);
	gl.uniform1i(grid_Tex2_Sampler, 2);

	gl.activeTexture(gl.TEXTURE3);
	gl.bindTexture(gl.TEXTURE_2D, grid_Texture3);
	gl.uniform1i(grid_Tex3_Sampler, 3);

	gl.activeTexture(gl.TEXTURE4);
	gl.bindTexture(gl.TEXTURE_2D, null);
	gl.uniform1i(depthMapSamplerUniform, 4);


	switch(lightChoice){

		case Lights.SHADOW:
		case Lights.SINGLE_LIGHT:
			
			//For Lights
			gl.uniform3fv(la_Uniform, lightAmbient);
			gl.uniform3fv(ld_Uniform, lightDiffuse);
			gl.uniform3fv(ls_Uniform, lightSpecular);
			gl.uniform4fv(lightPosition_Uniform, lightPosition);

			gl.uniform3fv(ka_Uniform, materialAmbient);
			gl.uniform3fv(kd_Uniform, materialDiffuse);
			gl.uniform3fv(ks_Uniform, materialSpecular);
			gl.uniform1f(shininess_Uniform, materialShininess);	
			break;

		case Lights.POINT_LIGHT:

			gl.uniform3fv(la_Uniform, lightAmbient);
			gl.uniform3fv(ld_Uniform, [0.2, 0.2, 0.2]);
			gl.uniform3fv(ls_Uniform, [0.2, 0.2, 0.2]);
			gl.uniform4fv(lightPosition_Uniform, lightPosition);


			gl.uniform3fv(pointLightLdUniform, pointLight_Ld);
			gl.uniform3fv(pointLightLsUniform, pointLight_Ls);
			gl.uniform4fv(pointLightPosUniform, pointLight_Position);

			// gl.uniform3fv(gl.getUniformLocation(shaderProgramObject_Grid, "u_cam_pos"), gCameraPosition);


			gl.uniform3fv(ka_Uniform, materialAmbient);
			gl.uniform3fv(kd_Uniform, materialDiffuse);
			gl.uniform3fv(ks_Uniform, materialSpecular);
			gl.uniform1f(shininess_Uniform, materialShininess);	

			break;


	}
	



	gl.bindVertexArray(vao_Grid);


		if(mode == SceneConst.MEMORY){

			gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Grid_Pos_Memory);
			gl.bufferData(gl.ARRAY_BUFFER, grid_Pos_Memory, gl.STATIC_DRAW);
			gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,  3, gl.FLOAT, false, 0, 0);
			gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
			gl.bindBuffer(gl.ARRAY_BUFFER, null);


			gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Grid_Normals_Memory);
			gl.bufferData(gl.ARRAY_BUFFER, grid_Nor_Memory, gl.STATIC_DRAW);
			gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_NORMAL,  4, gl.FLOAT, false, 0, 0);
			gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_NORMAL);
			gl.bindBuffer(gl.ARRAY_BUFFER, null);

			gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vbo_Grid_Index);
			gl.drawElements(gl.TRIANGLES, gGrid_NumOfElements, gl.UNSIGNED_SHORT, 0);
			gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);


		}
		else{
			gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Grid_Pos_Reality);
			gl.bufferData(gl.ARRAY_BUFFER, grid_Pos_Reality, gl.STATIC_DRAW);
			gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,  3, gl.FLOAT, false, 0, 0);
			gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
			gl.bindBuffer(gl.ARRAY_BUFFER, null);


			gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Grid_Normals_Reality);
			gl.bufferData(gl.ARRAY_BUFFER, grid_Nor_Reality, gl.STATIC_DRAW);
			gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_NORMAL,  4, gl.FLOAT, false, 0, 0);
			gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_NORMAL);
			gl.bindBuffer(gl.ARRAY_BUFFER, null);

			gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vbo_Grid_Index);
			gl.drawElements(gl.TRIANGLES, gGrid_NumOfElements, gl.UNSIGNED_SHORT, 0);
			gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
		}	




	gl.bindVertexArray(null);


	gl.activeTexture(gl.TEXTURE4);
	gl.bindTexture(gl.TEXTURE_2D, null);

	gl.activeTexture(gl.TEXTURE3);
	gl.bindTexture(gl.TEXTURE_2D, null);

	gl.activeTexture(gl.TEXTURE2);
	gl.bindTexture(gl.TEXTURE_2D, null);

	gl.activeTexture(gl.TEXTURE1);
	gl.bindTexture(gl.TEXTURE_2D, null);

	gl.activeTexture(gl.TEXTURE0);
	gl.bindTexture(gl.TEXTURE_2D, null);


	gl.useProgram(null);


	if(mode == SceneConst.MEMORY){
		drawModel(gModel_Lamp, gLampTexture_AAW, 1);	
		drawModel(gModel_Tree_Memory, gTreeTexture_AAW, 1);
	}
	else{
		drawModel(gModel_Tree_Reality, gTreeTexture_AAW, 1);
	}




	if(lightChoice != Lights.POINT_LIGHT){
		draw_CubeMap();
	}


}





function draw_terrain_with_shadow(mode, choice, texture, lightMode, end){

	
	// Pass 1 Depth Map 
	gl.bindFramebuffer(gl.FRAMEBUFFER, gFrameBuffer_DepthMap_AAW);

	gl.clearDepth(1.0);
        gl.clear(gl.DEPTH_BUFFER_BIT);
        
        gl.enable(gl.POLYGON_OFFSET_FILL);
        gl.polygonOffset(2.0, 4.0);

        	gl.viewport(0, 0, DEPTH_TEXTURE_SIZE_AAW, DEPTH_TEXTURE_SIZE_AAW);
	        //draw_terrain_depth(mode, choice, texture, 0);
	        
	        if(mode == SceneConst.MEMORY){
				drawModelWithShadow(gModel_Tree_Memory, gTreeTexture_AAW, 1, 0);
				drawModelWithShadow(gModel_Lamp, gLampTexture_AAW, 1, 0);
	        }
		else
			drawModelWithShadow(gModel_Tree_Reality, gTreeTexture_AAW, 1, 0);


	gl.disable(gl.POLYGON_OFFSET_FILL);

	gl.bindFramebuffer(gl.FRAMEBUFFER, null);



	//Pass 2 Use Depth Map for shadow
	gl.viewport(0, 0, canvas.width, canvas.height);

	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
	
	if(end == 0)
		draw_terrain_depth(mode, choice, texture, 1);
	
	if(mode == SceneConst.MEMORY){
		drawModelWithShadow(gModel_Tree_Memory, gTreeTexture_AAW,  1, 1);
		drawModelWithShadow(gModel_Lamp, gLampTexture_AAW, 1, 1);
	}
	else
		drawModelWithShadow(gModel_Tree_Reality, gTreeTexture_AAW, 1, 1);

	if(lightMode != Lights.POINT_LIGHT)
		draw_CubeMap();

}


function draw_terrain_depth(mode, choice, texture, shadow){

	var translateMatrix = mat4.create();
	var modelMatrix = mat4.create();

	var sceneModelMatrix = mat4.create();
	var sceneViewMatrix = mat4.create();
	var sceneProjectionMatrix = mat4.create();
	var shadowMatrix = mat4.create();

	var scaleBiasMatrix;



	mat4.identity(modelMatrix);
	mat4.identity(translateMatrix);


	var lightViewMatrix = mat4.create();
	var lightProjectionMatrix = mat4.create();
	var lightMVPMatrix = mat4.create();


	if(shadow == 0){

		mat4.lookAt(lightViewMatrix, gLightPosition_AAW, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
	        mat4.perspective(lightProjectionMatrix, 45.0, parseFloat(DEPTH_TEXTURE_SIZE_AAW) / parseFloat(DEPTH_TEXTURE_SIZE_AAW), 0.1, 4000.0);

	        mat4.multiply(lightMVPMatrix, lightMVPMatrix, lightProjectionMatrix);
	        mat4.multiply(lightMVPMatrix, lightMVPMatrix, lightViewMatrix);
		
		gl.useProgram(shaderProgramObject_Grid);

		gl.uniformMatrix4fv(modelMatUniform, false, modelMatrix);
		gl.uniformMatrix4fv(viewMatUniform, false, lightViewMatrix);
		gl.uniformMatrix4fv(projMatUniform, false, lightProjectionMatrix);


		//For Reflection Reflection and Normal mode Tooggleing
		gl.uniform1i(gChoice_Reflection_Refraction_Uniform, choice);
		gl.uniform1f(gWaterLevelUniform, gWater_Level);
		gl.uniform1i(grid_choiceUniform, mode);
		gl.uniform1i(shadowChoiceUniform, shadow);


		//For Multitexture and HeightMap
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.uniform1i(grid_HeightMap_Sampler, 0);

		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, grid_Texture1);
		gl.uniform1i(grid_Tex1_Sampler, 1);

		gl.activeTexture(gl.TEXTURE2);
		gl.bindTexture(gl.TEXTURE_2D, grid_Texture2);
		gl.uniform1i(grid_Tex2_Sampler, 2);

		gl.activeTexture(gl.TEXTURE3);
		gl.bindTexture(gl.TEXTURE_2D, grid_Texture3);
		gl.uniform1i(grid_Tex3_Sampler, 3);

		gl.activeTexture(gl.TEXTURE4);
		gl.bindTexture(gl.TEXTURE_2D, null);
		gl.uniform1i(depthMapSamplerUniform, 4);


		//For Lights
		gl.uniform3fv(la_Uniform, lightAmbient);
		gl.uniform3fv(ld_Uniform, lightDiffuse);
		gl.uniform3fv(ls_Uniform, lightSpecular);
		gl.uniform4fv(lightPosition_Uniform, lightPosition);

		gl.uniform3fv(ka_Uniform, materialAmbient);
		gl.uniform3fv(kd_Uniform, materialDiffuse);
		gl.uniform3fv(ks_Uniform, materialSpecular);
		gl.uniform1f(shininess_Uniform, materialShininess);	



		gl.bindVertexArray(vao_Grid);


			if(mode == SceneConst.MEMORY){

				gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Grid_Pos_Memory);
				gl.bufferData(gl.ARRAY_BUFFER, grid_Pos_Memory, gl.STATIC_DRAW);
				gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,  3, gl.FLOAT, false, 0, 0);
				gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
				gl.bindBuffer(gl.ARRAY_BUFFER, null);


				gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Grid_Normals_Memory);
				gl.bufferData(gl.ARRAY_BUFFER, grid_Nor_Memory, gl.STATIC_DRAW);
				gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_NORMAL,  4, gl.FLOAT, false, 0, 0);
				gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_NORMAL);
				gl.bindBuffer(gl.ARRAY_BUFFER, null);

				gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vbo_Grid_Index);
				gl.drawElements(gl.TRIANGLES, gGrid_NumOfElements, gl.UNSIGNED_SHORT, 0);
				gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);


			}
			else{
				gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Grid_Pos_Reality);
				gl.bufferData(gl.ARRAY_BUFFER, grid_Pos_Reality, gl.STATIC_DRAW);
				gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,  3, gl.FLOAT, false, 0, 0);
				gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
				gl.bindBuffer(gl.ARRAY_BUFFER, null);


				gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Grid_Normals_Reality);
				gl.bufferData(gl.ARRAY_BUFFER, grid_Nor_Reality, gl.STATIC_DRAW);
				gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_NORMAL,  4, gl.FLOAT, false, 0, 0);
				gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_NORMAL);
				gl.bindBuffer(gl.ARRAY_BUFFER, null);

				gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vbo_Grid_Index);
				gl.drawElements(gl.TRIANGLES, gGrid_NumOfElements, gl.UNSIGNED_SHORT, 0);
				gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
			}	




		gl.bindVertexArray(null);

		gl.activeTexture(gl.TEXTURE4);
		gl.bindTexture(gl.TEXTURE_2D, null);

		gl.activeTexture(gl.TEXTURE3);
		gl.bindTexture(gl.TEXTURE_2D, null);

		gl.activeTexture(gl.TEXTURE2);
		gl.bindTexture(gl.TEXTURE_2D, null);

		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, null);

		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, null);


		gl.useProgram(null);


	}
	else if(shadow == 1){

		gl.useProgram(shaderProgramObject_Grid);

		scaleBiasMatrix = mat4.fromValues(
					0.5, 0.0, 0.0, 0.0,
                                      	0.0, 0.5, 0.0, 0.0,
                                      	0.0, 0.0, 0.5, 0.0,
                                      	0.5, 0.5, 0.5, 1.0);

		

        	mat4.lookAt(lightViewMatrix, gLightPosition_AAW, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
	        mat4.perspective(lightProjectionMatrix, 45.0, parseFloat(DEPTH_TEXTURE_SIZE_AAW) / parseFloat(DEPTH_TEXTURE_SIZE_AAW), 0.1, 4000.0);

        	mat4.multiply(shadowMatrix, shadowMatrix, scaleBiasMatrix);
        	mat4.multiply(shadowMatrix, shadowMatrix, lightProjectionMatrix);
        	mat4.multiply(shadowMatrix, shadowMatrix, lightViewMatrix);

        	vec3.add(gCameraLookingAt, gCameraPosition, gCameraFront);
		mat4.lookAt(global_viewMatrix, gCameraPosition, gCameraLookingAt, gCameraUp);

		gl.uniformMatrix4fv(modelMatUniform, false, modelMatrix);
		gl.uniformMatrix4fv(viewMatUniform, false, global_viewMatrix);
		gl.uniformMatrix4fv(projMatUniform, false, gPerspectiveProjectionMatrix);
		gl.uniformMatrix4fv(shadowMatUniform, false, shadowMatrix);


		//For Reflection Reflection and Normal mode Tooggleing
		gl.uniform1i(gChoice_Reflection_Refraction_Uniform, choice);
		gl.uniform1f(gWaterLevelUniform, gWater_Level);
		gl.uniform1i(grid_choiceUniform, mode);
		gl.uniform1i(shadowChoiceUniform, Lights.SHADOW);


		//For Multitexture and HeightMap
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.uniform1i(grid_HeightMap_Sampler, 0);

		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, grid_Texture1);
		gl.uniform1i(grid_Tex1_Sampler, 1);

		gl.activeTexture(gl.TEXTURE2);
		gl.bindTexture(gl.TEXTURE_2D, grid_Texture2);
		gl.uniform1i(grid_Tex2_Sampler, 2);

		gl.activeTexture(gl.TEXTURE3);
		gl.bindTexture(gl.TEXTURE_2D, grid_Texture3);
		gl.uniform1i(grid_Tex3_Sampler, 3);

		gl.activeTexture(gl.TEXTURE4);
		gl.bindTexture(gl.TEXTURE_2D, gShadowMapTexture_AAW);
		gl.uniform1i(depthMapSamplerUniform, 4);


		//For Lights
		gl.uniform3fv(la_Uniform, lightAmbient);
		gl.uniform3fv(ld_Uniform, lightDiffuse);
		gl.uniform3fv(ls_Uniform, lightSpecular);
		gl.uniform4fv(lightPosition_Uniform, lightPosition);

		gl.uniform3fv(ka_Uniform, materialAmbient);
		gl.uniform3fv(kd_Uniform, materialDiffuse);
		gl.uniform3fv(ks_Uniform, materialSpecular);
		gl.uniform1f(shininess_Uniform, materialShininess);	



		gl.bindVertexArray(vao_Grid);


			if(mode == SceneConst.MEMORY){

				gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Grid_Pos_Memory);
				gl.bufferData(gl.ARRAY_BUFFER, grid_Pos_Memory, gl.STATIC_DRAW);
				gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,  3, gl.FLOAT, false, 0, 0);
				gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
				gl.bindBuffer(gl.ARRAY_BUFFER, null);


				gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Grid_Normals_Memory);
				gl.bufferData(gl.ARRAY_BUFFER, grid_Nor_Memory, gl.STATIC_DRAW);
				gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_NORMAL,  4, gl.FLOAT, false, 0, 0);
				gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_NORMAL);
				gl.bindBuffer(gl.ARRAY_BUFFER, null);

				gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vbo_Grid_Index);
				gl.drawElements(gl.TRIANGLES, gGrid_NumOfElements, gl.UNSIGNED_SHORT, 0);
				gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);


			}
			else{
				gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Grid_Pos_Reality);
				gl.bufferData(gl.ARRAY_BUFFER, grid_Pos_Reality, gl.STATIC_DRAW);
				gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,  3, gl.FLOAT, false, 0, 0);
				gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
				gl.bindBuffer(gl.ARRAY_BUFFER, null);


				gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Grid_Normals_Reality);
				gl.bufferData(gl.ARRAY_BUFFER, grid_Nor_Reality, gl.STATIC_DRAW);
				gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_NORMAL,  4, gl.FLOAT, false, 0, 0);
				gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_NORMAL);
				gl.bindBuffer(gl.ARRAY_BUFFER, null);

				gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vbo_Grid_Index);
				gl.drawElements(gl.TRIANGLES, gGrid_NumOfElements, gl.UNSIGNED_SHORT, 0);
				gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
			}	




		gl.bindVertexArray(null);

		gl.activeTexture(gl.TEXTURE4);
		gl.bindTexture(gl.TEXTURE_2D, null);

		gl.activeTexture(gl.TEXTURE3);
		gl.bindTexture(gl.TEXTURE_2D, null);

		gl.activeTexture(gl.TEXTURE2);
		gl.bindTexture(gl.TEXTURE_2D, null);

		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, null);

		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, null);


		gl.useProgram(null);


	}


}



function drawModelWithShadow(model, imgTexture, choice, shadow){

	var translateMatrix = mat4.create();
	var modelMatrix = mat4.create();

	var sceneModelMatrix = mat4.create();
	var sceneViewMatrix = mat4.create();
	var sceneProjectionMatrix = mat4.create();
	var shadowMatrix = mat4.create();

	var scaleBiasMatrix;

	var lightViewMatrix = mat4.create();
	var lightProjectionMatrix = mat4.create();
	var lightMVPMatrix = mat4.create();

	mat4.identity(modelMatrix);
    	mat4.identity(translateMatrix);

	if(shadow == 0){

		

    		mat4.lookAt(lightViewMatrix, gLightPosition_AAW, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        	mat4.perspective(lightProjectionMatrix, 45.0, parseFloat(DEPTH_TEXTURE_SIZE_AAW) / parseFloat(DEPTH_TEXTURE_SIZE_AAW), 0.1, 4000.0);


        	mat4.multiply(lightMVPMatrix, lightMVPMatrix, lightProjectionMatrix);
        	mat4.multiply(lightMVPMatrix, lightMVPMatrix, lightViewMatrix);


        	gl.useProgram(gShaderProgramObject_Model_AAW);
            
		gl.uniformMatrix4fv(gModelMatrixUniform_AAW, false, modelMatrix);
		gl.uniformMatrix4fv(gViewMatrixUniform_AAW, false, lightViewMatrix);
		gl.uniformMatrix4fv(gProjectionMatrixUniform_AAW, false, lightProjectionMatrix);
		gl.uniform1i(gChoiceUniform_AAW, choice);
		gl.uniform1i(gShadowChoiceUniform_AAW, shadow);

		model.drawModel();

        	gl.useProgram(null);


	}
	else if(shadow == 1){

		scaleBiasMatrix = mat4.fromValues(
					0.5, 0.0, 0.0, 0.0,
                                      	0.0, 0.5, 0.0, 0.0,
                                      	0.0, 0.0, 0.5, 0.0,
                                      	0.5, 0.5, 0.5, 1.0);

		mat4.lookAt(lightViewMatrix, gLightPosition_AAW, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        	mat4.perspective(lightProjectionMatrix, 45.0, parseFloat(DEPTH_TEXTURE_SIZE_AAW) / parseFloat(DEPTH_TEXTURE_SIZE_AAW), 0.1, 4000.0);

        	mat4.multiply(shadowMatrix, shadowMatrix, scaleBiasMatrix);
        	mat4.multiply(shadowMatrix, shadowMatrix, lightProjectionMatrix);
        	mat4.multiply(shadowMatrix, shadowMatrix, lightViewMatrix);

        	vec3.add(gCameraLookingAt, gCameraPosition, gCameraFront);
		mat4.lookAt(global_viewMatrix, gCameraPosition, gCameraLookingAt, gCameraUp);

        	//mat4.perspective(lightProjectionMatrix, 45.0, parseFloat(canvas.width) / parseFloat(canvas.height), 0.1, 4000.0);

        	gl.useProgram(gShaderProgramObject_Model_AAW);
            
		gl.uniformMatrix4fv(gModelMatrixUniform_AAW, false, modelMatrix);
		gl.uniformMatrix4fv(gViewMatrixUniform_AAW, false, global_viewMatrix);
		gl.uniformMatrix4fv(gProjectionMatrixUniform_AAW, false, gPerspectiveProjectionMatrix);
		gl.uniformMatrix4fv(gShadowMatrixUnifom_AAW, false, shadowMatrix);

		gl.uniform1i(gChoiceUniform_AAW, choice);
		gl.uniform1i(gShadowChoiceUniform_AAW, shadow);

		gl.uniform3fv(gLaUniform_AAW, gLightAmbient_AAW);
	        gl.uniform3fv(gLdUniform_AAW, gLightDiffuse_AAW);
	        gl.uniform3fv(gLsUniform_AAW, gLightSpecular_AAW);
	        gl.uniform4fv(gLightPositionUniform_AAW, lightPosition);

	        gl.uniform3fv(gKaUniform_AAW, gMaterialAmbient_AAW);
	        gl.uniform3fv(gKdUniform_AAW, gMaterialDiffuse_AAW);
	        gl.uniform3fv(gKsUniform_AAW, gMaterialSpecular_AAW);
	        gl.uniform1f(gMaterialShininessUniform_AAW, gMaterialShininess_AAW);

	        gl.activeTexture(gl.TEXTURE0);
	        gl.bindTexture(gl.TEXTURE_2D, gShadowMapTexture_AAW);
	        gl.uniform1i(gTextureSamplerUniform_AAW, 0);

		gl.activeTexture(gl.TEXTURE1);
        	gl.bindTexture(gl.TEXTURE_2D, imgTexture);
        	gl.uniform1i(gTextureSampler_AAW, 1);

		model.drawModel();

		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, null);

		gl.activeTexture(gl.TEXTURE0);
	        gl.bindTexture(gl.TEXTURE_2D, null);


        	gl.useProgram(null);


	}
    

}

function draw_terrain_tf(mode, choice, texture){

	var translateMatrix = mat4.create();
	var modelMatrix = mat4.create();


	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

	gl.useProgram(shaderProgramObject_Grid);


	mat4.identity(modelMatrix);
	mat4.identity(translateMatrix);
	mat4.identity(global_viewMatrix);

	
	switch(choice){

		//Reflection	
		case SceneConst.REFLECT_TEX:
			
			gCameraPosition[1] -= (2.0 * Math.abs(gCameraPosition[1] - gWater_Level));
			invertPitchAngle(-gPitchAngle);

			vec3.add(gCameraLookingAt, gCameraPosition, gCameraFront);
			mat4.lookAt(global_viewMatrix, gCameraPosition, gCameraLookingAt, gCameraUp);
			
			gCameraPosition[1] += (2.0 * Math.abs(gCameraPosition[1] - gWater_Level));
			invertPitchAngle(gPitchAngle);

			break;


		default:
			vec3.add(gCameraLookingAt, gCameraPosition, gCameraFront);
			mat4.lookAt(global_viewMatrix, gCameraPosition, gCameraLookingAt, gCameraUp);
			break;


	}
	

	//console.log(global_viewMatrix)
	gl.uniformMatrix4fv(modelMatUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatUniform, false, global_viewMatrix);
	gl.uniformMatrix4fv(projMatUniform, false, gPerspectiveProjectionMatrix);


	//For Reflection Reflection and Normal mode Tooggleing
	gl.uniform1i(gChoice_Reflection_Refraction_Uniform, choice);
	gl.uniform1f(gWaterLevelUniform, gWater_Level);
	gl.uniform1i(grid_choiceUniform, mode);
	gl.uniform1i(shadowChoiceUniform, 0);


	gl.activeTexture(gl.TEXTURE0);
	gl.bindTexture(gl.TEXTURE_2D, texture);
	gl.uniform1i(grid_HeightMap_Sampler, 0);

	gl.activeTexture(gl.TEXTURE1);
	gl.bindTexture(gl.TEXTURE_2D, grid_Texture1);
	gl.uniform1i(grid_Tex1_Sampler, 1);

	gl.activeTexture(gl.TEXTURE2);
	gl.bindTexture(gl.TEXTURE_2D, grid_Texture2);
	gl.uniform1i(grid_Tex2_Sampler, 2);

	gl.activeTexture(gl.TEXTURE3);
	gl.bindTexture(gl.TEXTURE_2D, grid_Texture3);
	gl.uniform1i(grid_Tex3_Sampler, 3);

	gl.activeTexture(gl.TEXTURE4);
	gl.bindTexture(gl.TEXTURE_2D, null);
	gl.uniform1i(depthMapSamplerUniform, 4);




	gl.enable(gl.RASTERIZER_DISCARD);

	gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, vbo_Grid_Normals_Memory);
	
	gl.beginTransformFeedback(gl.POINTS);


	gl.bindVertexArray(vao_Grid);


		if(mode == SceneConst.MEMORY){

			gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Grid_Pos_Memory);
			gl.bufferData(gl.ARRAY_BUFFER, grid_Pos_Memory, gl.STATIC_DRAW);
			gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,  3, gl.FLOAT, false, 0, 0);
			gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
			gl.bindBuffer(gl.ARRAY_BUFFER, null);


			gl.drawArrays(gl.POINTS, 0, GRID_WIDTH * GRID_HEIGHT);

		}
		else{
			gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Grid_Pos_Reality);
			gl.bufferData(gl.ARRAY_BUFFER, grid_Pos_Reality, gl.STATIC_DRAW);
			gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,  3, gl.FLOAT, false, 0, 0);
			gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
			gl.bindBuffer(gl.ARRAY_BUFFER, null);

			gl.drawArrays(gl.POINTS, 0, GRID_WIDTH * GRID_HEIGHT);
		}	




	gl.bindVertexArray(null);


	gl.activeTexture(gl.TEXTURE4);
	gl.bindTexture(gl.TEXTURE_2D, null);

	gl.activeTexture(gl.TEXTURE3);
	gl.bindTexture(gl.TEXTURE_2D, null);

	gl.activeTexture(gl.TEXTURE2);
	gl.bindTexture(gl.TEXTURE_2D, null);

	gl.activeTexture(gl.TEXTURE1);
	gl.bindTexture(gl.TEXTURE_2D, null);

	gl.activeTexture(gl.TEXTURE0);
	gl.bindTexture(gl.TEXTURE_2D, null);


	gl.endTransformFeedback();
	gl.disable(gl.RASTERIZER_DISCARD);


	gl.useProgram(null);



	CalculateGridNormals(mode);

}


function CalculateGridNormals(mode){


	var tempArr = new Float32Array(GRID_HEIGHT * GRID_WIDTH * 4);

	gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER, 0, tempArr);

	//console.log(tempArr);


	for(var i = 0; i < gGrid_NumOfElements; i = i + 3){

		var x, y, z;

		x = grid_Index[i + 0];
		y = grid_Index[i + 1];
		z = grid_Index[i + 2];

		var xi, xj;
		var yi, yj;
		var zi, zj;


		xi = Math.floor(x / GRID_WIDTH);
		yi = Math.floor(y / GRID_WIDTH);
		zi = Math.floor(z / GRID_WIDTH);

		xj = (x % GRID_WIDTH);
		yj = (y % GRID_WIDTH);
		zj = (z % GRID_WIDTH);


		//console.log(i, " -> ", xi, "/" , xj, " ", yi, "/", yj, " ", zi, "/", zj);
		// console.log(xj, yj, zj);

		var index0, index1, index2;

		index0 = (GRID_WIDTH * 4 * xi) + (xj * 4)
		index1 = (GRID_WIDTH * 4 * yi) + (yj * 4)
		index2 = (GRID_WIDTH * 4 * zi) + (zj * 4)


		var v0 = vec3.create();
		var v1 = vec3.create();
		var v2 = vec3.create();
		var normal = vec3.create();
		var tempNormal = vec3.create();

		var v1_v0 = vec3.create();
		var v2_v0 = vec3.create();

		v0 = [tempArr[index0 + 0], tempArr[index0 + 1], tempArr[index0 + 2]];
		v1 = [tempArr[index1 + 0], tempArr[index1 + 1], tempArr[index1 + 2]];
		v2 = [tempArr[index2 + 0], tempArr[index2 + 1], tempArr[index2 + 2]];

		// console.log(v0);

		vec3.subtract(v1_v0, v1, v0);
		vec3.subtract(v2_v0, v2, v0);

		vec3.cross(normal, v1_v0, v2_v0);

		if(mode == SceneConst.MEMORY){


			//Vertex 1
			tempNormal[0] = normal[0] + grid_Nor_Memory[index0 + 0];
			tempNormal[1] = normal[1] + grid_Nor_Memory[index0 + 1];
			tempNormal[2] = normal[2] + grid_Nor_Memory[index0 + 2];

			vec3.normalize(tempNormal, tempNormal);

			grid_Nor_Memory[index0 + 0] = tempNormal[0];
			grid_Nor_Memory[index0 + 1] = tempNormal[1];
			grid_Nor_Memory[index0 + 2] = tempNormal[2];
			grid_Nor_Memory[index0 + 3] = 1.0;



			//Vertex 2
			tempNormal[0] = normal[0] + grid_Nor_Memory[index1 + 0];
			tempNormal[1] = normal[1] + grid_Nor_Memory[index1 + 1];
			tempNormal[2] = normal[2] + grid_Nor_Memory[index1 + 2];

			vec3.normalize(tempNormal, tempNormal);


			grid_Nor_Memory[index1 + 0] = tempNormal[0];
			grid_Nor_Memory[index1 + 1] = tempNormal[1];
			grid_Nor_Memory[index1 + 2] = tempNormal[2];
			grid_Nor_Memory[index1 + 3] = 1.0;


			//Vertex 3
			tempNormal[0] = normal[0] + grid_Nor_Memory[index2 + 0];
			tempNormal[1] = normal[1] + grid_Nor_Memory[index2 + 1];
			tempNormal[2] = normal[2] + grid_Nor_Memory[index2 + 2];

			vec3.normalize(tempNormal, tempNormal);


			grid_Nor_Memory[index2 + 0] = tempNormal[0];
			grid_Nor_Memory[index2 + 1] = tempNormal[1];
			grid_Nor_Memory[index2 + 2] = tempNormal[2];
			grid_Nor_Memory[index2 + 3] = 1.0;



			// grid_Nor_Memory[index0 + 0] = 0.0;
			// grid_Nor_Memory[index0 + 1] = 1.0;
			// grid_Nor_Memory[index0 + 2] = 0.0;
			// grid_Nor_Memory[index0 + 3] = 1.0;

			// grid_Nor_Memory[index1 + 0] = 0.0;
			// grid_Nor_Memory[index1 + 1] = 1.0;
			// grid_Nor_Memory[index1 + 2] = 0.0;
			// grid_Nor_Memory[index1 + 3] = 1.0;


			// grid_Nor_Memory[index2 + 0] = 0.0;
			// grid_Nor_Memory[index2 + 1] = 1.0;
			// grid_Nor_Memory[index2 + 2] = 0.0;
			// grid_Nor_Memory[index2 + 3] = 1.0;

		}
		else{	

			//Vertex 1 
			tempNormal[0] = normal[0] + grid_Nor_Reality[index0 + 0];
			tempNormal[1] = normal[1] + grid_Nor_Reality[index0 + 1];
			tempNormal[2] = normal[2] + grid_Nor_Reality[index0 + 2];

			vec3.normalize(tempNormal, tempNormal);

			grid_Nor_Reality[index0 + 0] = tempNormal[0];
			grid_Nor_Reality[index0 + 1] = tempNormal[1];
			grid_Nor_Reality[index0 + 2] = tempNormal[2];
			grid_Nor_Reality[index0 + 3] = 1.0;


			//Vertex 2
			tempNormal[0] = normal[0] + grid_Nor_Reality[index1 + 0];
			tempNormal[1] = normal[1] + grid_Nor_Reality[index1 + 1];
			tempNormal[2] = normal[2] + grid_Nor_Reality[index1 + 2];

			vec3.normalize(tempNormal, tempNormal);


			grid_Nor_Reality[index1 + 0] = tempNormal[0];
			grid_Nor_Reality[index1 + 1] = tempNormal[1];
			grid_Nor_Reality[index1 + 2] = tempNormal[2];
			grid_Nor_Reality[index1 + 3] = 1.0;


			//Vertex 3
			tempNormal[0] = normal[0] + grid_Nor_Reality[index2 + 0];
			tempNormal[1] = normal[1] + grid_Nor_Reality[index2 + 1];
			tempNormal[2] = normal[2] + grid_Nor_Reality[index2 + 2];

			vec3.normalize(tempNormal, tempNormal);


			grid_Nor_Reality[index2 + 0] = tempNormal[0];
			grid_Nor_Reality[index2 + 1] = tempNormal[1];
			grid_Nor_Reality[index2 + 2] = tempNormal[2];
			grid_Nor_Reality[index2 + 3] = 1.0;



		}



		// console.log(v0);
		// console.log(v1);
		// console.log(v2);


		// console.log(tempArr[index0 + 0], tempArr[index0 + 1], tempArr[index0 + 2], tempArr[index0 + 3]);
		// console.log(tempArr[index1 + 0], tempArr[index1 + 1], tempArr[index1 + 2], tempArr[index1 + 3]);
		// console.log(tempArr[index2 + 0], tempArr[index2 + 1], tempArr[index2 + 2], tempArr[index2 + 3]);

		// console.log("Cross");
		// console.log(v1_v0);
		// console.log(v2_v0);



		// console.log("Nor");
		// console.log(normal[0], normal[1], normal[2]);
		//console.log(tempNormal);
		// console.log(grid_Nor_Memory[index0 + 0], grid_Nor_Memory[index0 + 1], grid_Nor_Memory[index0 + 2], grid_Nor_Memory[index0 + 3]);
		
		


	}

	var i = 0;

	while(i < ASM_INSTANCED_COUNT){

			var x, y;

			x = Math.random() * GRID_WIDTH;	
			x = Math.floor(x);

			y = Math.random() * GRID_WIDTH;
			y = Math.floor(y);

			var index0;

			index0 = (GRID_WIDTH * 4 * x) + (y * 4);

			var h = tempArr[index0 + 1];
			// console.log(h);

			if(mode == SceneConst.MEMORY && h < gWater_Level)
				continue;

			
	 		
	 		if(mode == SceneConst.MEMORY){
	 			gMemoryModelPos[i * 3 + 0] = tempArr[index0 + 0];
	 			gMemoryModelPos[i * 3 + 1] = tempArr[index0 + 1];
	 			gMemoryModelPos[i * 3 + 2] = tempArr[index0 + 2];
	 		}
	 		else{
	 			gRealityModelPos[i * 3 + 0] = tempArr[index0 + 0];
	 			gRealityModelPos[i * 3 + 1] = tempArr[index0 + 1];
	 			gRealityModelPos[i * 3 + 2] = tempArr[index0 + 2];

	 			//console.log(tempArr[index0 + 0] + " , " + tempArr[index0 + 1] + " , " + tempArr[index0 + 2]);
	 		}

	 		i++;
		}
		




	if(mode == SceneConst.MEMORY){


		
		
		
		i = 0;

		while(i < TOTAL_POINT_LIGHT){

			// var x, y;
			// var res = GRID_WIDTH;

			// x = 50 + Math.random() * 150;	
			// x = Math.floor(x);

			// y = 50 + Math.random() * 150;
			// y = Math.floor(y);

			// var index0;

			// index0 = (GRID_WIDTH * 4 * x) + (y * 4);

			// var x = tempArr[index0 + 0];
			// var h = tempArr[index0 + 1];
			// // console.log(h);


			// // if(i > 0 && Math.abs(x - pointLight_Position[(i - 1) * 4 + 0]) < 40.0)
			// // 	continue;

			// if(h < (gWater_Level + 5.0))
			// 	continue;


 		// 	pointLight_Position[i * 4 + 0] = tempArr[index0 + 0];
 		// 	pointLight_Position[i * 4 + 1] = tempArr[index0 + 1] + 20.0;
 		// 	pointLight_Position[i * 4 + 2] = tempArr[index0 + 2];
 		// 	pointLight_Position[i * 4 + 3] = 1.0;

 			// pointLight_Ld[i * 3 + 0] = Math.random();
 			// pointLight_Ld[i * 3 + 1] = Math.random();
 			// pointLight_Ld[i * 3 + 2] = Math.random();

 			// pointLight_Ls[i * 3 + 0] = Math.random();
 			// pointLight_Ls[i * 3 + 1] = Math.random();
 			// pointLight_Ls[i * 3 + 2] = Math.random();

			pointLight_Ld[i * 3 + 0] = 1.0;
 			pointLight_Ld[i * 3 + 1] = 0.50;
 			pointLight_Ld[i * 3 + 2] = 0.0;

 			pointLight_Ls[i * 3 + 0] = 1.0;
 			pointLight_Ls[i * 3 + 1] = 0.50;
 			pointLight_Ls[i * 3 + 2] = 0.0;

 		
 			// console.log(pointLight_Position);

			i++;
		}

		console.log(pointLight_Position);
	}	


	gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);




	console.log("End");

}	