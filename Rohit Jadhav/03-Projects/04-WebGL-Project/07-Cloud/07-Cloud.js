var canvas;
var gl;


//For Shader
var vertexShaderObject_Cloud;
var fragmentShaderObject_Cloud;
var shaderProgramObject_Cloud;


//For Projection
var orthoProjectionMatrix_Cloud;


//For Rect
var vao_Rect_Cloud;
var vbo_rect_Position_Cloud_Cloud;
var vbo_Rect_Texcoord_Cloud;
var rect_Position_Cloud = new Float32Array(4 * 3);


//For Uniform
var mvpUniform_Cloud;
var noiseFactorUniform_Cloud;
var octaveUniform_Cloud;
var lacunarityUniform_Cloud;
var gainUniform_Cloud;
var noiseAnimationUniform_Cloud;
var samplerUniform_Cloud;


var gfNoiseFactor_Cloud = 8.0;
var gfLacunarity_Cloud = 2.0;
var gfGain_Cloud = 0.60;
var giOctaves_Cloud = 5;
var gfvNoiseAnimation_Cloud = [0.0, 0.0];



function initialize_Cloud(){

	vertexShaderObject_Cloud = gl.createShader(gl.VERTEX_SHADER);
	var szVertexShaderSourceCode = 
		"#version 300 es" +
		"\n" +

		"precision mediump float;" +

		"in vec4 vPosition;" +
		"in vec2 vTex;" + 
		
		"out vec2 outTexCoord;" +

		"uniform mat4 u_mvp_matrix;" +
		
		"void main() {" +
			"outTexCoord = vTex;" +
			"gl_Position = u_mvp_matrix * vPosition;" +
		"}";

	gl.shaderSource(vertexShaderObject_Cloud, szVertexShaderSourceCode);

	gl.compileShader(vertexShaderObject_Cloud);

	var  shaderCompileStatus = gl.getShaderParameter(vertexShaderObject_Cloud, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(vertexShaderObject_Cloud);
		if(error.length > 0){
			alert("Vertex Shader Compilation Error: " + error);
			uninitialize();
			window.close();
		}
	}


	fragmentShaderObject_Cloud = gl.createShader(gl.FRAGMENT_SHADER);
	var szFragmentShaderSourceCode = 
		"#version 300 es" +
		"\n" +
		"precision mediump float;" +
		"precision highp sampler2D;" +
		
		"in vec2 outTexCoord;" +
		"uniform float u_fNoiseFactor;" +
		"uniform vec2 u_v2NoiseAnimation;" +

		"uniform int u_iOctaves;" +
		"uniform float u_fLacunarity;" +
		"uniform float u_fGain;" +

		"uniform float u_alpha;" +

		"out vec4 v4FragColor;" +
		

		"float value_random(vec2 st) {" +

			"float fDotProduct = dot(st.xy, vec2(12.9898, 78.233));" +
		
			"float fSin = sin(fDotProduct);" +
			
			"float fRet = fract(fSin * 43758.5453123);" +

			"return(fRet);" +

		"}" +


		"float value_noise(vec2 st){" +

			"vec2 v2IntPart = floor(st);" +
			"vec2 v2FractPart = fract(st);" +

			"float a = value_random(v2IntPart);" +
			"float b = value_random(v2IntPart + vec2(1.0f, 0.0f));" +
			"float c = value_random(v2IntPart + vec2(0.0f, 1.0f));" +
			"float d = value_random(v2IntPart + vec2(1.0f, 1.0f));" +

			"vec2 v2MixIntensity = v2FractPart * v2FractPart * (3.0f - 2.0f * v2FractPart);" +

			"float ab = mix(a, b, v2MixIntensity.x);" +
			"float cd = mix(c, d, v2MixIntensity.x);" +
			"float all = mix(ab, cd, v2MixIntensity.y);" +

			"return(all);" +

		"}" +


		"float fbm_noise(vec2 st){" +

			"float value = 0.0f;" +
			"float amplitude = 0.35f;" +
			"float frequency = 0.80f;" +

			"for(int i = 0; i < u_iOctaves; i++){" +

				"value = value + (amplitude * value_noise(frequency * st));" +

				"frequency = frequency * u_fLacunarity;" +

				"amplitude = amplitude * u_fGain;" +

			"}" +

			"return(value);" +

		"}" +





		"void main(void)" +
		"{" +
			
			
			"vec2 v2st = outTexCoord.xy;" +

			"v2st *= u_fNoiseFactor;" +

			// "vec3 v3Color = vec3(0.1230f, 0.1230f, 1.0f);" +

			"vec3 v3Color = vec3(0.0f);" +

			"v3Color += fbm_noise(v2st + u_v2NoiseAnimation);" +
			

			"v4FragColor = vec4(v3Color,  u_alpha);" +


		"}";

	gl.shaderSource(fragmentShaderObject_Cloud, szFragmentShaderSourceCode);
	gl.compileShader(fragmentShaderObject_Cloud);

	shaderCompileStatus = gl.getShaderParameter(fragmentShaderObject_Cloud, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(fragmentShaderObject_Cloud);
		if(error.length > 0){
			alert("Fragment Shader Compilation Error: "+ error);
			uninitialize();
			window.close();
		}
	}


	shaderProgramObject_Cloud = gl.createProgram();

	gl.attachShader(shaderProgramObject_Cloud, vertexShaderObject_Cloud);
	gl.attachShader(shaderProgramObject_Cloud, fragmentShaderObject_Cloud);

	gl.bindAttribLocation(shaderProgramObject_Cloud, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
	gl.bindAttribLocation(shaderProgramObject_Cloud, WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0, "vTex");

	gl.linkProgram(shaderProgramObject_Cloud);

	var programLinkStatus = gl.getProgramParameter(shaderProgramObject_Cloud, gl.LINK_STATUS);

	if(programLinkStatus == false){
		var error = gl.getProgramInfoLog(shaderProgramObject_Cloud);
		if(error.length > 0){
			alert("Program Linkink Error: " + error);
			uninitialize();
			window.close();
		}
	}



	mvpUniform_Cloud = gl.getUniformLocation(shaderProgramObject_Cloud, "u_mvp_matrix");
	noiseFactorUniform_Cloud = gl.getUniformLocation(shaderProgramObject_Cloud, "u_fNoiseFactor");
	octaveUniform_Cloud = gl.getUniformLocation(shaderProgramObject_Cloud, "u_iOctaves");
	lacunarityUniform_Cloud = gl.getUniformLocation(shaderProgramObject_Cloud, "u_fLacunarity");
	gainUniform_Cloud = gl.getUniformLocation(shaderProgramObject_Cloud, "u_fGain");
	noiseAnimationUniform_Cloud = gl.getUniformLocation(shaderProgramObject_Cloud, "u_v2NoiseAnimation");

	alphaUniform = gl.getUniformLocation(shaderProgramObject_Cloud, "u_alpha");


	
	

	var rect_Texcoord = new Float32Array([
			1.0, 1.0,
			0.0, 1.0,
			0.0, 0.0,
			1.0, 0.0,
		]);



	/********* Rectangle *********/
	vao_Rect_Cloud = gl.createVertexArray();
	gl.bindVertexArray(vao_Rect_Cloud);

		/********* Position **********/
		vbo_rect_Position_Cloud_Cloud = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_rect_Position_Cloud_Cloud);
		gl.bufferData(gl.ARRAY_BUFFER,  rect_Position_Cloud, gl.DYNAMIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION, 3, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Texture **********/
		vbo_Rect_Texcoord_Cloud = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Rect_Texcoord_Cloud);
		gl.bufferData(gl.ARRAY_BUFFER, rect_Texcoord, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0, 2, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);


	orthoProjectionMatrix_Cloud = mat4.create();

}



function uninitialize_Cloud(){


	if(vbo_Rect_Texcoord_Cloud){
		gl.deleteBuffer(vbo_Rect_Texcoord_Cloud);
		vbo_Rect_Texcoord_Cloud = 0;
	}


	if(vbo_rect_Position_Cloud_Cloud){
		gl.deleteBuffer(vbo_rect_Position_Cloud_Cloud);
		vbo_rect_Position_Cloud_Cloud = 0;
	}

	if(vao_Rect_Cloud){
		gl.deleteVertexArray(vao_Rect_Cloud);
		vao_Rect_Cloud = 0;
	}


	if(shaderProgramObject_Cloud){

		gl.useProgram(shaderProgramObject_Cloud);

			if(fragmentShaderObject_Cloud){
				gl.detachShader(shaderProgramObject_Cloud, fragmentShaderObject_Cloud);
				gl.deleteShader(fragmentShaderObject_Cloud);
				fragmentShaderObject_Cloud = 0;
			}

			if(vertexShaderObject_Cloud){
				gl.detachShader(shaderProgramObject_Cloud, vertexShaderObject_Cloud);
				gl.deleteShader(vertexShaderObject_Cloud);
				vertexShaderObject_Cloud = 0;
			}

		gl.useProgram(null);
		gl.deleteProgram(shaderProgramObject_Cloud);
		shaderProgramObject_Cloud = 0;
	}

}




function draw_Cloud(alpha){

	var modelViewMatrix = mat4.create();
	var modelViewProjectionMatrix  = mat4.create();


	mat4.ortho(orthoProjectionMatrix_Cloud,
		-canvas.width / 2.0, canvas.width / 2.0,
		-canvas.width / 2.0, canvas.width / 2.0,
		-1.0, 1.0);


	rect_Position_Cloud[0] = canvas.width / 2.0;
	rect_Position_Cloud[1] = canvas.width / 2.0;
	rect_Position_Cloud[2] = 0.0;

	rect_Position_Cloud[3] = -canvas.width / 2.0;
	rect_Position_Cloud[4] = canvas.width / 2.0;
	rect_Position_Cloud[5] = 0.0;

	rect_Position_Cloud[6] = -canvas.width / 2.0;
	rect_Position_Cloud[7] = -canvas.width / 2.0;
	rect_Position_Cloud[8] = 0.0;

	rect_Position_Cloud[9] = canvas.width / 2.0;
	rect_Position_Cloud[10] = -canvas.width / 2.0;
	rect_Position_Cloud[12] = 0.0;

	gl.enable(gl.BLEND);
	gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

	gl.useProgram(shaderProgramObject_Cloud);


		/********** Rectangle **********/
		mat4.identity(modelViewMatrix);
		mat4.identity(modelViewProjectionMatrix);
		mat4.multiply(modelViewProjectionMatrix, orthoProjectionMatrix_Cloud, modelViewMatrix);

		gl.uniformMatrix4fv(mvpUniform_Cloud, false, modelViewProjectionMatrix);

		
		gl.uniform1f(noiseFactorUniform_Cloud, gfNoiseFactor_Cloud);
		gl.uniform1f(lacunarityUniform_Cloud, gfLacunarity_Cloud);
		gl.uniform1f(gainUniform_Cloud, gfGain_Cloud);
		gl.uniform2fv(noiseAnimationUniform_Cloud, gfvNoiseAnimation_Cloud);
		gl.uniform1i(octaveUniform_Cloud, giOctaves_Cloud);

		gl.uniform1f(alphaUniform, alpha);

		gl.bindVertexArray(vao_Rect_Cloud);

			gl.bindBuffer(gl.ARRAY_BUFFER, vbo_rect_Position_Cloud_Cloud);
			gl.bufferData(gl.ARRAY_BUFFER, rect_Position_Cloud, gl.DYNAMIC_DRAW);
			gl.bindBuffer(gl.ARRAY_BUFFER, null);

			gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);

		gl.bindVertexArray(null);
		
	gl.useProgram(null);

	gl.disable(gl.BLEND);

	gfvNoiseAnimation_Cloud[0] += 0.01;
	// gfvNoiseAnimation_Cloud[1] += 0.01;

}
