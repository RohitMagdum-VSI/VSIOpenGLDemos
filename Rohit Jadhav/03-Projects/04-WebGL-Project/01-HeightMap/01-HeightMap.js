//External 
var canvas;
var gl;
var canvas_Width;
var canvas_Height;
var texture_Scene_Memory;
var texture_Scene_Reality;



//For Shader
var HeightMap_vertexShaderObject;
var HeightMap_fragmentShaderObject;
var HeightMap_shaderProgramObject;

//For Uniform
var HeightMap_mvpUniform;
var HeightMap_samplerUniform;
var HeightMap_choiceUniform;

//For Projection
var HeightMap_orthoProjectionMatrix;
var HeightMap_perspectiveProjectionMatrix;



//For Rect
var HeightMap_vao_Rect;
var HeightMap_vbo_Rect_Position;
var HeightMap_vbo_Rect_Texcoord;
var HeightMap_rect_Position = new Float32Array(4 * 3);
var HeightMap_rect_Position_normal = new Float32Array(4 * 3);


//For Uniform
var noiseFactorUniform;
var octaveUniform;
var lacunarityUniform;
var gainUniform;
var noiseAnimationUniform;


var gfNoiseFactor = 80.0;
var gfLacunarity = 1.50;
var gfGain = 1.0;
var giOctaves = 2;
var gfvNoiseAnimation = [8.0, 0.0];


//For Framebuffer
var frameBufferObject;
var renderBufferObject_Depth;	

var textureWidth = 512;
var textureHeight = 512;

var noiseAnimationX = 0.0;
var noiseAnimationY = 0.0;




function initialize_HeightMap(){


	gl.viewportWidth = canvas.width;
	gl.viewportHeight = canvas.height;



	HeightMap_vertexShaderObject = gl.createShader(gl.VERTEX_SHADER);
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

	gl.shaderSource(HeightMap_vertexShaderObject, szVertexShaderSourceCode);

	gl.compileShader(HeightMap_vertexShaderObject);

	var  shaderCompileStatus = gl.getShaderParameter(HeightMap_vertexShaderObject, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(HeightMap_vertexShaderObject);
		if(error.length > 0){
			alert("Height Map : Vertex Shader Compilation Error: " + error);
			uninitialize_HeightMap();
			window.close();
		}
	}
	else
		console.log("Height Map : Vertex Shader Compilation Done");


	HeightMap_fragmentShaderObject = gl.createShader(gl.FRAGMENT_SHADER);
	var szFragmentShaderSourceCode = 
		"#version 300 es" +
		"\n" +
		"precision mediump float;" +
		"precision highp sampler2D;" +
		
		"uniform sampler2D u_sampler;" +
		"uniform int u_choice;" +

		"in vec2 outTexCoord;" +
		"uniform float u_fNoiseFactor;" +
		"uniform vec2 u_v2NoiseAnimation;" +

		"uniform int u_iOctaves;" +
		"uniform float u_fLacunarity;" +
		"uniform float u_fGain;" +

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


		"float fbm_noise(vec2 st, float amp, float freq, float lacunarity, float gain, int octave){" +

			"float value = 0.0f;" +
			"float amplitude = amp;" +
			"float frequency = freq;" +

			"for(int i = 0; i < octave; i++){" +

				"value = value + (amplitude * value_noise(frequency * st));" +

				"frequency = frequency * lacunarity;" +

				"amplitude = amplitude * gain;" +

			"}" +

			"return(value);" +

		"}" +





		"void main(void)" +
		"{" +
				
			"vec2 v2Center = vec2(0.5);" +

			"float noiseFac = 0.0;" +
			"float amp = 0.250f;" +
			"float freq = 1.0f;" +
				
			"if(u_choice == 1){ " +	
			
				"vec2 v2st = outTexCoord.xy;" +

				"float dis = length(v2st - v2Center);" +

				"if(dis < 1.0){" +

					"noiseFac = 10.0;" +
				"}" +

				"v2st *= noiseFac;" +

				"vec3 v3Color = vec3(0.0f, 0.0f, 0.0f);" +

				"v3Color += fbm_noise(v2st + u_v2NoiseAnimation, amp, freq, u_fLacunarity, u_fGain, 2);" +
				

				"v4FragColor = vec4(v3Color, 1.0f);" +

			"}" +

			"else if(u_choice == 2){" +

				"vec2 v2st = outTexCoord.xy;" +

				"v2st *= 80.0f;" +

				"vec3 v3Color = vec3(0.0f, 0.0f, 0.0f);" +

				"v3Color += fbm_noise(v2st + u_v2NoiseAnimation, 0.5, 1.0f, 1.0f, 0.20f, 3);" +
				

				"v4FragColor = vec4(v3Color, 1.0f);" +	

			"}" +

			"else {" +

				"v4FragColor = texture(u_sampler, outTexCoord);" +
			
			"}" +

		"}";

	gl.shaderSource(HeightMap_fragmentShaderObject, szFragmentShaderSourceCode);
	gl.compileShader(HeightMap_fragmentShaderObject);

	shaderCompileStatus = gl.getShaderParameter(HeightMap_fragmentShaderObject, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(HeightMap_fragmentShaderObject);
		if(error.length > 0){
			alert("Height Map : Fragment Shader Compilation Error: "+ error);
			uninitialize_HeightMap();
			window.close();
		}
	}
	else
		console.log("Height Map : Fragment Shader Compilation Done");


	HeightMap_shaderProgramObject = gl.createProgram();

	gl.attachShader(HeightMap_shaderProgramObject, HeightMap_vertexShaderObject);
	gl.attachShader(HeightMap_shaderProgramObject, HeightMap_fragmentShaderObject);

	gl.bindAttribLocation(HeightMap_shaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
	gl.bindAttribLocation(HeightMap_shaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0, "vTex");

	gl.linkProgram(HeightMap_shaderProgramObject);

	var programLinkStatus = gl.getProgramParameter(HeightMap_shaderProgramObject, gl.LINK_STATUS);

	if(programLinkStatus == false){
		var error = gl.getProgramInfoLog(HeightMap_shaderProgramObject);
		if(error.length > 0){
			alert("Height Map : Program Linking Error: " + error);
			uninitialize_HeightMap();
			window.close();
		}
	}
	else
		console.log("Height Map : Program Linking Done");



	HeightMap_mvpUniform = gl.getUniformLocation(HeightMap_shaderProgramObject, "u_mvp_matrix");
	
	noiseFactorUniform = gl.getUniformLocation(HeightMap_shaderProgramObject, "u_fNoiseFactor");
	octaveUniform = gl.getUniformLocation(HeightMap_shaderProgramObject, "u_iOctaves");
	lacunarityUniform = gl.getUniformLocation(HeightMap_shaderProgramObject, "u_fLacunarity");
	gainUniform = gl.getUniformLocation(HeightMap_shaderProgramObject, "u_fGain");
	noiseAnimationUniform = gl.getUniformLocation(HeightMap_shaderProgramObject, "u_v2NoiseAnimation");

	HeightMap_choiceUniform = gl.getUniformLocation(HeightMap_shaderProgramObject, "u_choice");
	HeightMap_samplerUniform = gl.getUniformLocation(HeightMap_shaderProgramObject, "u_sampler");


	
	HeightMap_rect_Position_normal[0] = 1.0;
	HeightMap_rect_Position_normal[1] = 1.0;
	HeightMap_rect_Position_normal[2] = 0.0;

	HeightMap_rect_Position_normal[3] = -1.0;
	HeightMap_rect_Position_normal[4] = 1.0;
	HeightMap_rect_Position_normal[5] = 0.0;


	HeightMap_rect_Position_normal[6] = -1.0;
	HeightMap_rect_Position_normal[7] = -1.0;
	HeightMap_rect_Position_normal[8] = 0.0;

	HeightMap_rect_Position_normal[9] = 1.0;
	HeightMap_rect_Position_normal[10] = -1.0;
	HeightMap_rect_Position_normal[11] = 0.0;
	

	var rect_Texcoord = new Float32Array([
			1.0, 1.0,
			0.0, 1.0,
			0.0, 0.0,
			1.0, 0.0,
		]);



	/********* Rectangle *********/
	HeightMap_vao_Rect = gl.createVertexArray();
	gl.bindVertexArray(HeightMap_vao_Rect);

		/********* Position **********/
		HeightMap_vbo_Rect_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, HeightMap_vbo_Rect_Position);
		gl.bufferData(gl.ARRAY_BUFFER,  HeightMap_rect_Position, gl.DYNAMIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION, 3, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Texture **********/
		HeightMap_vbo_Rect_Texcoord = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, HeightMap_vbo_Rect_Texcoord);
		gl.bufferData(gl.ARRAY_BUFFER, rect_Texcoord, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0, 2, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);



	/********** FRAMEBUFFER **********/
	frameBufferObject = gl.createFramebuffer();
	gl.bindFramebuffer(gl.FRAMEBUFFER, frameBufferObject);


		/********** Texture Memory **********/
		texture_Scene_Memory = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, texture_Scene_Memory);

		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
		
		gl.texImage2D(gl.TEXTURE_2D, 0, 
					gl.RGBA, 
					textureWidth, textureHeight, 0,
					gl.RGBA, 
					gl.UNSIGNED_BYTE, null);

		gl.bindTexture(gl.TEXTURE_2D, null);


		/********** Texture Reality **********/
		texture_Scene_Reality = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, texture_Scene_Reality);

		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
		
		gl.texImage2D(gl.TEXTURE_2D, 0, 
					gl.RGBA, 
					textureWidth, textureHeight, 0,
					gl.RGBA, 
					gl.UNSIGNED_BYTE, null);

		gl.bindTexture(gl.TEXTURE_2D, null);

		
		gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture_Scene_Memory, 0);

		


		/********** For Depth **********/
		renderBufferObject_Depth = gl.createRenderbuffer();
		gl.bindRenderbuffer(gl.RENDERBUFFER, renderBufferObject_Depth);
		gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, textureWidth, textureHeight);
		gl.bindRenderbuffer(gl.RENDERBUFFER, null);

		gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, renderBufferObject_Depth);


		/********** Checking *********/
		if(gl.checkFramebufferStatus(gl.FRAMEBUFFER) != gl.FRAMEBUFFER_COMPLETE){
			alert("checkFramebufferStatus: Failed");
			uninitialize_HeightMap();
			window.close();
		}

	gl.bindFramebuffer(gl.FRAMEBUFFER, null);


	HeightMap_orthoProjectionMatrix = mat4.create();
	HeightMap_perspectiveProjectionMatrix = mat4.create();


	console.log("Height Map : Initialize Complete");

}



function uninitialize_HeightMap(){

	if(texture_Scene_Memory){
		gl.deleteTexture(texture_Scene_Memory);
		texture_Scene_Memory = 0;
	}


	if(frameBufferObject){
		gl.deleteFramebuffer(frameBufferObject);
		frameBufferObject = 0;
	}


	if(HeightMap_vbo_Rect_Texcoord){
		gl.deleteBuffer(HeightMap_vbo_Rect_Texcoord);
		HeightMap_vbo_Rect_Texcoord = 0;
	}


	if(HeightMap_vbo_Rect_Position){
		gl.deleteBuffer(HeightMap_vbo_Rect_Position);
		HeightMap_vbo_Rect_Position = 0;
	}

	if(HeightMap_vao_Rect){
		gl.deleteVertexArray(HeightMap_vao_Rect);
		HeightMap_vao_Rect = 0;
	}


	if(HeightMap_shaderProgramObject){

		gl.useProgram(HeightMap_shaderProgramObject);

			if(HeightMap_fragmentShaderObject){
				gl.detachShader(HeightMap_shaderProgramObject, HeightMap_fragmentShaderObject);
				gl.deleteShader(HeightMap_fragmentShaderObject);
				HeightMap_fragmentShaderObject = 0;
			}

			if(HeightMap_vertexShaderObject){
				gl.detachShader(HeightMap_shaderProgramObject, HeightMap_vertexShaderObject);
				gl.deleteShader(HeightMap_vertexShaderObject);
				HeightMap_vertexShaderObject = 0;
			}

		gl.useProgram(null);
		gl.deleteProgram(HeightMap_shaderProgramObject);
		HeightMap_shaderProgramObject = 0;
	}


	console.log("Height Map : Uninitialize Complete");

}



function draw_HeightMap(choice){

	var modelViewMatrix = mat4.create();
	var modelViewProjectionMatrix  = mat4.create();


	// Offscreen Rendering for creation of noise 512x512 texture
	gl.bindFramebuffer(gl.FRAMEBUFFER, frameBufferObject);

		if(choice == 1){
			gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture_Scene_Memory, 0);
		}
		else if(choice == 2){
			gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture_Scene_Reality, 0);
		}


		gl.clearColor(0.0, 0.0, 0.0, 1.0);
		gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

		gl.useProgram(HeightMap_shaderProgramObject);

			gl.viewport(0, 0, textureWidth, textureHeight);

			mat4.identity(HeightMap_orthoProjectionMatrix);
			mat4.ortho(HeightMap_orthoProjectionMatrix,
				-textureWidth / 2.0, textureWidth / 2.0,
				-textureHeight / 2.0, textureHeight / 2.0,
				-1.0, 1.0);


			HeightMap_rect_Position[0] = textureWidth / 2.0;
			HeightMap_rect_Position[1] = textureHeight / 2.0;
			HeightMap_rect_Position[2] = 0.0;

			HeightMap_rect_Position[3] = -textureWidth / 2.0;
			HeightMap_rect_Position[4] = textureHeight / 2.0;
			HeightMap_rect_Position[5] = 0.0;

			HeightMap_rect_Position[6] = -textureWidth / 2.0;
			HeightMap_rect_Position[7] = -textureHeight / 2.0;
			HeightMap_rect_Position[8] = 0.0;

			HeightMap_rect_Position[9] = textureWidth / 2.0;
			HeightMap_rect_Position[10] = -textureHeight / 2.0;
			HeightMap_rect_Position[12] = 0.0;


			//********* Rectangle *********
			mat4.identity(modelViewMatrix);
			mat4.identity(modelViewProjectionMatrix);
			mat4.multiply(modelViewProjectionMatrix, HeightMap_orthoProjectionMatrix, modelViewMatrix);

			gl.uniformMatrix4fv(HeightMap_mvpUniform, false, modelViewProjectionMatrix);
			gl.uniform1i(HeightMap_choiceUniform, choice);
			
			gl.uniform1f(noiseFactorUniform, gfNoiseFactor);
			gl.uniform1f(lacunarityUniform, gfLacunarity);
			gl.uniform1f(gainUniform, gfGain);
			gl.uniform2fv(noiseAnimationUniform, gfvNoiseAnimation);
			gl.uniform1i(octaveUniform, giOctaves);

			gl.bindVertexArray(HeightMap_vao_Rect);

				gl.bindBuffer(gl.ARRAY_BUFFER, HeightMap_vbo_Rect_Position);
				gl.bufferData(gl.ARRAY_BUFFER, HeightMap_rect_Position, gl.DYNAMIC_DRAW);
				gl.bindBuffer(gl.ARRAY_BUFFER, null);

				gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);

			gl.bindVertexArray(null);

		gl.useProgram(null);

	gl.bindFramebuffer(gl.FRAMEBUFFER, null);



	// // Rectangle that Render Noise 512x512 Texture
	// gl.viewport(0, 300, 300, 300);
	// HeightMap_perspectiveProjectionMatrix = mat4.perspective(perspectiveProjectionMatrix, 
	// 	45.0, parseFloat(300) / parseFloat(300), 0.1, 1000.0);	

	// gl.clearColor(0.0, 0.0, 0.0, 1.0);
	// gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

	// gl.useProgram(HeightMap_shaderProgramObject);

	// 	/********** Rectangle **********/
	// 	mat4.identity(modelViewMatrix);
	// 	mat4.identity(modelViewProjectionMatrix);

	// 	mat4.translate(modelViewMatrix, modelViewMatrix, [0.0, 0.0, -3.0]);
	// 	mat4.multiply(modelViewProjectionMatrix, HeightMap_perspectiveProjectionMatrix, modelViewMatrix);

	// 	gl.uniformMatrix4fv(HeightMap_mvpUniform, false, modelViewProjectionMatrix);

		
	// 	gl.uniform1i(HeightMap_choiceUniform, 0);

	// 	gl.activeTexture(gl.TEXTURE0);
	// 	gl.bindTexture(gl.TEXTURE_2D, texture_Scene_Memory);
	// 	gl.uniform1i(HeightMap_samplerUniform, 0);


	// 	gl.bindVertexArray(HeightMap_vao_Rect);

	// 		gl.bindBuffer(gl.ARRAY_BUFFER, HeightMap_vbo_Rect_Position);
	// 		gl.bufferData(gl.ARRAY_BUFFER, HeightMap_rect_Position_normal, gl.DYNAMIC_DRAW);
	// 		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	// 		gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);

	// 	gl.bindVertexArray(null);

	// 	gl.bindTexture(gl.TEXTURE_2D, null);

	// gl.useProgram(null);


	gl.viewport(0, 0, canvas.width, canvas.height);
	


	gfvNoiseAnimation[0] += noiseAnimationX;
	gfvNoiseAnimation[1] += noiseAnimationY;

	//console.log("display");

}

