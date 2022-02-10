//For External
var canvas;
var gl;
var global_viewMatrix;




//For Shader
var testRect_vertexShaderObject;
var testRect_fragmentShaderObject;
var testRect_shaderProgramObject;

var testRect_perspectiveProjectionMatrix;

//For Rect
var testRect_vao_Rect;
var testRect_vbo_Rect_Position;
var testRect_vbo_Rect_Texcoord;

var testRect_modelMatUniform;
var testRect_viewMatUniform;
var testRect_projMatUniform;



function initialize_testRect(){


	testRect_vertexShaderObject = gl.createShader(gl.VERTEX_SHADER);
	var szVertexShaderSourceCode = 
		"#version 300 es" +
		"\n" +

		"precision mediump float;" +

		"in vec4 vPosition;" +
		"in vec2 vTex;" + 
		
		"out vec2 outTexCoord;" +

		"uniform mat4 u_model_matrix;" +
		"uniform mat4 u_view_matrix;" +
		"uniform mat4 u_proj_matrix;" +

		
		"void main() {" +
			
			"outTexCoord = vTex;" +
			"gl_Position = u_proj_matrix * u_view_matrix * u_model_matrix * vPosition;" +
		"}";

	gl.shaderSource(testRect_vertexShaderObject, szVertexShaderSourceCode);

	gl.compileShader(testRect_vertexShaderObject);

	var  shaderCompileStatus = gl.getShaderParameter(testRect_vertexShaderObject, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(testRect_vertexShaderObject);
		if(error.length > 0){
			alert("00-testRect -> Vertex Shader Compilation Error: " + error);
			uninitialize_testRect();
			window.close();
		}
	}


	testRect_fragmentShaderObject = gl.createShader(gl.FRAGMENT_SHADER);
	var szFragmentShaderSourceCode = 
		"#version 300 es" +
		"\n" +
		"precision mediump float;" +
		"precision mediump sampler2D;" +
	
		"in vec2 outTexCoord;" +
		"uniform sampler2D u_sampler;" +
	
		"out vec4 v4FragColor;" +
		

		"void main(void)" +
		"{" +
			"float col = texture(u_sampler, outTexCoord).r;" +


			"v4FragColor = vec4(col);" +


			// "v4FragColor = texture(u_sampler, outTexCoord);" +
			// "v4FragColor = vec4(1.0);" +
		"}";

	gl.shaderSource(testRect_fragmentShaderObject, szFragmentShaderSourceCode);
	gl.compileShader(testRect_fragmentShaderObject);

	shaderCompileStatus = gl.getShaderParameter(testRect_fragmentShaderObject, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(testRect_fragmentShaderObject);
		if(error.length > 0){
			alert("02-testRect -> Fragment Shader Compilation Error: "+ error);
			uninitialize_testRect();
			window.close();
		}
	}


	testRect_shaderProgramObject = gl.createProgram();

	gl.attachShader(testRect_shaderProgramObject, testRect_vertexShaderObject);
	gl.attachShader(testRect_shaderProgramObject, testRect_fragmentShaderObject);

	gl.bindAttribLocation(testRect_shaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
	gl.bindAttribLocation(testRect_shaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0, "vTex");

	gl.linkProgram(testRect_shaderProgramObject);

	var programLinkStatus = gl.getProgramParameter(testRect_shaderProgramObject, gl.LINK_STATUS);

	if(programLinkStatus == false){
		var error = gl.getProgramInfoLog(testRect_shaderProgramObject);
		if(error.length > 0){
			alert("02-testRect -> Program Linkink Error: " + error);
			uninitialize_testRect();
			window.close();
		}
	}



	testRect_modelMatUniform = gl.getUniformLocation(testRect_shaderProgramObject, "u_model_matrix");
	testRect_viewMatUniform = gl.getUniformLocation(testRect_shaderProgramObject, "u_view_matrix");
	testRect_projMatUniform = gl.getUniformLocation(testRect_shaderProgramObject, "u_proj_matrix");
	testRect_samplerUniform = gl.getUniformLocation(testRect_shaderProgramObject, "u_sampler"); 
	

	
	var rect_pos = new Float32Array([
			1.0, 1.0, 0.0,
			-1.0, 1.0, 0.0,
			-1.0, -1.0, 0.0,
			1.0, -1.0, 0.0,
		]);

		

	var rect_Texcoord = new Float32Array([
			1.0, 1.0,
			0.0, 1.0,
			0.0, 0.0,
			1.0, 0.0,
		]);



	/********* Rectangle *********/
	testRect_vao_Rect = gl.createVertexArray();
	gl.bindVertexArray(testRect_vao_Rect);

		/********* Position **********/
		testRect_vbo_Rect_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, testRect_vbo_Rect_Position);
		gl.bufferData(gl.ARRAY_BUFFER,  rect_pos, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION, 3, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Texture **********/
		testRect_vbo_Rect_Texcoord = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, testRect_vbo_Rect_Texcoord);
		gl.bufferData(gl.ARRAY_BUFFER, rect_Texcoord, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0, 2, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);

	testRect_perspectiveProjectionMatrix = mat4.create();



	console.log("00-testRect -> initialize_testRect() complete");

}


function uninitialize_testRect(){


	if(testRect_vbo_Rect_Texcoord){
		gl.deleteBuffer(testRect_vbo_Rect_Texcoord);
		testRect_vbo_Rect_Texcoord = 0;
	}


	if(testRect_vbo_Rect_Position){
		gl.deleteBuffer(testRect_vbo_Rect_Position);
		testRect_vbo_Rect_Position = 0;
	}

	if(testRect_vao_Rect){
		gl.deleteVertexArray(testRect_vao_Rect);
		testRect_vao_Rect = 0;
	}


	if(testRect_shaderProgramObject){

		gl.useProgram(testRect_shaderProgramObject);

			if(testRect_fragmentShaderObject){
				gl.detachShader(testRect_shaderProgramObject, testRect_fragmentShaderObject);
				gl.deleteShader(testRect_fragmentShaderObject);
				testRect_fragmentShaderObject = 0;
			}

			if(testRect_vertexShaderObject){
				gl.detachShader(testRect_shaderProgramObject, testRect_vertexShaderObject);
				gl.deleteShader(testRect_vertexShaderObject);
				testRect_vertexShaderObject = 0;
			}

		gl.useProgram(null);
		gl.deleteProgram(testRect_shaderProgramObject);
		testRect_shaderProgramObject = 0;
	}

	console.log("00-testRect -> uninitialize_testRect() complete");

}

function display_testRect(texture){


	var testRect_modelMatrix = mat4.create();
	var testRect_viewMatrix = mat4.create();


	gl.viewport(0, 300, canvas.width/2, canvas.height/2);

	mat4.perspective(testRect_perspectiveProjectionMatrix, 45.0,
			parseFloat(canvas.width) / parseFloat(canvas.height),
			0.1,100.0);


	gl.useProgram(testRect_shaderProgramObject);


		/********** Rectangle **********/
		mat4.identity(testRect_modelMatrix);
		mat4.identity(testRect_viewMatrix);

		mat4.translate(testRect_modelMatrix, testRect_modelMatrix, [0.0, 0.0,-3.0]);

		gl.uniformMatrix4fv(testRect_modelMatUniform, false, testRect_modelMatrix);
		gl.uniformMatrix4fv(testRect_viewMatUniform, false, testRect_viewMatrix);
		gl.uniformMatrix4fv(testRect_projMatUniform, false, testRect_perspectiveProjectionMatrix);

			
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, texture);	
		gl.uniform1i(testRect_samplerUniform, 0);

		gl.bindVertexArray(testRect_vao_Rect);

			gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);

		gl.bindVertexArray(null);


		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, null);	

	gl.useProgram(null);


	gl.viewport(0, 0, canvas.width, canvas.height);
}