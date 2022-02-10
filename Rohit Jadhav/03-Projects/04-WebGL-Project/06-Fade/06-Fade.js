var canvas;
var gl;

//For Shader
var vertexShaderObject_Fade;
var fragmentShaderObject_Fade;
var shaderProgramObject_Fade;

//For Uniform
var mvpUniform_Fade;
var fadeUniform_Fade;


//For Projection
var orthoMatrix;


//For Rect
var vao_Rect_Fade;
var vbo_Rect_Position_Fade;


function initialize_Fade(){



	vertexShaderObject_Fade = gl.createShader(gl.VERTEX_SHADER);
	var szVertexShaderSourceCode = 
		"#version 300 es" +
		"\n" +
		"in vec4 vPosition;" +
		"uniform mat4 u_mvp_matrix;" +
		"void main() {" +
			"gl_Position = u_mvp_matrix * vPosition;" +
		"}";

	gl.shaderSource(vertexShaderObject_Fade, szVertexShaderSourceCode);

	gl.compileShader(vertexShaderObject_Fade);

	var  shaderCompileStatus = gl.getShaderParameter(vertexShaderObject_Fade, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(vertexShaderObject_Fade);
		if(error.length > 0){
			alert("06-Fade : Vertex Shader Compilation Error: " + error);
			uninitialize();
			window.close();
		}
	}


	fragmentShaderObject_Fade = gl.createShader(gl.FRAGMENT_SHADER);
	var szFragmentShaderSourceCode = 
		"#version 300 es" +
		"\n" +
		"precision highp float;" +

		"uniform float u_fade;" +

		"out vec4 FragColor;" +
		"void main(void) {" +
			"FragColor = vec4(0.0, 0.0, 0.0, u_fade);" +
		"}";

	gl.shaderSource(fragmentShaderObject_Fade, szFragmentShaderSourceCode);
	gl.compileShader(fragmentShaderObject_Fade);

	shaderCompileStatus = gl.getShaderParameter(fragmentShaderObject_Fade, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(fragmentShaderObject_Fade);
		if(error.length > 0){
			alert("06-Fade : Fragment Shader Compilation Error: "+ error);
			uninitialize();
			window.close();
		}
	}


	shaderProgramObject_Fade = gl.createProgram();

	gl.attachShader(shaderProgramObject_Fade, vertexShaderObject_Fade);
	gl.attachShader(shaderProgramObject_Fade, fragmentShaderObject_Fade);

	gl.bindAttribLocation(shaderProgramObject_Fade, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");

	gl.linkProgram(shaderProgramObject_Fade);

	var programLinkStatus = gl.getProgramParameter(shaderProgramObject_Fade, gl.LINK_STATUS);

	if(programLinkStatus == false){
		var error = gl.getProgramInfoLog(shaderProgramObject_Fade);
		if(error.length > 0){
			alert("06-Fade : Program Linkink Error: " + error);
			uninitialize();
			window.close();
		}
	}


	mvpUniform_Fade = gl.getUniformLocation(shaderProgramObject_Fade, "u_mvp_matrix");
	fadeUniform_Fade = gl.getUniformLocation(shaderProgramObject_Fade, "u_fade");


	var rect_Position = new Float32Array([
						1.0, 1.0, 0.0,
						-1.0, 1.0, 0.0,
						-1.0, -1.0, 0.0,
						1.0, -1.0, 0.0
					]);



	/********* Rectangle *********/
	vao_Rect_Fade = gl.createVertexArray();
	gl.bindVertexArray(vao_Rect_Fade);

		/********* Position **********/
		vbo_Rect_Position_Fade = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Rect_Position_Fade);
		gl.bufferData(gl.ARRAY_BUFFER,  rect_Position, gl.DYNAMIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);


	orthoMatrix = mat4.create();

}

function uninitialize_Fade(){

	if(vbo_Rect_Position_Fade){
		gl.deleteBuffer(vbo_Rect_Position_Fade);
		vbo_Rect_Position_Fade = 0;
	}

	if(vao_Rect_Fade){
		gl.deleteVertexArray(vao_Rect_Fade);
		vao_Rect_Fade = 0;
	}


	if(shaderProgramObject_Fade){

		gl.useProgram(shaderProgramObject_Fade);

			if(fragmentShaderObject_Fade){
				gl.detachShader(shaderProgramObject_Fade, fragmentShaderObject_Fade);
				gl.deleteShader(fragmentShaderObject_Fade);
				fragmentShaderObject_Fade = 0;
			}

			if(vertexShaderObject_Fade){
				gl.detachShader(shaderProgramObject_Fade, vertexShaderObject_Fade);
				gl.deleteShader(vertexShaderObject_Fade);
				vertexShaderObject_Fade = 0;
			}

		gl.useProgram(null);
		gl.deleteProgram(shaderProgramObject_Fade);
		shaderProgramObject_Fade = 0;
	}

}


function draw_Fade(fadeValue){

	var modelViewMatrix = mat4.create();
	var modelViewProjectionMatrix  = mat4.create();

	//gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

	var viewPortWidth = canvas.width;
	var viewPortHeight = canvas.height;

	mat4.identity(orthoMatrix);
	mat4.ortho(orthoMatrix,
		-viewPortWidth / 2.0, viewPortWidth / 2.0,	// L, R
		-viewPortHeight/ 2.0, viewPortHeight / 2.0,	// B, T
		-1.0, 1.0);							// N, F

	gl.enable(gl.BLEND);
	gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

	gl.useProgram(shaderProgramObject_Fade);


		/********** Rectangle **********/
		mat4.identity(modelViewMatrix);
		mat4.identity(modelViewProjectionMatrix);
		mat4.multiply(modelViewProjectionMatrix, orthoMatrix, modelViewMatrix);

		gl.uniformMatrix4fv(mvpUniform_Fade, false, modelViewProjectionMatrix);
		gl.uniform1f(fadeUniform_Fade, fadeValue);


		var Rect_Vertices = new Float32Array([
			viewPortWidth / 2.0, viewPortHeight / 2.0, 0.0,
			-viewPortWidth / 2.0, viewPortHeight / 2.0, 0.0,
			-viewPortWidth / 2.0, -viewPortHeight / 2.0, 0.0,
			viewPortWidth / 2.0, -viewPortHeight / 2.0, 0.0,
		]);

		gl.bindVertexArray(vao_Rect_Fade);

			gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Rect_Position_Fade);
			gl.bufferData(gl.ARRAY_BUFFER, Rect_Vertices, gl.DYNAMIC_DRAW);
			gl.bindBuffer(gl.ARRAY_BUFFER, null);

			gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
		gl.bindVertexArray(null);

			
	gl.useProgram(null);

	gl.disable(gl.BLEND);

}
