var canvas = null;
var gl = null;
var bIsFullScreen = false;
var canvas_original_width = 0;
var canvas_original_height = 0;

const WebGLMacros = {
	AMC_ATTRIBUTE_POSITION:0,
	AMC_ATTRIBUTE_COLOR:1,
	AMC_ATTRIBUTE_NORMAL:2,
	AMC_ATTRIBUTE_TEXCOORD0:3
};


//For Shader
var vertexShaderObject;
var fragmentShaderObject;
var shaderProgramObject;

//For Uniform
var mvpUniform;

//For matrix
var perspectiveProjectionMatrix;
var modelViewMatrix;
var modelViewProjectionMatrix;


//For I
var vao_I;
var vbo_I_Position;
var vbo_I_Color;

//For N
var vao_N;
var vbo_N_Position;
var vbo_N_Color;

//For D
var vao_D;
var vbo_D_Position;
var vbo_D_Color;

//For A
var vao_A;
var vbo_A_Position;
var vbo_A_Color;

//For Flag
var vao_Flag;
var vbo_Flag_Position;
var vbo_Flag_Color;



//For Starting Animation we need requestAnimationFrame()

var requestAnimationFrame = 
	window.requestAnimationFrame || window.webkitRequestAnimationFrame ||
	window.mozRequestAnimationFrame || window.oRequestAnimationFrame || 
	window.msRequestAnimationFrame || 
	null;

//For Stoping Animation we need cancelAnimationFrame()

var cancelAnimationFrame = 
	window.cancelAnimationFrame || 
	window.webkitCancelRequestAnimationFrame || window.webkitCancelAnimationFrame || 
	window.mozCancelRequestAnimationFrame || window.mozCancelAnimationFrame ||
	window.oCancelRequestAnimationFrame || window.oCancelAnimationFrame ||
	window.msCancelRequestAnimationFrame || window.msCancelAnimationFrame ||
	null;

function main(){

	canvas = document.getElementById("13-StaticIndia-RRJ");
	if(!canvas)
		console.log("Obtaining Canvas Failed!!\n");
	else
		console.log("Canvas Obtained!!\n");


	canvas_original_width = canvas.width;
	canvas_original_height = canvas.height;

	window.addEventListener("keydown", keyDown, false);
	window.addEventListener("click", mouseDown, false);
	window.addEventListener("resize", resize, false);

	initialize();

	resize();
	draw();

}

function toggleFullScreen(){

	var fullscreen_element = 
		document.fullscreenElement || 
		document.wegkitFullscreenElement || 
		document.mozFullScreenElement ||
		document.msFullscreenElement ||
		null;

	if(fullscreen_element == null){

		if(canvas.requestFullscreen)
			canvas.requestFullscreen();
		else if(canvas.webkitRequestFullscreen)
			canvas.webkitRequestFullscreen();
		else if(canvas.mozRequestFullScreen)
			canvas.mozRequestFullScreen();
		else if(canvas.msRequestFullscreen)
			canvas.msRequestFullscreen();

		bIsFullScreen = true;
	}
	else{

		if(document.exitFullscreen)
			document.exitFullscreen();
		else if(document.mozCancelFullScreen)
			document.mozCancelFullScreen();
		else if(document.webkitExitFullscreen)
			document.webkitExitFullscreen();
		else if(document.msExitFullscreen)
			document.msExitFullscreen();

		bIsFullScreen = false;
	}
}


function keyDown(event){

	switch(event.keyCode){
		case 27:
			uninitialize();
			window.close();
			break;

		case 70:
			toggleFullScreen();
			break;
	}
}

function mouseDown(){

}

function initialize(){


	gl = canvas.getContext("webgl2");
	if(gl == null){
		console.log("Obtaining Context Failed!!\n");
		return;
	}
	else
		console.log("Context Obtained!!\n");

	gl.viewportWidth = canvas.width;
	gl.viewportHeight = canvas.height;


	vertexShaderObject = gl.createShader(gl.VERTEX_SHADER);

	var szVertexShaderSourceCode = 
		"#version 300 es" +
		"\n" +
		"in vec4 vPosition;" +
		"in vec4 vColor;" + 
		"out vec4 outColor;" +
 		"uniform mat4 u_mvp_matrix;" +
		"void main(){" +
			"gl_Position = u_mvp_matrix * vPosition;" +
			"outColor = vColor;" + 
		"}";

	gl.shaderSource(vertexShaderObject, szVertexShaderSourceCode);
	gl.compileShader(vertexShaderObject);

	var shaderCompileStatus = gl.getShaderParameter(vertexShaderObject, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(vertexShaderObject);
		if(error.length > 0){
			alert("VertexShader Compilation Error : " + error);
			uninitialize();
			window.close();

		}
	}


	fragmentShaderObject = gl.createShader(gl.FRAGMENT_SHADER);
	var szFragmentShaderSourceCode = 
		"#version 300 es" +
		"\n" +
		"precision highp float;" +
		"in vec4 outColor;" +
		"out vec4 FragColor;" +
		"void main() {" +
			"FragColor = outColor;" +
		"}";

	gl.shaderSource(fragmentShaderObject, szFragmentShaderSourceCode);
	gl.compileShader(fragmentShaderObject);

	shaderCompileStatus = gl.getShaderParameter(fragmentShaderObject,  gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(fragmentShaderObject);
		if(error.length > 0){
			alert("Fragment Shader Compilation Error : " + error);
			uninitialize();
			window.close();

		}
	}



	shaderProgramObject = gl.createProgram();

	gl.attachShader(shaderProgramObject, vertexShaderObject);
	gl.attachShader(shaderProgramObject, fragmentShaderObject);

	gl.bindAttribLocation(shaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
	gl.bindAttribLocation(shaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_COLOR, "vColor");

	gl.linkProgram(shaderProgramObject);

	var programLinkStatus = gl.getProgramParameter(shaderProgramObject, gl.LINK_STATUS);

	if(programLinkStatus == false){
		var error = gl.getProgramInfoLog(shaderProgramObject);
		if(error.length > 0){
			alert("Program Linking Error : " + error);
			uninitialize();
			window.close();
		}
	}


	mvpUniform = gl.getUniformLocation(shaderProgramObject, "u_mvp_matrix");



	/********** Position and Color **********/
	
	var i_Position = new Float32Array([
					//Top
					-0.3, 1.0, 0.0,
					0.3, 1.0, 0.0,

					-0.3, 1.010, 0.0,
					0.3, 1.010, 0.0,
					-0.3, 0.990, 0.0,
					0.3, 0.990, 0.0,

					-0.3, 1.020, 0.0,
					0.3, 1.020, 0.0,
					-0.3, 0.980, 0.0,
					0.3, 0.980, 0.0,



			
					//Mid
					0.0, 1.0, 0.0,
					0.0, -1.0, 0.0,

					0.01, 1.0, 0.0,
					0.01, -1.0, 0.0,
					-0.01, 1.0, 0.0,
					-0.01, -1.0, 0.0,

					0.02, 1.0, 0.0,
					0.02, -1.0, 0.0,
					-0.02, 1.0, 0.0,
					-0.02, -1.0, 0.0,
	

					//Bottom
					-0.3, -1.0, 0.0,
					0.3, -1.0, 0.0,

					-0.3, -1.010, 0.0,
					0.3, -1.010, 0.0,
					-0.3, -0.990, 0.0,
					0.3, -0.990, 0.0,

					-0.3, -1.020, 0.0,
					0.3, -1.020, 0.0,
					-0.3, -0.980, 0.0,
					0.3, -0.980, 0.0,



				]);

	var i_Color = new Float32Array([
					//Top
					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,

					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,
					
					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,


					//Mid
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					

					//Bottom
					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,

					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,

					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,


				]);

	var n_Position = new Float32Array([

					//Top
					0.0, 1.06, 0.0,
					0.0, -1.06, 0.0,

					0.01, 1.06, 0.0,
					0.01, -1.06, 0.0,
					-0.01, 1.06, 0.0,
					-0.01, -1.06, 0.0,

					0.02, 1.06, 0.0,
					0.02, -1.06, 0.0,
					-0.02, 1.06, 0.0,
					-0.02, -1.06, 0.0,


					//Mid
					0.75, 1.06, 0.0,
					0.75, -1.06, 0.0,

					0.76, 1.06, 0.0,
					0.76, -1.06, 0.0,
					0.74, 1.06, 0.0,
					0.74, -1.06, 0.0,

					0.77, 1.06, 0.0,
					0.77, -1.06, 0.0,
					0.73, 1.06, 0.0,
					0.73, -1.06, 0.0,


					//Bottom
					0.0, 1.06, 0.0,
					0.75, -1.06, 0.0,

					0.01, 1.06, 0.0,
					0.76, -1.06, 0.0,
					-0.01, 1.06, 0.0,
					0.74, -1.06, 0.0,

					0.02, 1.06, 0.0,
					0.77, -1.06, 0.0,
					-0.02, 1.06, 0.0,
					0.73, -1.06, 0.0

				]);

	var n_Color = new Float32Array([
					//Top
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					//Mid
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,


					//Bottom
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,
				]);


	var d_Position = new Float32Array([

					//Left
					0.0, 1.0, 0.0,
					0.0, -1.0, 0.0,

					0.01, 1.0, 0.0,
					0.01, -1.0, 0.0,
					-0.01, 1.0, 0.0,
					-0.01, -1.0, 0.0,

					0.02, 1.0, 0.0,
					0.02, -1.0, 0.0,
					-0.02, 1.0, 0.0,
					-0.02, -1.0, 0.0,



					//Top
					-0.1, 1.0, 0.0,
					0.6, 1.0, 0.0,

					-0.1, 1.01, 0.0,
					0.6, 1.01, 0.0,
					-0.1, 0.990, 0.0,
					0.6, 0.990, 0.0,

					-0.1, 1.02, 0.0,
					0.6, 1.02, 0.0,
					-0.1, 0.980, 0.0,
					0.6, 0.980, 0.0,



					//Bottom
					-0.1, -1.0, 0.0,
					0.6, -1.0, 0.0,

					-0.1, -1.01, 0.0,
					0.6, -1.01, 0.0,
					-0.1, -0.990, 0.0,
					0.6, -0.990, 0.0,

					-0.1, -1.02, 0.0,
					0.6, -1.02, 0.0,
					-0.1, -0.980, 0.0,
					0.6, -0.980, 0.0,


					//Right
					0.6, 1.0, 0.0,
					0.6, -1.0, 0.0,

					0.61, 1.0, 0.0,
					0.61, -1.0, 0.0,
					0.59, 1.0, 0.0,
					0.59, -1.0, 0.0,

					0.62, 1.0, 0.0,
					0.62, -1.0, 0.0,
					0.58, 1.0, 0.0,
					0.58, -1.0, 0.0,

				]);

	var d_Color = new Float32Array([

					//Left
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					//Top
					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,

					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,

					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,

					//Bottom
					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,

					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,

					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,

					//Right
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,
				]);

	var a_Position = new Float32Array([

					//Left
					0.0, 1.06, 0.0,
					-0.5, -1.06, 0.0,

					0.01, 1.06, 0.0,
					-0.49, -1.06, 0.0,
					0.01, 1.06, 0.0,
					-0.51, -1.06, 0.0,

					0.02, 1.06, 0.0,
					-0.48, -1.06, 0.0,
					0.02, 1.06, 0.0,
					-0.52, -1.06, 0.0,

					//Right
					0.0, 1.06, 0.0,
					0.5, -1.06, 0.0,

					0.01, 1.06, 0.0,
					0.49, -1.06, 0.0,
					0.01, 1.06, 0.0,
					0.51, -1.06, 0.0,

					0.02, 1.06, 0.0,
					0.48, -1.06, 0.0,
					0.02, 1.06, 0.0,
					0.52, -1.06, 0.0,


				]);

	var a_Color = new Float32Array([
					//Left
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,


					//Right
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,
				]);

	var flag_Position = new Float32Array([

					//Orange
					-0.207, 0.06, 0.0,
					0.227, 0.06, 0.0,

					-0.207, 0.07, 0.0,
					0.227, 0.07, 0.0,
					-0.207, 0.05, 0.0,
					0.227, 0.05, 0.0,

					-0.207, 0.08, 0.0,
					0.227, 0.08, 0.0,
					-0.207, 0.04, 0.0,
					0.227, 0.04, 0.0,

					//White
					-0.218, 0.0, 0.0,
					0.239, 0.0, 0.0,

					-0.218, 0.01, 0.0,
					0.239, 0.01, 0.0,
					-0.218, -0.01, 0.0,
					0.239, -0.01, 0.0,

					-0.218, 0.02, 0.0,
					0.239, 0.02, 0.0,
					-0.218, -0.02, 0.0,
					0.239, -0.02, 0.0,

					//Green
					-0.235, -0.06, 0.0,
					0.245, -0.06, 0.0,

					-0.235, -0.07, 0.0,
					0.245, -0.07, 0.0,
					-0.235, -0.05, 0.0,
					0.245, -0.05, 0.0,

					-0.235, -0.08, 0.0,
					0.245, -0.08, 0.0,
					-0.235, -0.04, 0.0,
					0.245, -0.04, 0.0,
				]);

	var flag_Color = new Float32Array([
					//Orange
					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,

					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,

					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,

					//White
					1.0, 1.0, 1.0,
					1.0, 1.0, 1.0,

					1.0, 1.0, 1.0,
					1.0, 1.0, 1.0,
					1.0, 1.0, 1.0,
					1.0, 1.0, 1.0,

					1.0, 1.0, 1.0,
					1.0, 1.0, 1.0,
					1.0, 1.0, 1.0,
					1.0, 1.0, 1.0,


					//Green
					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,

					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,

					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,
				]);





	/********** I **********/
	vao_I = gl.createVertexArray();
	gl.bindVertexArray(vao_I);

		/********** Position **********/
		vbo_I_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_I_Position);
		gl.bufferData(gl.ARRAY_BUFFER,
					i_Position,
					gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Color **********/
		vbo_I_Color = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_I_Color);
		gl.bufferData(gl.ARRAY_BUFFER,
					i_Color,
					gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_COLOR,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);



	/********** N **********/
	vao_N = gl.createVertexArray();
	gl.bindVertexArray(vao_N);

		/********** Position **********/
		vbo_N_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_N_Position);
		gl.bufferData(gl.ARRAY_BUFFER,
					n_Position,
					gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Color **********/
		vbo_N_Color = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_N_Color);
		gl.bufferData(gl.ARRAY_BUFFER,
					n_Color,
					gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_COLOR,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);



	/********** D **********/
	vao_D = gl.createVertexArray();
	gl.bindVertexArray(vao_D);

		/********** Position **********/
		vbo_D_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_D_Position);
		gl.bufferData(gl.ARRAY_BUFFER,
					d_Position,
					gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Color **********/
		vbo_D_Color = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_D_Color);
		gl.bufferData(gl.ARRAY_BUFFER,
					d_Color,
					gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_COLOR,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);



	/********** A **********/
	vao_A = gl.createVertexArray();
	gl.bindVertexArray(vao_A);

		/********** Position **********/
		vbo_A_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_A_Position);
		gl.bufferData(gl.ARRAY_BUFFER,
					a_Position,
					gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Color **********/
		vbo_A_Color = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_A_Color);
		gl.bufferData(gl.ARRAY_BUFFER,
					a_Color,
					gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_COLOR,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);


	/********** Flag **********/
	vao_Flag = gl.createVertexArray();
	gl.bindVertexArray(vao_Flag);

		/********** Position **********/
		vbo_Flag_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Flag_Position);
		gl.bufferData(gl.ARRAY_BUFFER,
					flag_Position,
					gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Color **********/
		vbo_Flag_Color = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Flag_Color);
		gl.bufferData(gl.ARRAY_BUFFER,
					flag_Color,
					gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_COLOR,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);


	gl.enable(gl.DEPTH_TEST);
	gl.depthFunc(gl.LEQUAL);

	//gl.enable(gl.PROGRAM_POINT_SIZE);
	
	gl.clearDepth(1.0);
	gl.clearColor(0.0, 0.0, 0.0, 1.0);


	perspectiveProjectionMatrix = mat4.create();
	modelViewMatrix = mat4.create();
	modelViewProjectionMatrix = mat4.create();
}


function uninitialize(){

	//Flag
	if (vbo_Flag_Color) {
		gl.deleteBuffer(vbo_Flag_Color);
		vbo_Flag_Color = 0;
	}

	if (vbo_Flag_Position) {
		gl.deleteBuffer(vbo_Flag_Position);
		vbo_Flag_Position = 0;
	}

	if (vao_Flag) {
		gl.deleteVertexArray(vao_Flag);
		vao_Flag = 0;
	}


	//A
	if (vbo_A_Color) {
		gl.deleteBuffer(vbo_A_Color);
		vbo_A_Color = 0;
	}

	if (vbo_A_Position) {
		gl.deleteBuffer(vbo_A_Position);
		vbo_A_Position = 0;
	}

	if (vao_A) {
		gl.deleteVertexArray(vao_A);
		vao_A = 0;
	}

	//D
	if (vbo_D_Color) {
		gl.deleteBuffer(vbo_D_Color);
		vbo_D_Color = 0;
	}

	if (vbo_D_Position) {
		gl.deleteBuffer(vbo_D_Position);
		vbo_D_Position = 0;
	}

	if (vao_D) {
		gl.deleteVertexArray(vao_D);
		vao_D = 0;
	}

	//N
	if (vbo_N_Color) {
		gl.deleteBuffer(vbo_N_Color);
		vbo_N_Color = 0;
	}

	if (vbo_N_Position) {
		gl.deleteBuffer(vbo_N_Position);
		vbo_N_Position = 0;
	}

	if (vao_N) {
		gl.deleteVertexArray(vao_N);
		vao_N = 0;
	}

	//I
	if (vbo_I_Color) {
		gl.deleteBuffer(vbo_I_Color);
		vbo_I_Color = 0;
	}

	if (vbo_I_Position) {
		gl.deleteBuffer(vbo_I_Position);
		vbo_I_Position = 0;
	}

	if (vao_I) {
		gl.deleteVertexArray(vao_I);
		vao_I = 0;
	}


	if(shaderProgramObject){

		gl.useProgram(shaderProgramObject);

			if(fragmentShaderObject){
				gl.detachShader(shaderProgramObject, fragmentShaderObject);
				gl.deleteShader(fragmentShaderObject);
				fragmentShaderObject = null;
			}

			if(vertexShaderObject){
				gl.detachShader(shaderProgramObject, vertexShaderObject);
				gl.deleteShader(vertexShaderObject);
				vertexShaderObject = null;
			}

		gl.useProgram(null);
		gl.deleteProgram(shaderProgramObject);
		shaderProgramObject = null;
	}
}

function resize(){

	if(bIsFullScreen == true){
		canvas.width = window.innerWidth;
		canvas.height = window.innerHeight;
	}
	else{
		canvas.width = canvas_original_width;
		canvas.height = canvas_original_height;
	}

	gl.viewport(0, 0, canvas.width, canvas.height);

	mat4.perspective(perspectiveProjectionMatrix, 
					45.0,
					parseFloat(canvas.width) / parseFloat(canvas.height),
					0.1,
					100.0);
}

function draw(){

	var modelViewMatrix = mat4.create();
	var modelViewProjectionMatrix = mat4.create();
	

	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);


	gl.useProgram(shaderProgramObject);

		//I
		My_I(-2.0, 0.0, -8.0, 20.0);

		//N
		My_N(-1.35, 0.0, -8.0, 20.0);

		//D
		My_D(-0.15, 0.0, -8.0, 20.0);

		//I
		My_I(1.02, 0.0, -8.0, 20.0);

		//A
		My_A(2.0, 0.0, -8.0, 20.0);

		//Flag
		My_Flag(2.0, 0.0, -8.0, 20.0);
		


	gl.useProgram(null);


	


	requestAnimationFrame(draw, canvas);
}




function My_I(x, y,  z,  fWidth) {

	mat4.identity(modelViewMatrix);
	mat4.identity(modelViewProjectionMatrix);

	mat4.translate(modelViewMatrix, modelViewMatrix, [x, y, z]);
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);	

	
	gl.bindVertexArray(vao_I);
	gl.drawArrays(gl.LINES, 0, 30);
	gl.bindVertexArray(null);
}

function My_N(x, y, z, fWidth) {

	mat4.identity(modelViewMatrix);
	mat4.identity(modelViewProjectionMatrix);

	mat4.translate(modelViewMatrix, modelViewMatrix, [x, y, z]);
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);	

	gl.bindVertexArray(vao_N);
	gl.drawArrays(gl.LINES, 0, 30);
	gl.bindVertexArray(null);
}


function My_D(x, y, z, fWidth) {

	mat4.identity(modelViewMatrix);
	mat4.identity(modelViewProjectionMatrix);

	mat4.translate(modelViewMatrix, modelViewMatrix, [x, y, z]);
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);	

	gl.bindVertexArray(vao_D);
	gl.drawArrays(gl.LINES, 0, 40);
	gl.bindVertexArray(null);
}

function My_A(x, y, z, fWidth) {

	mat4.identity(modelViewMatrix);
	mat4.identity(modelViewProjectionMatrix);

	mat4.translate(modelViewMatrix, modelViewMatrix, [x, y, z]);
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);	

	gl.bindVertexArray(vao_A);
	gl.drawArrays(gl.LINES, 0, 20);
	gl.bindVertexArray(null);
}



function My_Flag(x, y, z, fWidth) {

	mat4.identity(modelViewMatrix);
	mat4.identity(modelViewProjectionMatrix);

	mat4.translate(modelViewMatrix, modelViewMatrix, [x, y, z]);
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);	

	gl.bindVertexArray(vao_Flag);
	gl.drawArrays(gl.LINES, 0, 30);
	gl.bindVertexArray(null);
}

