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
var vbo_D_Color
var d_Color = new Float32Array([

					//Left
					1.0, 0.6, 0.2, 0.0,
					0.0705, 0.533, 0.0274,0.0,

					1.0, 0.6, 0.2,0.0,
					0.0705, 0.533, 0.0274,0.0,
					1.0, 0.6, 0.2,0.0,
					0.0705, 0.533, 0.0274,0.0,

					1.0, 0.6, 0.2,0.0,
					0.0705, 0.533, 0.0274,0.0,
					1.0, 0.6, 0.2,0.0,
					0.0705, 0.533, 0.0274,0.0,

					//Top
					1.0, 0.6, 0.2,0.0,
					1.0, 0.6, 0.2,0.0,

					1.0, 0.6, 0.2,0.0,
					1.0, 0.6, 0.2,0.0,
					1.0, 0.6, 0.2,0.0,
					1.0, 0.6, 0.2,0.0,

					1.0, 0.6, 0.2,0.0,
					1.0, 0.6, 0.2,0.0,
					1.0, 0.6, 0.2,0.0,
					1.0, 0.6, 0.2,0.0,

					//Bottom
					0.0705, 0.533, 0.0274,0.0,
					0.0705, 0.533, 0.0274,0.0,

					0.0705, 0.533, 0.0274,0.0,
					0.0705, 0.533, 0.0274,0.0,
					0.0705, 0.533, 0.0274,0.0,
					0.0705, 0.533, 0.0274,0.0,

					0.0705, 0.533, 0.0274,0.0,
					0.0705, 0.533, 0.0274,0.0,
					0.0705, 0.533, 0.0274,0.0,
					0.0705, 0.533, 0.0274,0.0,

					//Right
					1.0, 0.6, 0.2,0.0,
					0.0705, 0.533, 0.0274,0.0,

					1.0, 0.6, 0.2,0.0,
					0.0705, 0.533, 0.0274,0.0,
					1.0, 0.6, 0.2,0.0,
					0.0705, 0.533, 0.0274,0.0,

					1.0, 0.6, 0.2,0.0,
					0.0705, 0.533, 0.0274,0.0,
					1.0, 0.6, 0.2,0.0,
					0.0705, 0.533, 0.0274,0.0,
			]);

var fD_Fading = 0.0;


//For A
var vao_A;
var vbo_A_Position;
var vbo_A_Color;

//For V
var vao_V;
var vbo_V_Position;
var vbo_V_Color;

//For F
var vao_F;
var vbo_F_Position;
var vbo_F_Color;

//For Flag
var vao_Flag;
var vbo_Flag_Position;
var vbo_Flag_Color;


//For Plane's Triangle Part
var vao_Plane_Triangle;
var vbo_Plane_Triangle_Position;
var vbo_Plane_Triangle_Color;

//For Plane's Rectangle Part
var vao_Plane_Rect;
var vbo_Plane_Rect_Position;
var vbo_Plane_Rect_Color;

//For Plane's Polygon Part
var vao_Plane_Polygon;
var vbo_Plane_Polygon_Position;
var vbo_Plane_Polygon_Color;

//For Fading Flag
var vao_Fading_Flag;
var vbo_Fading_Flag_Position;
var vbo_Fading_Flag_Color;
var fading_Flag_Position = new Float32Array([
		//Top
		-1.0, 0.1, 0.0,
		-0.50, 0.1, 0.0,

		-1.0, 0.11, 0.0,
		-0.50, 0.11, 0.0,
		-1.0, 0.09, 0.0,
		-0.50, 0.09, 0.0,

		-1.0, 0.12, 0.0,
		-0.50, 0.12, 0.0,
		-1.0, 0.08, 0.0,
		-0.50, 0.08, 0.0,

		//Middle
		-1.0, 0.0, 0.0,
		-0.50, 0.0, 0.0,

		-1.0, 0.01, 0.0,
		-0.50, 0.01, 0.0,
		-1.0, -0.01, 0.0,
		-0.50, -0.01, 0.0,

		-1.0, 0.02, 0.0,
		-0.50, 0.02, 0.0,
		-1.0, -0.02, 0.0,
		-0.50, -0.02, 0.0,

		//Bottom
		-1.0, -0.1, 0.0,
		-0.5, -0.1, 0.0,

		-1.0, -0.11, 0.0,
		-0.5, -0.11, 0.0,
		-1.0, -0.09, 0.0,
		-0.5, -0.09, 0.0,

		-1.0, -0.12, 0.0,
		-0.5, -0.12, 0.0,
		-1.0, -0.08, 0.0,
		-0.5, -0.08, 0.0,
	]);


//For Plane Movement and Translation
var NOT_REACH =  0;
var HALF_WAY = 1;
var REACH = 2;
var END = 3;


var Plane1_Count = 1000.0;
var Plane2_Count = 1000.0;
var Plane3_Count = 1000.0;

var bPlane1Reached = NOT_REACH;
var bPlane2Reached = NOT_REACH;
var bPlane3Reached = NOT_REACH;
var iFadingFlag1 = 0;
var iFadingFlag2 = 0;
var iFadingFlag3 = 0;


//For Sequence
var iSequence = 1;


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

	canvas = document.getElementById("14-DynamicIndia-RRJ");
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

	toggleFullScreen();
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

	/*var d_Color = new Float32Array([

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
				]);-*/




	var v_Position = new Float32Array([

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

	var v_Color = new Float32Array([
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
					-0.207, 0.1, 0.0,
					0.227, 0.1, 0.0,

					-0.207, 0.11, 0.0,
					0.227, 0.11, 0.0,
					-0.207, 0.09, 0.0,
					0.227, 0.09, 0.0,

					-0.207, 0.12, 0.0,
					0.227, 0.12, 0.0,
					-0.207, 0.08, 0.0,
					0.227, 0.08, 0.0,

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
					-0.245, -0.1, 0.0,
					0.255, -0.1, 0.0,

					-0.245, -0.11, 0.0,
					0.255, -0.11, 0.0,
					-0.245, -0.09, 0.0,
					0.255, -0.09, 0.0,

					-0.245, -0.12, 0.0,
					0.255, -0.12, 0.0,
					-0.245, -0.08, 0.0,
					0.255, -0.08, 0.0,
				]);


	var flag_Color = new Float32Array([
					//Orange
					0.0, 0.0, 0.0,
					1.0, 0.6, 0.2,

					0.0, 0.0, 0.0,
					1.0, 0.6, 0.2,
					0.0, 0.0, 0.0,
					1.0, 0.6, 0.2,

					0.0, 0.0, 0.0,
					1.0, 0.6, 0.2,
					0.0, 0.0, 0.0,
					1.0, 0.6, 0.2,

					//White
					0.0, 0.0, 0.0,
					1.0, 1.0, 1.0,

					0.0, 0.0, 0.0,
					1.0, 1.0, 1.0,
					0.0, 0.0, 0.0,
					1.0, 1.0, 1.0,

					0.0, 0.0, 0.0,
					1.0, 1.0, 1.0,
					0.0, 0.0, 0.0,
					1.0, 1.0, 1.0,


					//Green
					0.0, 0.0, 0.0,
					0.0705, 0.533, 0.0274,

					0.0, 0.0, 0.0,
					0.0705, 0.533, 0.0274,
					0.0, 0.0, 0.0,
					0.0705, 0.533, 0.0274,

					0.0, 0.0, 0.0,
					0.0705, 0.533, 0.0274,
					0.0, 0.0, 0.0,
					0.0705, 0.533, 0.0274,
				]);


	var a_Position = new Float32Array([
					0.0, 1.06, 0.0,
					-0.5, -1.06, 0.0,

					0.0, 1.06, 0.0,
					0.5, -1.06, 0.0,

					-0.250, 0.0, 0.0,
					0.25, 0.0, 0.0,
				]);


	var a_Color = new Float32Array([
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,
				]);


	//For F
	var f_Position = new Float32Array([
					0.10, 1.0, 0.0,
					0.10, -1.0, 0.0,

					0.00, 1.0, 0.0,
					0.90, 1.0, 0.0,

					0.10, 0.1, 0.0,
					0.80, 0.1, 0.0,
				]);

	var f_Color = new Float32Array([
					1.0, 0.6, 0.2,
					0.0705, 0.533, 0.0274,

					1.0, 0.6, 0.2,
					1.0, 0.6, 0.2,

					0.0705, 0.533, 0.0274,
					0.0705, 0.533, 0.0274,
				]);
	


	//For Plane Triangle Part
	var plane_Triangle_Position = new Float32Array([
					//Front
					5.0, 0.0, 0.0,
					2.50, 0.65, 0.0,
					2.50, -0.65, 0.0,
				]);

	var plane_Triangle_Color = new Float32Array([
		//Front
		0.7294, 0.8862, 0.9333,	//Power Blue
		0.7294, 0.8862, 0.9333,
		0.7294, 0.8862, 0.9333,
	]);


	//For Plane Rect Part
	var plane_Rect_Position = new Float32Array([
					//Middle
					2.50, 0.65, 0.0,
					-2.50, 0.65, 0.0,
					-2.50, -0.65, 0.0,
					2.50, -0.65, 0.0,

					//Upper_Fin
					0.75, 0.65, 0.0,
					-1.20, 2.5, 0.0,
					-2.50, 2.5, 0.0,
					-2.0, 0.65, 0.0,

					//Lower_Fin
					0.75, -0.65, 0.0,
					-1.20, -2.50, 0.0,
					-2.50, -2.50, 0.0,
					-2.0, -0.65, 0.0,

					//Back
					-2.50, 0.65, 0.0,
					-3.0, 0.75, 0.0,
					-3.0, -0.75, 0.0,
					-2.5, -0.65, 0.0,
				]);


	var plane_Rect_Color = new Float32Array([
					//Middle
					0.7294, 0.8862, 0.9333,
					0.7294, 0.8862, 0.9333,
					0.7294, 0.8862, 0.9333,
					0.7294, 0.8862, 0.9333,

					//Upper_Fin
					0.7294, 0.8862, 0.9333,
					0.7294, 0.8862, 0.9333,
					0.7294, 0.8862, 0.9333,
					0.7294, 0.8862, 0.9333,

					//Lower_Fin
					0.7294, 0.8862, 0.9333,
					0.7294, 0.8862, 0.9333,
					0.7294, 0.8862, 0.9333,
					0.7294, 0.8862, 0.9333,

					//Back
					0.7294, 0.8862, 0.9333,
					0.7294, 0.8862, 0.9333,
					0.7294, 0.8862, 0.9333,
					0.7294, 0.8862, 0.9333
				]);


	//For Plane Polygon Part
	var plane_Polygon_Position = new Float32Array([
					//Upper Tail
					-3.0, 0.75, 0.0,
					-3.90, 1.5, 0.0,
					-4.5, 1.5, 0.0,
					-4.0, 0.0, 0.0,
					-3.0, 0.0, 0.0,

					//Lower Tail
					-3.0, -0.75, 0.0,
					-3.90, -1.5, 0.0,
					-4.5, -1.5, 0.0,
					-4.0, 0.0, 0.0,
					-3.0, 0.0, 0.0,
				]);

	var plane_Polygon_Color = new Float32Array([
					//Upper Tail
					0.7294, 0.8862, 0.9333,
					0.7294, 0.8862, 0.9333,
					0.7294, 0.8862, 0.9333,
					0.7294, 0.8862, 0.9333,
					0.7294, 0.8862, 0.9333,

					//Lower Tail
					0.7294, 0.8862, 0.9333,
					0.7294, 0.8862, 0.9333,
					0.7294, 0.8862, 0.9333,
					0.7294, 0.8862, 0.9333,
					0.7294, 0.8862, 0.9333,
				]);


	var fading_Flag_Color = new Float32Array([
					//Orange
					0.0, 0.0, 0.0,
					1.0, 0.6, 0.2,

					0.0, 0.0, 0.0,
					1.0, 0.6, 0.2,
					0.0, 0.0, 0.0,
					1.0, 0.6, 0.2,

					0.0, 0.0, 0.0,
					1.0, 0.6, 0.2,
					0.0, 0.0, 0.0,
					1.0, 0.6, 0.2,



					//White
					0.0, 0.0, 0.0,
					1.0, 1.0, 1.0,

					0.0, 0.0, 0.0,
					1.0, 1.0, 1.0,
					0.0, 0.0, 0.0,
					1.0, 1.0, 1.0,

					0.0, 0.0, 0.0,
					1.0, 1.0, 1.0,
					0.0, 0.0, 0.0,
					1.0, 1.0, 1.0,


					//Green
					0.0, 0.0, 0.0,
					0.0705, 0.533, 0.0274,

					0.0, 0.0, 0.0,
					0.0705, 0.533, 0.0274,
					0.0, 0.0, 0.0,
					0.0705, 0.533, 0.0274,

					0.0, 0.0, 0.0,
					0.0705, 0.533, 0.0274,
					0.0, 0.0, 0.0,
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
					gl.DYNAMIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_COLOR,
							4,
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


	/********** V **********/
	vao_V = gl.createVertexArray();
	gl.bindVertexArray(vao_V);

		/********** Position **********/
		vbo_V_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_V_Position);
		gl.bufferData(gl.ARRAY_BUFFER,
					v_Position,		
					gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Color **********/
		vbo_V_Color = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_V_Color);
		gl.bufferData(gl.ARRAY_BUFFER,
					v_Color,
					gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_COLOR,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);


	/********** F **********/
	vao_F = gl.createVertexArray();
	gl.bindVertexArray(vao_F);

		/********** Position **********/
		vbo_F_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_F_Position);
		gl.bufferData(gl.ARRAY_BUFFER,
					f_Position,		
					gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Color **********/
		vbo_F_Color = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_F_Color);
		gl.bufferData(gl.ARRAY_BUFFER,
					f_Color,
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



	/********** Plane's Triangle Part **********/
	vao_Plane_Triangle = gl.createVertexArray();
	gl.bindVertexArray(vao_Plane_Triangle);

		/********** Position **********/
		vbo_Plane_Triangle_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Plane_Triangle_Position);
		gl.bufferData(gl.ARRAY_BUFFER,
					plane_Triangle_Position,		
					gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Color **********/
		vbo_Plane_Triangle_Color = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Plane_Triangle_Color);
		gl.bufferData(gl.ARRAY_BUFFER,
					plane_Triangle_Color,
					gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_COLOR,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);



	/********** Plane's Rectangle Part **********/
	vao_Plane_Rect = gl.createVertexArray();
	gl.bindVertexArray(vao_Plane_Rect);

		/********** Position **********/
		vbo_Plane_Rect_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Plane_Rect_Position);
		gl.bufferData(gl.ARRAY_BUFFER,
					plane_Rect_Position,		
					gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Color **********/
		vbo_Plane_Rect_Color = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Plane_Rect_Color);
		gl.bufferData(gl.ARRAY_BUFFER,
					plane_Rect_Color,
					gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_COLOR,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);




	/********** Plane's Polygon Part **********/
	vao_Plane_Polygon = gl.createVertexArray();
	gl.bindVertexArray(vao_Plane_Polygon);

		/********** Position **********/
		vbo_Plane_Polygon_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Plane_Polygon_Position);
		gl.bufferData(gl.ARRAY_BUFFER,
					plane_Polygon_Position,		
					gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Color **********/
		vbo_Plane_Polygon_Color = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Plane_Polygon_Color);
		gl.bufferData(gl.ARRAY_BUFFER,
					plane_Polygon_Color,
					gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_COLOR,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);



	/********** Fading Flag **********/
	vao_Fading_Flag = gl.createVertexArray();
	gl.bindVertexArray(vao_Fading_Flag);

		/********** Position **********/
		vbo_Fading_Flag_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Fading_Flag_Position);
		gl.bufferData(gl.ARRAY_BUFFER,
					fading_Flag_Position,
					gl.DYNAMIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Color **********/
		vbo_Fading_Flag_Color = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Fading_Flag_Color);
		gl.bufferData(gl.ARRAY_BUFFER,
					fading_Flag_Color,
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

	gl.enable(gl.BLEND);
	gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

	//gl.enable(gl.PROGRAM_POINT_SIZE);
	
	gl.clearDepth(1.0);
	gl.clearColor(0.0, 0.0, 0.0, 1.0);


	perspectiveProjectionMatrix = mat4.create();
	modelViewMatrix = mat4.create();
	modelViewProjectionMatrix = mat4.create();
}


function uninitialize(){

	//Fading Flag
	if (vbo_Fading_Flag_Color) {
		gl.deleteBuffer(vbo_Fading_Flag_Color);
		vbo_Fading_Flag_Color = 0;
	}

	if (vbo_Fading_Flag_Position) {
		gl.deleteBuffer(vbo_Fading_Flag_Position);
		vbo_Fading_Flag_Position = 0;
	}

	if (vao_Fading_Flag) {
		gl.deleteVertexArray(vao_Fading_Flag);
		vao_Fading_Flag = 0;
	}



	//Plane Polygon Part
	if (vbo_Plane_Polygon_Color) {
		gl.deleteBuffer(vbo_Plane_Polygon_Color);
		vbo_Plane_Polygon_Color = 0;
	}

	if (vbo_Plane_Polygon_Position) {
		gl.deleteBuffer(vbo_Plane_Polygon_Position);
		vbo_Plane_Polygon_Position = 0;
	}

	if (vao_Plane_Polygon) {
		gl.deleteVertexArray(vao_Plane_Polygon);
		vao_Plane_Polygon = 0;
	}

	//Plane Rectangle Part
	if (vbo_Plane_Rect_Color) {
		gl.deleteBuffer(vbo_Plane_Rect_Color);
		vbo_Plane_Rect_Color = 0;
	}

	if (vbo_Plane_Rect_Position) {
		gl.deleteBuffer(vbo_Plane_Rect_Position);
		vbo_Plane_Rect_Position = 0;
	}

	if (vao_Plane_Rect) {
		gl.deleteVertexArray(vao_Plane_Rect);
		vao_Plane_Rect = 0;
	}


	//Plane Triangle Part
	if (vbo_Plane_Triangle_Color) {
		gl.deleteBuffer(vbo_Plane_Triangle_Color);
		vbo_Plane_Triangle_Color = 0;
	}

	if (vbo_Plane_Triangle_Position) {
		gl.deleteBuffer(vbo_Plane_Triangle_Position);
		vbo_Plane_Triangle_Position = 0;
	}

	if (vao_Plane_Triangle) {
		gl.deleteVertexArray(vao_Plane_Triangle);
		vao_Plane_Triangle = 0;
	}


	//F
	if (vbo_F_Color) {
		gl.deleteBuffer(vbo_F_Color);
		vbo_F_Color = 0;
	}

	if (vbo_F_Position) {
		gl.deleteBuffer(vbo_F_Position);
		vbo_F_Position = 0;
	}

	if (vao_F) {
		gl.deleteVertexArray(vao_F);
		vao_F = 0;
	}


	//V
	if (vbo_V_Color) {
		gl.deleteBuffer(vbo_V_Color);
		vbo_V_Color = 0;
	}

	if (vbo_V_Position) {
		gl.deleteBuffer(vbo_V_Position);
		vbo_V_Position = 0;
	}

	if (vao_V) {
		gl.deleteVertexArray(vao_V);
		vao_V = 0;
	}


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




//For Plane Translation
var fXTranslation = 0.0;
var fYTranslation = 0.0;

//For Plane
var angle_Plane1 = Math.PI;
var angle_Plane3 = Math.PI;

var XTrans_Plane1 = 0.0;
var YTrans_Plane1 = 0.0;

var XTrans_Plane2 = 0.0;

var XTrans_Plane3 = 0.0;
var YTrans_Plane3 = 0.0;

var ZRot_Plane1 = -60.0;
var ZRot_Plane3 = 60.0;



function draw(){

	var modelViewMatrix = mat4.create();
	var modelViewProjectionMatrix = mat4.create();
	

	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);


	gl.useProgram(shaderProgramObject);

		
		switch (iSequence) {
		case 1:

			My_Letters('I', -7.50 + fXTranslation, 0.0, -8.0);
			fXTranslation = fXTranslation + 0.015;
			if ((-7.5 + fXTranslation) >= -2.0) {
				fXTranslation = 0.0;
				iSequence = 2;
			}
			break;

		case 2:
			My_Letters('I', -2.0, 0.0, -8.0, 20.0);
			My_Letters('V', 8.50 - fXTranslation, 0.0, -8.0);
			fXTranslation = fXTranslation + 0.015;
			if ((8.5 - fXTranslation) <= 2.0) {
				fXTranslation = 0.0;
				iSequence = 3;
			}
			break;

		case 3:
			My_Letters('I', -2.0, 0.0, -8.0);
			My_Letters('V', 2.0, 0.0, -8.0);
			My_Letters('N', -1.35, (6.0 - fYTranslation), -8.0);
			fYTranslation = fYTranslation + 0.015;
			if ((6.0 - fYTranslation) < 0.0) {
				fYTranslation = 0.0;
				iSequence = 4;
			}
			break;

		case 4:
			My_Letters('I', -2.0, 0.0, -8.0);
			My_Letters('V', 2.0, 0.0, -8.0);
			My_Letters('N', -1.35, 0.0, -8.0);
			My_Letters('I', 1.02, (-5.0 + fYTranslation), -8.0);
			fYTranslation = fYTranslation + 0.015;
			if ((-5.0 + fYTranslation) > 0.0) {
				fYTranslation = 0.0;
				iSequence = 5;
			}
			break;

		case 5:


			for(var k = 0; k < 40; k++){
				d_Color[(k * 4) + 3] = fD_Fading;
			}
			
			My_Letters('I', -2.0, 0.0, -8.0);
			My_Letters('N', -1.35, 0.0, -8.0);
			My_D(-0.15, 0.0, -8.0);
			My_Letters('I', 1.02, 0.0, -8.0);
			My_Letters('V', 2.0, 0.0, -8.0);

			if (fD_Fading > 1.0) {
				iSequence = 6;
			}
			else
				fD_Fading = fD_Fading + 0.001;
			break;


		case 6:
			My_Letters('I', -2.0, 0.0, -8.0);
			My_Letters('N', -1.35, 0.0, -8.0);
			My_D(-0.15, 0.0, -8.0);
			My_Letters('I', 1.02, 0.0, -8.0);
			My_Letters('V', 2.0, 0.0, -8.0);






			/********** Plane 1 **********/
			if (bPlane1Reached == NOT_REACH) {
				XTrans_Plane1 = ((3.2 * Math.cos(angle_Plane1)) + (-2.5));
				YTrans_Plane1 = ((4.0 * Math.sin(angle_Plane1)) + (4.0));
				angle_Plane1 = angle_Plane1 + 0.005;
				ZRot_Plane1 = ZRot_Plane1 + 0.2;


				if (angle_Plane1 >= (3.0 * Math.PI) / 2.0) {
					bPlane1Reached = HALF_WAY;
					YTrans_Plane1 = 0.00;

				}
				else if (ZRot_Plane1 >= 0.0)
					ZRot_Plane1 = 0.0;

			}
			else if (bPlane1Reached == HALF_WAY) {
				XTrans_Plane1 = XTrans_Plane1 + 0.010;
				YTrans_Plane1 = 0.00;

				if (XTrans_Plane1 >= 3.00) {	//2.6
					bPlane1Reached = REACH;
					angle_Plane1 = (3.0 * Math.PI) / 2.0;
					ZRot_Plane1 = 0.0;
				}
			}
			else if (bPlane1Reached == REACH) {

				if (Plane1_Count <= 0.0) {
					iFadingFlag1 = 2;
					XTrans_Plane1 = ((3.0 * Math.cos(angle_Plane1)) + (3.0));		//2.6
					YTrans_Plane1 = ((4.0 * Math.sin(angle_Plane1)) + (4.0));

					if (XTrans_Plane1 >= 6.00 || YTrans_Plane1 >= 5.0)
						bPlane1Reached = END;

					angle_Plane1 = angle_Plane1 + 0.005;
					ZRot_Plane1 = ZRot_Plane1 + 0.2;
				}
				else
					iFadingFlag1 = 1;

				Plane1_Count = Plane1_Count - 1.0;
			}
			else if (bPlane1Reached == END) {
				angle_Plane1 = 0.0;
				ZRot_Plane1 = 0.0;
			}

			/*********** Fading Flag ***********/
			if (bPlane1Reached == NOT_REACH)
				My_Fading_Flag(XTrans_Plane1, YTrans_Plane1, -8.0, ZRot_Plane1);

			My_Plane(XTrans_Plane1, YTrans_Plane1, -8.0, 0.18, 0.18, 0.0, ZRot_Plane1);










			/********** Plane 2 **********/
			if (bPlane2Reached == NOT_REACH) {
				if ((-6.0 + XTrans_Plane2) > -2.5) {
					bPlane2Reached = HALF_WAY;
				}
				else
					XTrans_Plane2 = XTrans_Plane2 + 0.011;

			}
			else if (bPlane2Reached == HALF_WAY) {
				XTrans_Plane2 = XTrans_Plane2 + 0.010;
				if ((-6.0 + XTrans_Plane2) >= 3.0) {	//2.6
					bPlane2Reached = REACH;
				}
			}
			else if (bPlane2Reached == REACH) {
				if (Plane2_Count <= 0.00) {
					iFadingFlag2 = 2;
					XTrans_Plane2 = XTrans_Plane2 + 0.010;
				}
				else
					iFadingFlag2 = 1;


				if ((-6.0 + XTrans_Plane2) >= 12.0)
					bPlane2Reached = END;


				Plane2_Count = Plane2_Count - 1.0;
			}
			else if (bPlane2Reached == END) {
				XTrans_Plane2 = 20.0;
			}

			/*********** Fading_Flag **********/
			if (iFadingFlag2 < 2)
				My_Fading_Flag((-6.0 + XTrans_Plane2), 0.0, -8.0, 0.0);

			My_Plane((-6.0 + XTrans_Plane2), 0.0, -8.0, 0.18, 0.18, 0.0, 0.0);







			/********** Plane 3 **********/
			if (bPlane3Reached == NOT_REACH) {
				XTrans_Plane3 = ((3.2 * Math.cos(angle_Plane3)) + (-2.5));
				YTrans_Plane3 = ((4.0 * Math.sin(angle_Plane3)) + (-4.0));
				angle_Plane3 = angle_Plane3 - 0.005;
				ZRot_Plane3 = ZRot_Plane3 - 0.2;


				if (angle_Plane3 < (Math.PI) / 2.0) {
					bPlane3Reached = HALF_WAY;
					YTrans_Plane3 = 0.00;

				}
				else if (ZRot_Plane3 < 0.0)
					ZRot_Plane3 = 0.0;

			}
			else if (bPlane3Reached == HALF_WAY) {
				XTrans_Plane3 = XTrans_Plane3 + 0.010;
				YTrans_Plane3 = 0.00;

				if (XTrans_Plane3 >= 3.00) {	//2.6
					bPlane3Reached = REACH;
					angle_Plane3 = Math.PI / 2.0;
					ZRot_Plane3 = 0.0;
				}
			}
			else if (bPlane3Reached == REACH) {

				if (Plane3_Count <= 0.0) {
					iFadingFlag3 = 2;
					XTrans_Plane3 = ((3.0 * Math.cos(angle_Plane3)) + (3.0));		//2.6
					YTrans_Plane3 = ((4.0 * Math.sin(angle_Plane3)) + (-4.0));

					if (XTrans_Plane3 >= 6.00 || YTrans_Plane3 < -5.0)
						bPlane3Reached = END;

					angle_Plane3 = angle_Plane3 - 0.005;
					ZRot_Plane3 = ZRot_Plane3 - 0.2;
				}
				else
					iFadingFlag3 = 1;

				Plane3_Count = Plane3_Count - 1.0;
			}
			else if (bPlane3Reached == END) {
				angle_Plane3 = 0.0;
				ZRot_Plane3 = 0.0;
			}



			/*********** Fading Flag ***********/
			if (bPlane2Reached == NOT_REACH)
				My_Fading_Flag(XTrans_Plane3, YTrans_Plane3, -8.0, ZRot_Plane3);


			My_Plane(XTrans_Plane3, YTrans_Plane3, -8.0, 0.18, 0.18, 0.0, ZRot_Plane3);


			if (iFadingFlag1 == 2 || iFadingFlag2 == 2 || iFadingFlag3 == 2)
				My_Flag(2.0, 0.0, -8.0);


			break;

		}


	gl.useProgram(null);


	


	requestAnimationFrame(draw, canvas);
}



function degToRad(angle){
	return(angle * (Math.PI / 180.0));
}


function My_Letters(c, x, y, z) {

	mat4.identity(modelViewMatrix);
	mat4.identity(modelViewProjectionMatrix);

	mat4.translate(modelViewMatrix, modelViewMatrix, [x, y, z]);
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

	
	switch (c) {
	case 'I':
		gl.bindVertexArray(vao_I);
		gl.drawArrays(gl.LINES, 0, 30);
		gl.bindVertexArray(null);
		break;

	case 'N':
		gl.bindVertexArray(vao_N);
		gl.drawArrays(gl.LINES, 0, 30);
		gl.bindVertexArray(null);
		break;

	case 'V':
		gl.bindVertexArray(vao_V);
		gl.drawArrays(gl.LINES, 0, 20);
		gl.bindVertexArray(null);
		break;



	}


}

function My_D(x, y, z) {

	mat4.identity(modelViewMatrix);
	mat4.identity(modelViewProjectionMatrix);

	mat4.translate(modelViewMatrix, modelViewMatrix, [x, y, z]);
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);


	gl.bindVertexArray(vao_D);

		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_D_Color);
		gl.bufferData(gl.ARRAY_BUFFER,
			d_Color,
			gl.DYNAMIC_DRAW);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

		gl.drawArrays(gl.LINES, 0, 40);
	gl.bindVertexArray(null);

}

function My_Fading_Flag(x, y, z, fAngle) {


	if (bPlane2Reached != REACH) {
		for(var p = 0; p < 30; p++){
			fading_Flag_Position[(p * 6) ] -= 0.005;
		}

	}
	else if (bPlane2Reached == REACH) {

		for(var p = 0; p < 30; p++){
			fading_Flag_Position[(p * 6) ] += 0.007;
		}
	}

	mat4.identity(modelViewMatrix);
	mat4.identity(modelViewProjectionMatrix);

	mat4.translate(modelViewMatrix, modelViewMatrix, [x, y, z]);
	mat4.rotateZ(modelViewMatrix, modelViewMatrix, degToRad(fAngle));
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);



	gl.bindVertexArray(vao_Fading_Flag);

	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Fading_Flag_Position);
	gl.bufferData(gl.ARRAY_BUFFER,
		fading_Flag_Position,
		gl.DYNAMIC_DRAW);

	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.drawArrays(gl.LINES, 0, 30);

	gl.bindVertexArray(null);

}



function My_Flag(x, y, z) {

	mat4.identity(modelViewMatrix);
	mat4.identity(modelViewProjectionMatrix);

	mat4.translate(modelViewMatrix, modelViewMatrix, [x, y, z]);
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);	

	gl.bindVertexArray(vao_Flag);
	gl.drawArrays(gl.LINES, 0, 30);
	gl.bindVertexArray(null);
}



function My_Plane(x, y, z, scaleX, scaleY, scaleZ, ZRot_Angle) {

	
	mat4.identity(modelViewMatrix);
	mat4.identity(modelViewProjectionMatrix);

	mat4.translate(modelViewMatrix, modelViewMatrix, [x, y, z]);
	mat4.scale(modelViewMatrix, modelViewMatrix, [scaleX, scaleY, scaleY]);
	mat4.rotateZ(modelViewMatrix, modelViewMatrix, degToRad(ZRot_Angle));
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);	




	//Triangle
	gl.bindVertexArray(vao_Plane_Triangle);
		gl.drawArrays(gl.TRIANGLES, 0, 3);
	gl.bindVertexArray(null);

	//Rectangle
	gl.bindVertexArray(vao_Plane_Rect);

		//For Middle
		gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);

		//For Upper and Lower Fin
		gl.drawArrays(gl.TRIANGLE_FAN, 4, 8);

		//For Back
		gl.drawArrays(gl.TRIANGLE_FAN, 12, 4);

	gl.bindVertexArray(null);


	//Polygon
	gl.bindVertexArray(vao_Plane_Polygon);
		gl.drawArrays(gl.TRIANGLE_FAN, 0, 10);
	gl.bindVertexArray(null);



	//I

	mat4.translate(modelViewMatrix, modelViewMatrix, [-1.5, 0.0, 0.0]);
	mat4.scale(modelViewMatrix, modelViewMatrix, [0.70, 0.70, 0.0])
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);
	

	gl.bindVertexArray(vao_I);
	gl.drawArrays(gl.LINES, 0, 30);
	gl.bindVertexArray(null);






	//A

	mat4.translate(modelViewMatrix, modelViewMatrix, [1.0, 0.0, 0.0]);
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);
	

	gl.bindVertexArray(vao_A);
	gl.drawArrays(gl.LINES, 0, 6);
	gl.bindVertexArray(null);

	




	//F
	mat4.translate(modelViewMatrix, modelViewMatrix, [0.70, 0.0, 0.0]);
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);
	

	gl.bindVertexArray(vao_F);
	gl.drawArrays(gl.LINES, 0, 6);
	gl.bindVertexArray(null);



}




