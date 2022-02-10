var canvas = null;
var gl = null;
var bIsFullScreen = false;
var canvas_original_width = 0;
var canvas_original_height = 0;



const WebGLMacros = {
	AMC_ATTRIBUTE_POSITION:0,
	AMC_ATTRIBUTE_COLOR:1,
	AMC_ATTRIBUTE_NORMAL:2,
	AMC_ATTRIBUTE_TEXCOORD0:3,
};


//For Starting Animation we need requestAnimationFrame()

var requestAnimationFrame = 
	window.requestAnimationFrame || 
	window.webkitRequestAnimationFrame ||
	window.mozRequestAnimationFrame || 
	window.oRequestAnimationFrame || 
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


//For Shader
var vertexShaderObject;
var fragmentShaderObject;
var shaderProgramObject;

//For Uniform
var modelMatrix_Uniform;
var viewMatrix_Uniform;
var projectionMatrix_Uniform;


//For Projection Matrix;
var perspectiveProjectionMatrix;


//For Cube
var vao_Cube;
var vbo_Cube_Position;
var vbo_Cube_Normal;
var angle_Cube = 360.0;


//For Light Uniform
var kd_Uniform;
var ld_Uniform;
var lightPosition_Uniform;
var LKeyPress_Uniform;
var bLights = false;

function main(){

	canvas = document.getElementById("19-DiffuseLightOnCube-RRJ");
	if(!canvas){
		console.log("Obtaining Canvas Failed!!\n");
		return;
	}
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
		document.webkitFullscreenElement ||
		document.mozFullScreenElement ||
		document.msFullscreenElement || 
		document.oFullscreenElement ||
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
		else if(document.webkitExitFullscreen)
			document.webkitExitFullscreen();
		else if(document.mozCancelFullScreen)
			document.mozCancelFullScreen();
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

		//L 
		case 76:
			if(bLights == false)
				bLights = true;
			else
				bLights = false;
			break;


		//F
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
	var vertexShaderSourceCode = 
		"#version 300 es" +
		"\n" +
		"in vec4 vPosition;" +
		"in vec3 vNormal;" +


		"uniform vec3 u_ld;" +
		"uniform vec3 u_kd;" +
		"uniform vec4 u_light_position;" +
		"uniform int u_LKey;" + 

		"uniform mat4 u_model_matrix;" +
		"uniform mat4 u_view_matrix;" +
		"uniform mat4 u_projection_matrix;" +

		"out vec3 outDiffuseLight;" +

		"void main() {" +

			"if(u_LKey == 1) {" +
				"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" +
				"vec3  source = normalize(vec3(u_light_position - eyeCoordinate));" +
				"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" +
				"vec3 normal = normalize(vec3(normalMatrix * vNormal));" +
				"float S_Dot_N = max(dot(source, normal), 0.0);" +
				"outDiffuseLight = u_ld * u_kd * S_Dot_N;" + 
			"}"+ 
			"else{ " +
				"outDiffuseLight = vec3(1.0, 1.0, 1.0);" +
			"}" + 

			"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" +
			
		"}";

	gl.shaderSource(vertexShaderObject, vertexShaderSourceCode);
	gl.compileShader(vertexShaderObject);

	var shaderCompileStatus = gl.getShaderParameter(vertexShaderObject, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(vertexShaderObject);
		if(error.length > 0){
			alert("Vertex Shader Compilation Error: " + error);
			uninitialize();
			window.close();
		}
	}


	fragmentShaderObject = gl.createShader(gl.FRAGMENT_SHADER);
	var fragmentShaderSourceCode = 
		"#version 300 es" +
		"\n" +
		"precision highp float;" +
		"in vec3 outDiffuseLight;" +
		"out vec4 FragColor;" +
		"void main(){" +
			"FragColor = vec4(outDiffuseLight, 1.0);" +
		"}";

	gl.shaderSource(fragmentShaderObject, fragmentShaderSourceCode);
	gl.compileShader(fragmentShaderObject);

	shaderCompileStatus = gl.getShaderParameter(fragmentShaderObject, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(fragmentShaderObject);
		if(error.length > 0){
			alert("Fragment Shader Compilation Error: " + error);
			uninitialize();
			window.close();
		}
	}


	shaderProgramObject = gl.createProgram();

	gl.attachShader(shaderProgramObject, vertexShaderObject);
	gl.attachShader(shaderProgramObject, fragmentShaderObject);

	gl.bindAttribLocation(shaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
	gl.bindAttribLocation(shaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_NORMAL, "vNormal");


	gl.linkProgram(shaderProgramObject);

	
	if(!gl.getProgramParameter(shaderProgramObject, gl.LINK_STATUS)){
		var error = gl.getProgramInfoLog(shaderProgramObject);
		if(error.length > 0){
			alert("Shader Program Linking Error: " + error);
			uninitialize();
			window.close();
		}
	}


	modelMatrix_Uniform = gl.getUniformLocation(shaderProgramObject, "u_model_matrix");
	viewMatrix_Uniform = gl.getUniformLocation(shaderProgramObject, "u_view_matrix");
	projectionMatrix_Uniform = gl.getUniformLocation(shaderProgramObject, "u_projection_matrix");

	ld_Uniform = gl.getUniformLocation(shaderProgramObject, "u_ld");
	kd_Uniform = gl.getUniformLocation(shaderProgramObject, "u_kd");
	lightPosition_Uniform = gl.getUniformLocation(shaderProgramObject, "u_light_position");
	LKeyPress_Uniform = gl.getUniformLocation(shaderProgramObject, "u_LKey");



	/********** Position and Normal **********/

	var cube_Position = new Float32Array([

						1.0, 1.0, -1.0,
						-1.0, 1.0, -1.0,
						-1.0, 1.0, 1.0,
						1.0, 1.0, 1.0,
						//Bottom
						1.0, -1.0, -1.0,
						-1.0, -1.0, -1.0,
						-1.0, -1.0, 1.0,
						1.0, -1.0, 1.0,
						//Front
						1.0, 1.0, 1.0,
						-1.0, 1.0, 1.0,
						-1.0, -1.0, 1.0,
						1.0, -1.0, 1.0,
						//Back
						1.0, 1.0, -1.0,
						-1.0, 1.0, -1.0,
						-1.0, -1.0, -1.0,
						1.0, -1.0, -1.0,
						//Right
						1.0, 1.0, -1.0,
						1.0, 1.0, 1.0,
						1.0, -1.0, 1.0,
						1.0, -1.0, -1.0,
						//Left
						-1.0, 1.0, 1.0, 
						-1.0, 1.0, -1.0, 
						-1.0, -1.0, -1.0, 
						-1.0, -1.0, 1.0,
					]);

	
	var cube_Normal = new Float32Array([
						//Top
						0.0, 1.0, 0.0,
						0.0, 1.0, 0.0,
						0.0, 1.0, 0.0,
						0.0, 1.0, 0.0,
						
						//Bottom
						0.0, -1.0, 0.0,
						0.0, -1.0, 0.0,
						0.0, -1.0, 0.0,
						0.0, -1.0, 0.0,
						
						//Front
						0.0, 0.0, 1.0,
						0.0, 0.0, 1.0,
						0.0, 0.0, 1.0,
						0.0, 0.0, 1.0,
						
						//Back
						0.0, 0.0, -1.0,
						0.0, 0.0, -1.0,
						0.0, 0.0, -1.0,
						0.0, 0.0, -1.0,
						
						//Right
						1.0, 0.0, 0.0,
						1.0, 0.0, 0.0,
						1.0, 0.0, 0.0,
						1.0, 0.0, 0.0,
						
						//Left
						-1.0, 0.0, 0.0,
						-1.0, 0.0, 0.0,
						-1.0, 0.0, 0.0,
						-1.0, 0.0, 0.0,
					]);




	/********** Cube **********/
	vao_Cube = gl.createVertexArray();
	gl.bindVertexArray(vao_Cube);

		/********** Position **********/
		vbo_Cube_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Cube_Position);
		gl.bufferData(gl.ARRAY_BUFFER, cube_Position, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Normal ***********/
		vbo_Cube_Normal = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Cube_Normal);
		gl.bufferData(gl.ARRAY_BUFFER, cube_Normal, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_NORMAL, 
							3,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_NORMAL);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);


	gl.enable(gl.DEPTH_TEST);
	gl.depthFunc(gl.LEQUAL);

	gl.disable(gl.CULL_FACE);
	
	gl.clearDepth(1.0);

	gl.clearColor(0.0, 0.0, 0.0, 1.0);
	

	perspectiveProjectionMatrix = mat4.create();
}



function uninitialize(){


	if(vbo_Cube_Normal){
		gl.deleteBuffer(vbo_Cube_Normal);
		vbo_Cube_Normal = 0;
	}

	if(vbo_Cube_Position){
		gl.deleteBuffer(vbo_Cube_Position);
		vbo_Cube_Position = 0;
	}

	if(vao_Cube){
		gl.deleteVertexArray(vao_Cube);
		vao_Cube = 0;
	}


	if(shaderProgramObject){

		gl.useProgram(shaderProgramObject);

			if(fragmentShaderObject){
				gl.detachShader(shaderProgramObject, fragmentShaderObject);
				gl.deleteShader(fragmentShaderObject);
				fragmentShaderObject = 0;
				console.log("Fragment Shader Detach and Deleted!!\n");
			}

			if(vertexShaderObject){
				gl.detachShader(shaderProgramObject, vertexShaderObject);
				gl.deleteShader(vertexShaderObject);
				vertexShaderObject = 0;
				console.log("Vertex Shader Detach and Deleted!!\n");
			}

		gl.useProgram(null);
		gl.deleteProgram(shaderProgramObject);
		shaderProgramObject = 0;
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

	 mat4.identity(perspectiveProjectionMatrix);
	 mat4.perspective(perspectiveProjectionMatrix, 
	 				45.0,
	 				parseFloat(canvas.width) / parseFloat(canvas.height),
	 				0.1,
	 				100.0);
}

function draw(){

	var modelMatrix = mat4.create();
	var viewMatrix = mat4.create();

	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);


	gl.useProgram(shaderProgramObject);



		
		/********** Cube ***********/
		mat4.identity(modelMatrix);
		mat4.identity(viewMatrix);
		mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -5.0]);
		mat4.rotateX(modelMatrix, modelMatrix, degToRad(angle_Cube));
		mat4.rotateY(modelMatrix, modelMatrix, degToRad(angle_Cube));
		mat4.rotateZ(modelMatrix, modelMatrix, degToRad(angle_Cube));
		
		gl.uniformMatrix4fv(modelMatrix_Uniform, false, modelMatrix);
		gl.uniformMatrix4fv(viewMatrix_Uniform, false, viewMatrix);
		gl.uniformMatrix4fv(projectionMatrix_Uniform, false, perspectiveProjectionMatrix)
		

		if(bLights == true){
			gl.uniform1i(LKeyPress_Uniform, 1);
			gl.uniform3f(ld_Uniform, 1.0, 1.0, 1.0);
			gl.uniform3f(kd_Uniform, 0.5, 0.5, 0.5);
			gl.uniform4f(lightPosition_Uniform, 0.0, 0.0, 5.0, 1.0);
		}
		else
			gl.uniform1i(LKeyPress_Uniform, 0);

		gl.bindVertexArray(vao_Cube);

			gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
			gl.drawArrays(gl.TRIANGLE_FAN, 4, 4);
			gl.drawArrays(gl.TRIANGLE_FAN, 8, 4);
			gl.drawArrays(gl.TRIANGLE_FAN, 12, 4);
			gl.drawArrays(gl.TRIANGLE_FAN, 16, 4);
			gl.drawArrays(gl.TRIANGLE_FAN, 20, 4);

		gl.bindVertexArray(null);


	gl.useProgram(null);

	update();

	requestAnimationFrame(draw, canvas);
}

function update(){

	angle_Cube = angle_Cube - 0.7;

	if(angle_Cube < 0.0)
		angle_Cube = 360.0;
}

function degToRad(angle){
	return(angle * (Math.PI / 180.0));
}
