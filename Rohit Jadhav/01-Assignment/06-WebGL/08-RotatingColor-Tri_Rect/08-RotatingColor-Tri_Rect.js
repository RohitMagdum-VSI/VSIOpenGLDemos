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



var requestAnimationFrame = 
	window.requestAnimationFrame || 
	window.webkitRequestAnimationFrame || 
	window.mozRequestAnimationFrame || 
	window.msRequestAnimationFrame ||
	window.oRequestAnimationFrame ||
	null;


var cancelAnimationFrame = 
	window.cancelAnimationFrame || 
	window.webkitCancelRequestAnimationFrame || window.webkitCancelAnimationFrame || 
	window.mozCancelRequestAnimationFrame || window.mozCancelAnimationFrame || 
	window.msCancelRequestAnimationFrame || window.msCancelAnimationFrame ||
	window.oCancelRequestAnimationFrame || window.oCancelAnimationFrame ||
	null;



//For Shader
var vertexShaderObject;
var fragmentShaderObject;
var shaderProgramObject;

//For Uniform
var mvpUniform;

//For Projection
var perspectiveProjectionMatrix;

//For Triangle
var vao_Tri;
var vbo_Tri_Position;
var vbo_Tri_Color;
var angle_Tri = 0.0;

//For Rect
var vao_Rect;
var vbo_Rect_Position;
var angle_Rect = 360.0;


function main(){

	canvas = document.getElementById("08-RotatingColor-Tri_Rect-RRJ");
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
		null;


	if(fullscreen_element == null){

		if(canvas.requestFullscreen)
			canvas.requestFullscreen();
		else if(canvas.webkitRequestFullscreen)
			canvas.wubkitRequestFullscreen();
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
		"void main() {" +
			"gl_Position = u_mvp_matrix * vPosition;" +
			"outColor = vColor;" +
		"}";

	gl.shaderSource(vertexShaderObject, szVertexShaderSourceCode);

	gl.compileShader(vertexShaderObject);

	var  shaderCompileStatus = gl.getShaderParameter(vertexShaderObject, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(vertexShaderObject);
		if(error.length > 0){
			alert("Vertex Shader Compilation Error: " + error);
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
		"void main(void) {" +
			"FragColor = outColor;" +
		"}";

	gl.shaderSource(fragmentShaderObject, szFragmentShaderSourceCode);
	gl.compileShader(fragmentShaderObject);

	shaderCompileStatus = gl.getShaderParameter(fragmentShaderObject, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(fragmentShaderObject);
		if(error.length > 0){
			alert("Fragment Shader Compilation Error: "+ error);
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
			alert("Program Linkink Error: " + error);
			uninitialize();
			window.close();
		}
	}


	mvpUniform = gl.getUniformLocation(shaderProgramObject, "u_mvp_matrix");


	var tri_Position = new Float32Array([
						0.0, 1.0, 0.0,
						-1.0, -1.0, 0.0,
						1.0, -1.0, 0.0
					]);

	var tri_Color = new Float32Array([
						1.0, 0.0, 0.0,
						0.0, 1.0, 0.0,
						0.0, 0.0, 1.0
					]);

	var rect_Position = new Float32Array([
						1.0, 1.0, 0.0,
						-1.0, 1.0, 0.0,
						-1.0, -1.0, 0.0,
						1.0, -1.0, 0.0
					]);




	/********** Triangle *********/
	vao_Tri = gl.createVertexArray();
	gl.bindVertexArray(vao_Tri);

		/********** Position **********/
		vbo_Tri_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Tri_Position);
		gl.bufferData(gl.ARRAY_BUFFER, tri_Position, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION, 
							3,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Color *********/
		vbo_Tri_Color = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Tri_Color);
		gl.bufferData(gl.ARRAY_BUFFER, tri_Color, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_COLOR,
							3,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);


	/********* Rectangle *********/
	vao_Rect = gl.createVertexArray();
	gl.bindVertexArray(vao_Rect);

		/********* Position **********/
		vbo_Rect_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Rect_Position);
		gl.bufferData(gl.ARRAY_BUFFER,  rect_Position, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

		/********** Color *********/
		gl.vertexAttrib3f(WebGLMacros.AMC_ATTRIBUTE_COLOR, 0.0, 0.0, 1.0);

	gl.bindVertexArray(null);



	gl.enable(gl.DEPTH_TEST);
	gl.depthFunc(gl.LEQUAL);

	gl.disable(gl.CULL_FACE);

	gl.clearColor(0.0, 0.0, 0.0, 1.0);


	perspectiveProjectionMatrix = mat4.create();

}

function uninitialize(){


	if(vbo_Rect_Position){
		gl.deleteBuffer(vbo_Rect_Position);
		vbo_Rect_Position = 0;
	}

	if(vao_Rect){
		gl.deleteVertexArray(vao_Rect);
		vao_Rect = 0;
	}


	if(vbo_Tri_Color){
		gl.deleteBuffer(vbo_Tri_Color);
		vbo_Tri_Color = 0;
	}

	if(vbo_Tri_Position){
		gl.deleteBuffer(vbo_Tri_Position);
		vbo_Tri_Position = 0;
	}

	if(vao_Tri){
		gl.deleteVertexArray(vao_Tri);
		vao_Tri = 0;
	}

	if(shaderProgramObject){

		gl.useProgram(shaderProgramObject);

			if(fragmentShaderObject){
				gl.detachShader(shaderProgramObject, fragmentShaderObject);
				gl.deleteShader(fragmentShaderObject);
				fragmentShaderObject = 0;
			}

			if(vertexShaderObject){
				gl.detachShader(shaderProgramObject, vertexShaderObject);
				gl.deleteShader(vertexShaderObject);
				vertexShaderObject = 0;
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

	mat4.perspective(perspectiveProjectionMatrix, 
					45.0,
					parseFloat(canvas.width) / parseFloat(canvas.height),
					0.1,
					100.0);
}

function draw(){

	var modelViewMatrix = mat4.create();
	var modelViewProjectionMatrix  = mat4.create();

	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);


	gl.useProgram(shaderProgramObject);

		/*********** Triangle **********/
		mat4.translate(modelViewMatrix, modelViewMatrix, [-1.50, 0.0, -4.0]);
		mat4.rotateY(modelViewMatrix, modelViewMatrix, degToRad(angle_Tri));
		mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

		gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

		gl.bindVertexArray(vao_Tri);
			gl.drawArrays(gl.TRIANGLES, 0, 3);
		gl.bindVertexArray(null);


		/********** Rectangle **********/
		mat4.identity(modelViewMatrix);
		mat4.identity(modelViewProjectionMatrix);
		mat4.translate(modelViewMatrix, modelViewMatrix, [1.50, 0.0, -4.0]);
		mat4.rotateX(modelViewMatrix, modelViewMatrix, degToRad(angle_Rect));
		mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

		gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

		gl.bindVertexArray(vao_Rect);
			gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
		gl.bindVertexArray(null);

			
	gl.useProgram(null);

	update();

	requestAnimationFrame(draw, canvas);

}

function degToRad(angle){
	return(angle * (Math.PI / 180.0));
}

function update(){

	angle_Tri = angle_Tri + 0.5;
	angle_Rect = angle_Rect - 0.5;

	if(angle_Tri > 360.0)
		angle_Tri = 0.0;

	if(angle_Rect < 0.0)
		angle_Rect = 360.0;
}