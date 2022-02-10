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
var mvpUniform;
var samplerUniform;

//For Projection
var perspectiveProjectionMatrix;

//For Pyramid
var vao_Pyramid;
var vbo_Pyramid_Position;
var vbo_Pyramid_Texcoord;
var angle_Pyramid = 0.0;
var texture_Stone;

//For Cube
var vao_Cube;
var vbo_Cube_Position;
var vbo_Cube_Texcoord;
var angle_Cube = 360.0;
var texture_Kundali;


function main(){

	canvas = document.getElementById("18-Texture_Pyramid_Cube-RRJ");
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
		"in vec2 vTex;" +
		"uniform mat4 u_mvp_matrix;" +
		"out vec2 outTex;" +
		"void main() {" +
			"gl_Position = u_mvp_matrix * vPosition;" +
			"outTex = vTex;" +
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
		"precision highp sampler2D;" +
		"in vec2 outTex;" +
		"uniform sampler2D u_sampler;" +
		"out vec4 FragColor;" +
		"void main(){" +
			"FragColor = texture(u_sampler, outTex);" +
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
	gl.bindAttribLocation(shaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0, "vTex");


	gl.linkProgram(shaderProgramObject);

	
	if(!gl.getProgramParameter(shaderProgramObject, gl.LINK_STATUS)){
		var error = gl.getProgramInfoLog(shaderProgramObject);
		if(error.length > 0){
			alert("Shader Program Linking Error: " + error);
			uninitialize();
			window.close();
		}
	}


	mvpUniform = gl.getUniformLocation(shaderProgramObject, "u_mvp_matrix");
	samplerUniform = gl.getUniformLocation(shaderProgramObject, "u_sampler");




	/*********** Pyramid **********/
	texture_Stone = gl.createTexture();
	texture_Stone.image = new Image();
	texture_Stone.image.src = "stone.png";
	texture_Stone.image.onload = function(){
		gl.bindTexture(gl.TEXTURE_2D, texture_Stone);
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, texture_Stone.image);
		gl.generateMipmap(gl.TEXTURE_2D);
		gl.bindTexture(gl.TEXTURE_2D, null);
	};




	/********** Cube **********/
	texture_Kundali = gl.createTexture();
	texture_Kundali.image = new Image();
	texture_Kundali.image.src = "kundali.png";
	texture_Kundali.image.onload = function(){
		gl.bindTexture(gl.TEXTURE_2D, texture_Kundali);
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, texture_Kundali.image);
		gl.bindTexture(gl.TEXTURE_2D, null);
	}






	var pyramid_Position = new Float32Array([
						//Face
						0.0, 1.0, 0.0,
						-1.0, -1.0, 1.0,
						1.0, -1.0, 1.0,
						//Right
						0.0, 1.0, 0.0,
						1.0, -1.0, 1.0,
						1.0, -1.0, -1.0,
						//Back
						0.0, 1.0, 0.0,
						1.0, -1.0, -1.0,
						-1.0, -1.0, -1.0,
						//Left
						0.0, 1.0, 0.0,
						-1.0, -1.0, -1.0,
						-1.0, -1.0, 1.0,
					]);


	var pyramid_Texcoord = new Float32Array([
						//Face
						0.5, 1.0,
						0.0, 0.0,
						1.0, 0.0,
						//Right
						0.5, 1.0,
						1.0, 0.0,
						0.0, 0.0,
						//Back
						0.5, 1.0,
						0.0, 0.0,
						1.0, 0.0,
						//Left
						0.5, 1.0,
						1.0, 0.0,
						0.0, 0.0,
					]);
	

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

	
	var cube_Texcoord = new Float32Array([
						//Top
						1.0, 1.0,
						0.0, 1.0,
						0.0, 0.0,
						1.0, 0.0,
						//Back
						1.0, 1.0,
						0.0, 1.0,
						0.0, 0.0,
						1.0, 0.0,
						//Face
						1.0, 1.0,
						0.0, 1.0,
						0.0, 0.0,
						1.0, 0.0,
						//Back
						1.0, 1.0, 
						0.0, 1.0,
						0.0, 0.0,
						1.0, 0.0,
						//Right
						1.0, 1.0,
						0.0, 1.0,
						0.0, 0.0,
						1.0, 0.0,
						//Left
						1.0, 1.0,
						0.0, 1.0, 
						0.0, 0.0,
						1.0, 0.0,
					]);


	/********** Pyramid ***********/
	vao_Pyramid = gl.createVertexArray();
	gl.bindVertexArray(vao_Pyramid);

		/********** Position **********/
		vbo_Pyramid_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Pyramid_Position);
		gl.bufferData(gl.ARRAY_BUFFER, pyramid_Position, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);



		/********** Texcoord **********/
		vbo_Pyramid_Texcoord = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Pyramid_Texcoord);
		gl.bufferData(gl.ARRAY_BUFFER, pyramid_Texcoord, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0, 
							2,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);



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


		/********** Texcoord ***********/
		vbo_Cube_Texcoord = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Cube_Texcoord);
		gl.bufferData(gl.ARRAY_BUFFER, cube_Texcoord, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0, 
							2,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0);
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

	if(texture_Kundali){
		gl.deleteTexture(texture_Kundali);
		texture_Kundali = 0;
	}

	if(vbo_Cube_Texcoord){
		gl.deleteBuffer(vbo_Cube_Texcoord);
		vbo_Cube_Texcoord = 0;
	}

	if(vbo_Cube_Position){
		gl.deleteBuffer(vbo_Cube_Position);
		vbo_Cube_Position = 0;
	}

	if(vao_Cube){
		gl.deleteVertexArray(vao_Cube);
		vao_Cube = 0;
	}


	if(texture_Stone){
		gl.deleteTexture(texture_Stone);
		texture_Stone = 0;
	}

	if(vbo_Pyramid_Texcoord){
		gl.deleteBuffer(vbo_Pyramid_Texcoord);
		vbo_Pyramid_Texcoord = 0;
	}

	if(vbo_Pyramid_Position){
		gl.deleteBuffer(vbo_Pyramid_Position);
		vbo_Pyramid_Position = 0;
	}

	if(vao_Pyramid){
		gl.deleteVertexArray(vao_Pyramid);
		vao_Pyramid = 0;
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

	var modelViewMatrix = mat4.create();
	var modelViewProjectionMatrix = mat4.create();


	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);


	gl.useProgram(shaderProgramObject);


		/*********** Pyramid *********/
		mat4.identity(modelViewMatrix);
		mat4.identity(modelViewProjectionMatrix);
		mat4.translate(modelViewMatrix, modelViewMatrix, [-1.5, 0.0, -5.0]);
		mat4.rotateY(modelViewMatrix, modelViewMatrix, degToRad(angle_Pyramid));
		mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

		gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

		gl.bindTexture(gl.TEXTURE_2D, texture_Stone);
		gl.uniform1i(samplerUniform, 0);

		gl.bindVertexArray(vao_Pyramid);
			gl.drawArrays(gl.TRIANGLES, 0, 12);
		gl.bindVertexArray(null);

		
		/********** Cube ***********/
		mat4.identity(modelViewMatrix);
		mat4.identity(modelViewProjectionMatrix);
		mat4.translate(modelViewMatrix, modelViewMatrix, [1.5, 0.0, -5.0]);
		mat4.scale(modelViewMatrix, modelViewMatrix, [0.9, 0.9, 0.9]);
		mat4.rotateX(modelViewMatrix, modelViewMatrix, degToRad(angle_Cube));
		mat4.rotateY(modelViewMatrix, modelViewMatrix, degToRad(angle_Cube));
		mat4.rotateZ(modelViewMatrix, modelViewMatrix, degToRad(angle_Cube));
		mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);
		
		gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

		gl.bindTexture(gl.TEXTURE_2D, texture_Kundali);
		gl.uniform1i(samplerUniform, 0);

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

	angle_Pyramid = angle_Pyramid + 0.7;
	angle_Cube = angle_Cube - 0.7;

	if(angle_Pyramid > 360.0)
		angle_Pyramid = 0.0;

	if(angle_Cube < 0.0)
		angle_Cube = 360.0;
}

function degToRad(angle){
	return(angle * (Math.PI / 180.0));
}
