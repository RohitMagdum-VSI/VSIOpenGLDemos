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
var samplerUniform;

//For Projection
var perspectiveProjectionMatrix;


//For Rect
var vao_Rect;
var vbo_Rect_Position;
var vbo_Rect_Texcoord;
var texture_Checkerboard;
var rect_Position = new Float32Array(4 * 3);

//For Checker board
const CHECKIMAGE_WIDTH = 64;
const CHECKIMAGE_HEIGHT = 64;
var checkImageData = new Uint8Array(CHECKIMAGE_WIDTH * CHECKIMAGE_HEIGHT * 4);



function main(){

	canvas = document.getElementById("17-CheckerBoard-RRJ");
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
		"in vec2 vTex;" + 
		"out vec2 outTex;" +
		"uniform mat4 u_mvp_matrix;" +
		"void main() {" +
			"outTex = vTex;" +
			"gl_Position = u_mvp_matrix * vPosition;" +
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
		"precision highp sampler2D;" +
		"in vec2 outTex;" +
		"out vec4 FragColor;" +
		"uniform sampler2D u_sampler;" +
		"void main(void) {" +
			"FragColor = texture(u_sampler, outTex);"  +
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
	gl.bindAttribLocation(shaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0, "vTex");

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



	//Load Texture
	makeCheckImage();

	texture_Checkerboard = gl.createTexture();

		gl.bindTexture(gl.TEXTURE_2D, texture_Checkerboard);
		gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
		gl.texImage2D(gl.TEXTURE_2D, 0, 
					gl.RGBA, 
					CHECKIMAGE_WIDTH, CHECKIMAGE_HEIGHT, 0,
					gl.RGBA, 
					gl.UNSIGNED_BYTE, checkImageData);
		gl.generateMipmap(gl.TEXTURE_2D);
		gl.bindTexture(gl.TEXTURE_2D, null);
	




	mvpUniform = gl.getUniformLocation(shaderProgramObject, "u_mvp_matrix");
	samplerUniform = gl.getUniformLocation(shaderProgramObject, "u_sampler");


	

	var rect_Texcoord = new Float32Array([
						1.0, 1.0,
						0.0, 1.0,
						0.0, 0.0,
						1.0, 0.0,
					]);



	/********* Rectangle *********/
	vao_Rect = gl.createVertexArray();
	gl.bindVertexArray(vao_Rect);

		/********* Position **********/
		vbo_Rect_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Rect_Position);
		gl.bufferData(gl.ARRAY_BUFFER,  rect_Position, gl.DYNAMIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Texture **********/
		vbo_Rect_Texcoord = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Rect_Texcoord);
		gl.bufferData(gl.ARRAY_BUFFER, rect_Texcoord, gl.STATIC_DRAW);
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



	gl.clearColor(0.0, 0.0, 0.0, 1.0);


	perspectiveProjectionMatrix = mat4.create();

}

function makeCheckImage(){

	var c;

	for(var i = 0; i < CHECKIMAGE_HEIGHT; i++){
		for(var j = 0; j < CHECKIMAGE_WIDTH; j++){

			c = (((i & 8) ^ (j & 8)) * 255);
			checkImageData[(i * CHECKIMAGE_WIDTH + j) * 4 + 0] = c;
			checkImageData[(i * CHECKIMAGE_WIDTH + j) * 4 + 1] = c;
			checkImageData[(i * CHECKIMAGE_WIDTH + j) * 4 + 2] = c;
			checkImageData[(i * CHECKIMAGE_WIDTH + j) * 4 + 3] = 255;
		}
	}

}


function uninitialize(){


	if(texture_Checkerboard){
		gl.deleteTexture(texture_Checkerboard);
		texture_Checkerboard = 0;
	}


	if(vbo_Rect_Texcoord){
		gl.deleteBuffer(vbo_Rect_Texcoord);
		vbo_Rect_Texcoord = 0;
	}


	if(vbo_Rect_Position){
		gl.deleteBuffer(vbo_Rect_Position);
		vbo_Rect_Position = 0;
	}

	if(vao_Rect){
		gl.deleteVertexArray(vao_Rect);
		vao_Rect = 0;
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


		/********** Rectangle **********/
		mat4.identity(modelViewMatrix);
		mat4.identity(modelViewProjectionMatrix);
		mat4.translate(modelViewMatrix, modelViewMatrix, [0.0, 0.0, -4.0]);
		mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

		gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

		gl.bindTexture(gl.TEXTURE_2D, texture_Checkerboard);
		gl.uniform1i(samplerUniform, 0);


		for(var i = 1; i <= 2; i++){
				if(i == 1){

				rect_Position[0] = -2.0;
				rect_Position[1] = -1.0;
				rect_Position[2] = 0.0;

				rect_Position[3] = -2.0;
				rect_Position[4] = 1.0;
				rect_Position[5] = 0.0;

				rect_Position[6] = 0.0;
				rect_Position[7] = 1.0;
				rect_Position[8] = 0.0;

				rect_Position[9] = 0.0;
				rect_Position[10] = -1.0;
				rect_Position[11] = 0.0;
			}
			else{

				rect_Position[0] = 1.0;
				rect_Position[1] = -1.0;
				rect_Position[2] = 0.0;

				rect_Position[3] = 1.0;
				rect_Position[4] = 1.0;
				rect_Position[5] = 0.0;

				rect_Position[6] = 2.4142;
				rect_Position[7] = 1.0;
				rect_Position[8] = -1.4142;

				rect_Position[9] = 2.4142;
				rect_Position[10] = -1.0;
				rect_Position[11] = -1.4142;
			}

			gl.bindVertexArray(vao_Rect);

				gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Rect_Position);
					gl.bufferData(gl.ARRAY_BUFFER, rect_Position, gl.DYNAMIC_DRAW);
				gl.bindBuffer(gl.ARRAY_BUFFER, null);

				gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);

			gl.bindVertexArray(null);

		}
			
	gl.useProgram(null);

	requestAnimationFrame(draw, canvas);

}

