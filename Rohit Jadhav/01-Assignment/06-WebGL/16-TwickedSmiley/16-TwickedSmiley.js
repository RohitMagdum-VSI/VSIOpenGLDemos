var canvas = null;
var gl = null;
var bIsFullScreen = false;
var canvas_original_width = 0;
var canvas_original_height = 0;


const WebGLMacros = {
	AMC_ATTRIBUTE_POSITION:1,
	AMC_ATTRIBUTE_COLOR:2,
	AMC_ATTRIBUTE_NORMAL:3,
	AMC_ATTRIBUTE_TEXCOORD0:4,
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

//For Rect
var vao_Rect;
var vbo_Rect_Position;
var vbo_Rect_Texcoord;
var texture_Smiley;
var rect_Texcoord = new Float32Array(4 * 2);
var iKey = 1;


function main(){

	canvas = document.getElementById("16-TwickedSmiley-RRJ");
	if(canvas)
		console.log("Canvas Obtained!!");
	else
		console.log("Failed To Get Canvas!!");


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

		case 49:
			iKey = 1;
			console.log("1");
			break;

		case 50:
			iKey = 2;
			break;

		case 51:
			iKey = 3;
			break;

		case 52:
			iKey = 4;
			break;

		case 70:
			toggleFullScreen();
			console.log(event);
			console.log(event.keyCode);
			break;
	}




}

function mouseDown(){

}




function initialize(){

	gl = canvas.getContext("webgl2");
	if(gl)
		console.log("Context Obtained!!");
	else
		console.log("Context Not Obtained!!");


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
		"void main() {" +
			"FragColor = texture(u_sampler, outTex);" +
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


	texture_Smiley = gl.createTexture();
	texture_Smiley.image = new Image();
	texture_Smiley.image.src = "smiley.png";
	texture_Smiley.image.onload = function(){
		gl.bindTexture(gl.TEXTURE_2D, texture_Smiley);
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, texture_Smiley.image);
		gl.generateMipmap(gl.TEXTURE_2D);
		gl.bindTexture(gl.TEXTURE_2D, null);
	};

	mvpUniform = gl.getUniformLocation(shaderProgramObject, "u_mvp_matrix");
	samplerUniform = gl.getUniformLocation(shaderProgramObject, "u_sampler");


	var rect_Position = new Float32Array([
						1.0, 1.0, 0.0,
						-1.0, 1.0, 0.0,
						-1.0, -1.0, 0.0,
						1.0, -1.0, 0.0,
					]);

	



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


		/********** Texture **********/
		vbo_Rect_Texcoord = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Rect_Texcoord);
		gl.bufferData(gl.ARRAY_BUFFER, rect_Texcoord, gl.DYNAMIC_DRAW);
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



function uninitialize(){


	if(texture_Smiley){
		gl.deleteTexture(texture_Smiley);
		texture_Smiley = 0;
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


		if(iKey == 1){
			rect_Texcoord[0] = 0.50;
			rect_Texcoord[1] = 0.50;

			rect_Texcoord[2] = 0.0;
			rect_Texcoord[3] = 0.50;

			rect_Texcoord[4] = 0.0;
			rect_Texcoord[5] = 0.0;

			rect_Texcoord[6] = 0.50;
			rect_Texcoord[7] = 0.0;
		}
		else if(iKey == 2){

			rect_Texcoord[0] = 1.0;
			rect_Texcoord[1] = 1.0;

			rect_Texcoord[2] = 0.0;
			rect_Texcoord[3] = 1.0;

			rect_Texcoord[4] = 0.0;
			rect_Texcoord[5] = 0.0;

			rect_Texcoord[6] = 1.0;
			rect_Texcoord[7] = 0.0;

		}
		else if(iKey == 3){
			rect_Texcoord[0] = 2.0;
			rect_Texcoord[1] = 2.0;

			rect_Texcoord[2] = 0.0;
			rect_Texcoord[3] = 2.0;

			rect_Texcoord[4] = 0.0;
			rect_Texcoord[5] = 0.0;

			rect_Texcoord[6] = 2.0;
			rect_Texcoord[7] = 0.0;
		}
		else if(iKey == 4){
			rect_Texcoord[0] = 0.50;
			rect_Texcoord[1] = 0.50;

			rect_Texcoord[2] = 0.50;
			rect_Texcoord[3] = 0.50;

			rect_Texcoord[4] = 0.50;
			rect_Texcoord[5] = 0.50;

			rect_Texcoord[6] = 0.50;
			rect_Texcoord[7] = 0.50;
		}

		gl.bindTexture(gl.TEXTURE_2D, texture_Smiley);
		gl.uniform1i(samplerUniform, 0);


		gl.bindVertexArray(vao_Rect);

			gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Rect_Texcoord);
				gl.bufferData(gl.ARRAY_BUFFER, rect_Texcoord, gl.DYNAMIC_DRAW);
			gl.bindBuffer(gl.ARRAY_BUFFER, null);

			gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
		gl.bindVertexArray(null);

			
	gl.useProgram(null);

	requestAnimationFrame(draw, canvas);

}

