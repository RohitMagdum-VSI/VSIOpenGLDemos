var canvas_RRJ = null;
var gl = null;
var bIsFullScreen_RRJ = false;
var canvas_original_width_RRJ = 0;
var canvas_original_height_RRJ = 0;

const WebGLMacros = {
	AMC_ATTRIBUTE_POSITION:0,
	AMC_ATTRIBUTE_COLOR:1,
	AMC_ATTRIBUTE_NORMAL:2,
	AMC_ATTRIBUTE_TEXCOORD0:3
};


//For Shader
var vertexShaderObject_RRJ;
var fragmentShaderObject_RRJ;
var shaderProgramObject_RRJ;

//For Uniform
var mvpUniform_RRJ;

//For matrix
var perspectiveProjectionMatrix_RRJ;

//For Triangle
var vao_Triangle_RRJ;
var vbo_Triangle_Position_RRJ;
var vbo_Triangle_Color_RRJ;

//For Viewport
var viewPortWidth_RRJ;
var viewPortHeight_RRJ;
var iViewportNo_RRJ = 0;


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

	canvas_RRJ = document.getElementById("31-Multi-Viewport-RRJ");
	if(!canvas_RRJ)
		console.log("Obtaining Canvas Failed!!\n");
	else
		console.log("Canvas Obtained!!\n");


	canvas_original_width_RRJ = canvas_RRJ.width;
	canvas_original_height_RRJ = canvas_RRJ.height;

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

		if(canvas_RRJ.requestFullscreen)
			canvas_RRJ.requestFullscreen();
		else if(canvas_RRJ.webkitRequestFullscreen)
			canvas_RRJ.webkitRequestFullscreen();
		else if(canvas_RRJ.mozRequestFullScreen)
			canvas_RRJ.mozRequestFullScreen();
		else if(canvas_RRJ.msRequestFullscreen)
			canvas_RRJ.msRequestFullscreen();

		bIsFullScreen_RRJ = true;
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

		bIsFullScreen_RRJ = false;
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


		case 48 :
			iViewportNo_RRJ = 0;
			break;

		case 49:
			iViewportNo_RRJ = 1;
			break;

		case 50:
			iViewportNo_RRJ = 2;
			break;

		case  51:
			iViewportNo_RRJ = 3;
			break;

		case 52:
			iViewportNo_RRJ = 4;
			break;

		case 53:
			iViewportNo_RRJ = 5;
			break;

		case 54:
			iViewportNo_RRJ = 6;
			break;

		case 55:
			iViewportNo_RRJ = 7;
			break;

		case 56:
			iViewportNo_RRJ = 8;
			break;

		case 57:
			iViewportNo_RRJ = 9;
			break;
	}

	if(iViewportNo_RRJ >= 0 && iViewportNo_RRJ <= 9)
		resize();
}

function mouseDown(){

}

function initialize(){


	gl = canvas_RRJ.getContext("webgl2");
	if(gl == null){
		console.log("Obtaining Context Failed!!\n");
		return;
	}
	else
		console.log("Context Obtained!!\n");

	gl.viewportWidth = canvas_RRJ.width;
	gl.viewportHeight = canvas_RRJ.height;


	vertexShaderObject_RRJ = gl.createShader(gl.VERTEX_SHADER);

	var szVertexShaderSourceCode = 
		"#version 300 es" +
		"\n" +
		"in vec4 vPosition;" +
		"in vec4 vColor;" +
		"out vec4 outColor;" +
		"uniform mat4 u_mvp_matrix;" +
		"void main(){" +
			"outColor = vColor;" +
			"gl_Position = u_mvp_matrix * vPosition;" +
		"}";

	gl.shaderSource(vertexShaderObject_RRJ, szVertexShaderSourceCode);
	gl.compileShader(vertexShaderObject_RRJ);

	var shaderCompileStatus = gl.getShaderParameter(vertexShaderObject_RRJ, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(vertexShaderObject_RRJ);
		if(error.length > 0){
			alert("VertexShader Compilation Error : " + error);
			uninitialize();
			window.close();
		}
	}


	fragmentShaderObject_RRJ = gl.createShader(gl.FRAGMENT_SHADER);
	var szFragmentShaderSourceCode = 
		"#version 300 es" +
		"\n" +
		"precision highp float;" +
		"in vec4 outColor;" +
		"out vec4 FragColor;" +
		"void main() {" +
			"FragColor = outColor;" +
		"}";

	gl.shaderSource(fragmentShaderObject_RRJ, szFragmentShaderSourceCode);
	gl.compileShader(fragmentShaderObject_RRJ);

	shaderCompileStatus = gl.getShaderParameter(fragmentShaderObject_RRJ,  gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(fragmentShaderObject_RRJ);
		if(error.length > 0){
			alert("Fragment Shader Compilation Error : " + error);
			uninitialize();
			window.close();

		}
	}



	shaderProgramObject_RRJ = gl.createProgram();

	gl.attachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
	gl.attachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);

	gl.bindAttribLocation(shaderProgramObject_RRJ, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
	gl.bindAttribLocation(shaderProgramObject_RRJ, WebGLMacros.AMC_ATTRIBUTE_COLOR, "vColor");

	gl.linkProgram(shaderProgramObject_RRJ);

	var programLinkStatus = gl.getProgramParameter(shaderProgramObject_RRJ, gl.LINK_STATUS);

	if(programLinkStatus == false){
		var error = gl.getProgramInfoLog(shaderProgramObject_RRJ);
		if(error.length > 0){
			alert("Program Linking Error : " + error);
			uninitialize();
			window.close();
		}
	}


	mvpUniform_RRJ = gl.getUniformLocation(shaderProgramObject_RRJ, "u_mvp_matrix");

	var triangle_Position = new Float32Array([
			0.0, 1.0, 0.0,
			-1.0, -1.0, 0.0,
			1.0, -1.0, 0.0
		]);

	var triangle_Color = new Float32Array([
			1.0, 0.0, 0.0,
			0.0, 1.0, 0.0,
			0.0, 0.0, 1.0
		]);


	vao_Triangle_RRJ = gl.createVertexArray();
	gl.bindVertexArray(vao_Triangle_RRJ);

		/********** Position **********/
		vbo_Triangle_Position_RRJ = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Triangle_Position_RRJ);
		gl.bufferData(gl.ARRAY_BUFFER, triangle_Position, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION, 
							3,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Color **********/
		vbo_Triangle_Color_RRJ = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Triangle_Color_RRJ);
		gl.bufferData(gl.ARRAY_BUFFER, triangle_Color, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_COLOR, 
							3, 
							gl.FLOAT, false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);	

	gl.enable(gl.DEPTH_TEST);
	gl.depthFunc(gl.LEQUAL);

	gl.clearColor(0.0, 0.0, 0.0, 1.0);

	perspectiveProjectionMatrix_RRJ = mat4.create();
}

function uninitialize(){

	if(vbo_Triangle_Color_RRJ){
		gl.deleteBuffer(vbo_Triangle_Color_RRJ);
		vbo_Triangle_Color_RRJ = null;
	}

	if(vbo_Triangle_Position_RRJ){
		gl.deleteBuffer(vbo_Triangle_Position_RRJ);
		vbo_Triangle_Position_RRJ = null;
	}

	if(vao_Triangle_RRJ){
		gl.deleteVertexArray(vao_Triangle_RRJ);
		vao_Triangle_RRJ = null;
	}

	if(shaderProgramObject_RRJ){

		gl.useProgram(shaderProgramObject_RRJ);

			if(fragmentShaderObject_RRJ){
				gl.detachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);
				gl.deleteShader(fragmentShaderObject_RRJ);
				fragmentShaderObject_RRJ = null;
			}

			if(vertexShaderObject_RRJ){
				gl.detachShaderObject(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
				gl.deleteShader(vertexShaderObject_RRJ);
				vertexShaderObject_RRJ = null;
			}

		gl.useProgram(null);
		gl.deleteProgram(shaderProgramObject_RRJ);
		shaderProgramObject_RRJ = null;
	}
}

function resize(){

	if(bIsFullScreen_RRJ == true){
		canvas_RRJ.width = window.innerWidth;
		canvas_RRJ.height = window.innerHeight;
	}
	else{
		canvas_RRJ.width = canvas_original_width_RRJ;
		canvas_RRJ.height = canvas_original_height_RRJ;
	}

	viewPortWidth_RRJ = canvas_RRJ.width;
	viewPortHeight_RRJ = canvas_RRJ.height;

	if (iViewportNo_RRJ == 0)
		gl.viewport(0, 0, viewPortWidth_RRJ, viewPortHeight_RRJ);
	else if (iViewportNo_RRJ == 1)
		gl.viewport(0, 0, (viewPortWidth_RRJ) / 2, (viewPortHeight_RRJ) / 2);
	else if (iViewportNo_RRJ == 2)
		gl.viewport(viewPortWidth_RRJ / 2, 0, viewPortWidth_RRJ / 2, viewPortHeight_RRJ / 2);
	else if (iViewportNo_RRJ == 3)
		gl.viewport(viewPortWidth_RRJ / 2, viewPortHeight_RRJ / 2, viewPortWidth_RRJ / 2, viewPortHeight_RRJ / 2);
	else if (iViewportNo_RRJ == 4)
		gl.viewport(0, viewPortHeight_RRJ / 2, viewPortWidth_RRJ / 2, viewPortHeight_RRJ / 2);
	else if (iViewportNo_RRJ == 5)
		gl.viewport(0, 0, viewPortWidth_RRJ / 2, viewPortHeight_RRJ);
	else if (iViewportNo_RRJ == 6)
		gl.viewport(viewPortWidth_RRJ / 2, 0, viewPortWidth_RRJ / 2, viewPortHeight_RRJ);
	else if (iViewportNo_RRJ == 7)
		gl.viewport(0, viewPortHeight_RRJ / 2, viewPortWidth_RRJ, viewPortHeight_RRJ / 2);
	else if (iViewportNo_RRJ == 8)
		gl.viewport(0, 0, viewPortWidth_RRJ, viewPortHeight_RRJ / 2);
	else if (iViewportNo_RRJ == 9)
		gl.viewport(viewPortWidth_RRJ / 4, viewPortHeight_RRJ / 4, viewPortWidth_RRJ / 2, viewPortHeight_RRJ / 2);


	mat4.perspective(perspectiveProjectionMatrix_RRJ, 
					45.0,
					parseFloat(canvas_RRJ.width) / parseFloat(canvas_RRJ.height),
					0.1,
					100.0);
}

function draw(){

	var translateMatrix = mat4.create();
	var modelViewMatrix = mat4.create();
	var modelViewProjectionMatrix = mat4.create();


	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);


	gl.useProgram(shaderProgramObject_RRJ);

	mat4.translate(modelViewMatrix, modelViewMatrix, [0.0, 0.0, -4.0]);
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix_RRJ, modelViewMatrix);

	gl.uniformMatrix4fv(mvpUniform_RRJ, false, modelViewProjectionMatrix);

	gl.bindVertexArray(vao_Triangle_RRJ);

		gl.drawArrays(gl.TRIANGLES, 0, 3);

	gl.bindVertexArray(null);

	gl.useProgram(null);

	requestAnimationFrame(draw, canvas_RRJ);
}

