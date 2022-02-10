var canvas = null;
var gl = null;
var bIsFullScreen = false;
var canvas_original_width;
var canvas_original_height;


var requestAnimationFrame = 
	window.requestAnimationFrame ||
	window.mozRequestAnimationFrame ||
	window.webkitRequestAnimationFrame ||
	window.msRequestAnimationFrame || 
	window.oRequestAnimationFrame ||
	null;


var cancelAnimationFrame = 
	window.cancelAnimationFrame ||
	window.webkitCancelRequestAnimationFrame || window.webkitCancelAnimationFrame || 
	window.mozCancelRequestAnimationFrame || window.mozCancelAnimationFrame ||
	window.oCancelRequestAnimationFrame || window.oCancelAnimationFrame || 
	window.msCancelRequestAnimationFrame || window.msCancelAnimationFrame || 
	null;


function main(){

	canvas = document.getElementById("RRJ-Ortho");
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
		document.mozFullScreenElement || 
		document.webkitFullscreenElement || 
		document.msFullscreenElement || 
		null;


	if(fullscreen_element == null){

		if(canvas.requestFullscreen)
			canvas.requestFullscreen();
		else if(canvas.mozRequestFullScreen)
			canvas.mozRequestFullScreen();
		else if(canvas.webkitRequestFullscreen)
			canvas.webkitRequestFullscreen();
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



const WEBGLMacros = {
	AMC_ATTRIBUTE_POSITION:0,
	AMC_ATTRIBUTE_COLOR:1,
	AMC_ATTRIBUTE_NORMAL: 2,
	AMC_ATTRIBUTE_TEXCOORD0:3
};


//For Shader
var vertexShaderObject;
var fragmentShaderObject;
var shaderProgramObject;

//For Triangle
var vao_Triangle;
var vbo_Triangle_Position;

//For Uniform
var mvpUniform;


//For Projection
var orthographicProjectionMatrix;


function initialize(){

	gl = canvas.getContext("webgl2");
	if(gl == null){
		console.log("Context Failed!!\n");
		return;
	}
	

	gl.viewportWidth = canvas.width;
	gl.viewportHeight = canvas.height;



	vertexShaderObject = gl.createShader(gl.VERTEX_SHADER);

	var vertexShaderCode = 
		"#version 300 es" +
		"\n" +
		"in vec4 vPosition;" +
		"uniform mat4 u_mvp_matrix;" +
		"void main(void) " +
		"{" +
			"gl_Position = u_mvp_matrix * vPosition;" +
		"}";

	gl.shaderSource(vertexShaderObject, vertexShaderCode);
	gl.compileShader(vertexShaderObject);
	if(gl.getShaderParameter(vertexShaderObject, gl.COMPILE_STATUS) == false){
		var error = gl.getShaderInfoLog(vertexShaderObject);
		if(error.length > 0){
			alert(error);
			uninitialize();
		}
	}


	fragmentShaderObject = gl.createShader(gl.FRAGMENT_SHADER);

	var fragmentShaderCode = 
		"#version 300 es" +
		"\n" +
		"precision highp float;" +
		"out vec4 FragColor;" +
		"void main(void) " +
		"{" +
			"FragColor = vec4(1.0, 1.0, 1.0, 1.0);" +
		"}";

	gl.shaderSource(fragmentShaderObject, fragmentShaderCode);
	gl.compileShader(fragmentShaderObject);

	if(gl.getShaderParameter(fragmentShaderObject, gl.COMPILE_STATUS) == false){
		var error = gl.getShaderInfoLog(fragmentShaderObject);
		if(error.length > 0){
			alert(error);
			uninitialize();
		}
	}


	shaderProgramObject = gl.createProgram();

	gl.attachShader(shaderProgramObject, vertexShaderObject);
	gl.attachShader(shaderProgramObject, fragmentShaderObject);

	gl.bindAttribLocation(shaderProgramObject, WEBGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");

	gl.linkProgram(shaderProgramObject);

	if(!gl.getProgramParameter(shaderProgramObject, gl.LINK_STATUS)){
		var error = gl.getProgramInfoLog(shaderProgramObject);
		if(error.length > 0){
			alert(error);
			uninitialize();
		}
	}


	mvpUniform = gl.getUniformLocation(shaderProgramObject, "u_mvp_matrix");



	/********** Position **********/
	var triangle_Position = new Float32Array([
									0.0, 50.0, 0.0,
									-50.0, -50.0, 0.0,
									50.0, -50.0, 0.0 
								]);

	vao_Triangle = gl.createVertexArray();
	gl.bindVertexArray(vao_Triangle);

		/********** Position **********/
		vbo_Triangle_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Triangle_Position);
		gl.bufferData(gl.ARRAY_BUFFER, triangle_Position, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WEBGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WEBGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);
	
	gl.bindVertexArray(null);

	gl.clearColor(0.0, 0.0, 1.0, 1.0);

	orthographicProjectionMatrix = mat4.create()
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

	if(canvas.width <= canvas.height){
		mat4.ortho(orthographicProjectionMatrix, 
			-100.0, 100.0,
			(-100.0 * (canvas.height / canvas.width)), (100.0 * (canvas.height / canvas.width)),
			-100.0,
			100.0);
	}
	else{
		mat4.ortho(orthographicProjectionMatrix,
				(-100.0 * (canvas.width / canvas.height)), (100.0 * (canvas.width / canvas.height)),
				-100.0, 100.0,
				-100.0, 100.0);
	}
}

function draw(){


	gl.clear(gl.COLOR_BUFFER_BIT);

	gl.useProgram(shaderProgramObject);

		var modelViewMatrix = mat4.create();
		var modelViewProjectionMatrix = mat4.create();

		mat4.multiply(modelViewProjectionMatrix, orthographicProjectionMatrix, modelViewMatrix);

		gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

		gl.bindVertexArray(vao_Triangle);
			gl.drawArrays(gl.TRIANGLES, 0, 3);
		gl.bindVertexArray(null);

	gl.useProgram(null)

	requestAnimationFrame(draw, canvas);
}

function uninitialize(){

	

	if(vbo_Triangle_Position){
		gl.deleteBuffer(vbo_Triangle_Position);
		vbo_Triangle_Position = null;
	}

	if(vao_Triangle){
		gl.deleteVertexArray(vao_Triangle);
		vao_Triangle = null;
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

function keyDown(event){

	switch(event.keyCode){
		case 70:
			toggleFullScreen();
			break;

		case 27:
			uninitialize();
			window.close();
			break;
	}
}

function mouseDown(){

}

