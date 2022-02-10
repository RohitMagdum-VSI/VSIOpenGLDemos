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


//For Grid
var vao_Grid;
var vbo_Grid_Position;

//For Red_And_Blue_Axis
var vao_Axis;
var vbo_Axis_Position;
var vbo_Axis_Color;
var axis_Position = new Float32Array(6);
var axis_Color = new Float32Array(6);




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

	canvas = document.getElementById("10-GraphPaper-RRJ");
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
			console.log("in esc!!\n");
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
	var grid_Position = new Float32Array(40 * 6);

	fillGridPosition(grid_Position);
	

	/********** Grid **********/
	vao_Grid =  gl.createVertexArray();
	gl.bindVertexArray(vao_Grid);
			
		/********** Position **********/
		vbo_Grid_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Grid_Position);
		gl.bufferData(gl.ARRAY_BUFFER, grid_Position,gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

		/********** Color **********/
		gl.vertexAttrib3f(WebGLMacros.AMC_ATTRIBUTE_COLOR, 0.0, 0.0, 1.0);

	gl.bindVertexArray(null);




	/********** Axis **********/
	vao_Axis = gl.createVertexArray();
	gl.bindVertexArray(vao_Axis);

		/********** Position **********/
		vbo_Axis_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Axis_Position);
		gl.bufferData(gl.ARRAY_BUFFER, axis_Position, gl.DYNAMIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);
		
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

		/********** Color **********/
		vbo_Axis_Color = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Axis_Color);
		gl.bufferData(gl.ARRAY_BUFFER , axis_Color, gl.DYNAMIC_DRAW);

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

	gl.clearColor(0.0, 0.0, 0.0, 1.0);

	perspectiveProjectionMatrix = mat4.create();
}


function uninitialize(){

	console.log("in uninitialize() !!\n");

	if(vbo_Axis_Color){
		gl.deleteBuffer(vbo_Axis_Color);
		vbo_Axis_Color = 0;
	}

	if(vbo_Axis_Position){
		gl.deleteBuffer(vbo_Axis_Position);
		vbo_Axis_Position = 0;
	}
	
	if(vao_Axis){
		gl.deleteVertexArray(vao_Axis);
		vao_Axis = 0;
	}


	if(vbo_Grid_Position){
		gl.deleteBuffer(vbo_Grid_Position);
		vbo_Grid_Position = 0;
	}

	if(vao_Grid){
		gl.deleteVertexArray(vao_Grid);
		vao_Grid = 0;
	}

	if(shaderProgramObject){

		gl.useProgram(shaderProgramObject);

			if(fragmentShaderObject){
				gl.detachShader(shaderProgramObject, fragmentShaderObject);
				gl.deleteShader(fragmentShaderObject);
				fragmentShaderObject = null;
			}

			if(vertexShaderObject){
				gl.detachShaderObject(shaderProgramObject, vertexShaderObject);
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
	var axis_Position = new Float32Array(6);
	var axis_Color = new Float32Array(6);

	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);


	gl.useProgram(shaderProgramObject);

		/*********** Grid ***********/
		mat4.translate(modelViewMatrix, modelViewMatrix, [0.0, 0.0, -3.0]);
		mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

		gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

		gl.bindVertexArray(vao_Grid);

			gl.drawArrays(gl.LINES, 0, 40 * 2);

		gl.bindVertexArray(null);



		/************ Axis ***********/
		for(var i = 1; i <= 2; i++){
		
			mat4.identity(modelViewMatrix);
			mat4.identity(modelViewProjectionMatrix);

			mat4.translate(modelViewMatrix, modelViewMatrix, [0.0, 0.0, -3.0]);
			mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

			gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

			
			


			if(i == 1){
				axis_Position[0] = 0.0;
				axis_Position[1] = 1.0;
				axis_Position[2] = 0.0;
			
				axis_Position[3] = 0.0;
				axis_Position[4] = -1.0;
				axis_Position[5] = 0.0;

				axis_Color[0] = 1.0;
				axis_Color[1] = 0.0;
				axis_Color[2] = 0.0;

				axis_Color[3] = 1.0;
				axis_Color[4] = 0.0;
				axis_Color[5] = 0.0;
			}
			else if( i == 2){
				axis_Position[0] = -1.0;
				axis_Position[1] = 0.0;
				axis_Position[2] = 0.0;

				axis_Position[3] = 1.0;
				axis_Position[4] = 0.0;
				axis_Position[5] = 0.0;


				axis_Color[0] = 0.0;
				axis_Color[1] = 1.0;
				axis_Color[2] = 0.0;
				
				axis_Color[3] = 0.0;
				axis_Color[4] = 1.0;
				axis_Color[5] = 0.0;

			}

			gl.bindVertexArray(vao_Axis);

				/***** Position *****/
				gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Axis_Position);
					
					gl.bufferData(gl.ARRAY_BUFFER,
							axis_Position,
							gl.DYNAMIC_DRAW);
				
				gl.bindBuffer(gl.ARRAY_BUFFER, null);

				/***** Color *****/
				gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Axis_Color);
					
					gl.bufferData(gl.ARRAY_BUFFER,
							axis_Color,
							gl.DYNAMIC_DRAW);

				gl.bindBuffer(gl.ARRAY_BUFFER, null);

				
				gl.drawArrays(gl.LINES, 0, 2);

			gl.bindVertexArray(null);

		}






	gl.useProgram(null);

	requestAnimationFrame(draw, canvas);
}


function fillGridPosition(arr){
	
	var j = 0;


	//Vertical Lines 20
	
	for(var i = 0.1; i < 1.0; i = i + 0.1){
	
		//1
		arr[j] = 0.0 + i;
		arr[j + 1] = 1.0;
		arr[j + 2] = 0.0;

		arr[j + 3] = 0.0 + i;
		arr[j + 4] = -1.0;
		arr[j + 5] = 0.0;

		//2
		arr[j + 6] = 0.0 - i;
		arr[j + 7] = 1.0;
		arr[j + 8] = 0.0;
		
		arr[j + 9] = 0.0 - i;
		arr[j + 10] = -1.0;
		arr[j + 11] = 0.0;

		j = j + 12;
	}


	//Horizontal Lines 20 
	for(var i = 0.1; i < 1.0; i = i + 0.1){
		//1
		arr[j] = -1.0;
		arr[j + 1] = 0.0 + i;
		arr[j + 2] = 0.0;

		arr[j + 3] = 1.0;
		arr[j + 4] = 0.0 + i;
	       	arr[j + 5] = 0.0;

		//2
		arr[j + 6] = -1.0;
		arr[j + 7] = 0.0 - i;
		arr[j + 8] = 0.0;

		arr[j + 9] = 1.0;
		arr[j + 10] = 0.0 - i;
		arr[j + 11] = 0.0;

		j = j + 12;	
	}
}



