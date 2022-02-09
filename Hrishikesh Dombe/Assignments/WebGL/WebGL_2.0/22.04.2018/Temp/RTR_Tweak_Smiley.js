//Global Variables
var canvas = null;
var gl = null; //WebGL Context
var bFullScreen = false;
var canvas_original_width;
var canvas_original_height;

const WebGLMacros = 
{
	NRK_ATTRIBUTE_VERTEX: 0,
	NRK_ATTRIBUTE_COLOR: 1,
	NRK_ATTRIBUTE_NORMAL: 2,
	NRK_ATTRIBUTE_TEXTURE0: 3,
};

var vertexShaderObject;
var fragmentShaderObject;
var shaderProgramObject;

var vao;
var vbo_position;
var vbo_texture;

var mvpUniform;

var perspectiveProjectionMatrix;

var smiley_texture = 0;

var uniform_texture0_sampler;

var digitPressedIs = 0;

var squareTexCoords = new Float32Array(8);

//To Start Animation
var requestAnimationFrame = 
	window.requestAnimationFrame || 
	window.webkitRequestAnimationFrame || 
	window.mozRequestAnimationFrame || 
	window.oRequestAnimationFrame || 
	window.msRequestAnimationFrame;

//To Stop Animation
var cnacelAnimationFrame = 
	window.cancelAnimationFrame || 
	window.webkitCancelRequestAnimationFrame || window.webkitCancelAnimationFrame || 
	window.mozCancelRequestAnimationFrame || window.mozCancelAnimationFrame || 
	window.oCancelRequestAnimationFrame || window.oCancelAnimationFrame || 
	window.msCancelRequestAnimationFrame || window.msCancelAnimationFrame;

//onload Function
function main()
{
	//Get Canvas Element
	canvas = document.getElementById("HAD");

	if (!canvas)
		console.log("Obtaining Canvas failed!\n");
	else
		console.log("Obtaining Canvas succeeded!\n");

	canvas_original_width = canvas.width;
	canvas_original_height = canvas.height;

	//Register events
	window.addEventListener("keydown", keyDown, false);
	window.addEventListener("click", mouseDown, false);
	window.addEventListener("resize", resize, false);

	//Initialize WebGL
	init();

	//Start drawing here as warming-up
	resize();
	draw();
}

function toggleFullScreen()
{
	var fullScreen_element =
		document.fullscreenElement ||
		document.webkitFullscreenElement ||
		document.mozFullScreenElement ||
		document.msFullscreenElement ||
		null;

	//If not fullscreen
	if (fullScreen_element == null) {
		if (canvas.requestFullscreen)
			canvas.requestFullscreen();
		else if (canvas.mozRequestFullScreen)
			canvas.mozRequestFullScreen();
		else if (canvas.webkitRequestFullscreen)
			canvas.webkitRequestFullscreen();
		else if (canvas.msRequestFullscreen)
			canvas.msRequestFullscreen();

		bFullScreen = true;
	}
	else //If already fullscreen
	{
		if (document.exitFullscreen)
			document.exitFullscreen();
		else if (document.mozCancelFullScreen)
			document.mozCancelFullScreen();
		else if (document.webkitExitFullscreen)
			document.webkitExitFullscreen();
		else if (document.msExitFullscreen)
			document.msExitFullscreen();

		bFullScreen = false;
	}
}

function init()
{
	//Get WebGL 2.0 Context
	gl = canvas.getContext("webgl2");

	if(gl == null)
	{
		console.log("Failed to get WebGL Context!\n");
		return;
	}

	gl.viewportWidth = canvas.width;
	gl.viewportHeight = canvas.height;

	//Vertex Shader
	//Create Shader
	vertexShaderObject = gl.createShader(gl.VERTEX_SHADER);

	//Shader Source Code
	var vertexShaderSourceCode = 
		"#version 300 es"+
		"\n" +
		"in vec4 vPosition;" +
		"in vec2 vTexture0_Coord;" +
		"out vec2 out_texture0_coord;" +
		"uniform mat4 u_mvp_matrix;" +
		"void main(void)" +
		"{" +
		"gl_Position = u_mvp_matrix * vPosition;" +
		"out_texture0_coord = vTexture0_Coord;" +
		"}";

	//Provide Source Code to Shader Object
	gl.shaderSource(vertexShaderObject, vertexShaderSourceCode);

	//Compile Shader
	gl.compileShader(vertexShaderObject);

	//Error Checking
	if(gl.getShaderParameter(vertexShaderObject, gl.COMPILE_STATUS) == false)
	{
		var error = gl.getShaderInfoLog(vertexShaderObject);
		if(error.length > 0)
		{
			alert(error);
			uninitialize();
		}
	}

	//Fragment Shader
	//Create Shader
	fragmentShaderObject = gl.createShader(gl.FRAGMENT_SHADER);

	//Shader Source Code
	var fragmentShaderSourceCode = 
		"#version 300 es" +
		"\n" +
		"precision highp float;" +
		"in vec2 out_texture0_coord;" +
		"uniform highp sampler2D u_texture0_sampler;" +
		"out vec4 FragColor;" +
		"void main(void)" +
		"{" +
		"FragColor = texture(u_texture0_sampler, out_texture0_coord);" +
		"}";

	//Provide Source Code to Shader Object
	gl.shaderSource(fragmentShaderObject, fragmentShaderSourceCode);

	//Compile Shader
	gl.compileShader(fragmentShaderObject);

	//Error Checking
	if(gl.getShaderParameter(fragmentShaderObject, gl.COMPILE_STATUS) == false)
	{
		var error = gl.getShaderInfoLog(fragmentShaderObject);
		if(error.length > 0)
		{
			alert(error);
			uninitialize();
		}
	}

	//Shader Program
	//Create Shader Program
	shaderProgramObject = gl.createProgram();

	//Attach Shaders
	gl.attachShader(shaderProgramObject, vertexShaderObject);
	gl.attachShader(shaderProgramObject, fragmentShaderObject);

	//Pre-Link Binding
	gl.bindAttribLocation(shaderProgramObject, WebGLMacros.NRK_ATTRIBUTE_VERTEX, "vPosition");
	gl.bindAttribLocation(shaderProgramObject, WebGLMacros.NRK_ATTRIBUTE_TEXTURE0, "vTexture0_Coord");

	//Link
	gl.linkProgram(shaderProgramObject);

	//Error Checking
	if(!gl.getProgramParameter(shaderProgramObject, gl.LINK_STATUS))
	{
		var error = gl.getProgramInfoLog(shaderProgramObject);

		if(error.length > 0)
		{
			alert(error);
			unitialize();
		}
	}

	//Load Pyramid Textures
	smiley_texture = gl.createTexture();
	smiley_texture.image = new Image();
	smiley_texture.image.src = "smiley.png";
	smiley_texture.image.onload = function()
	{ //Function Inside a function. Similar to lambda
		gl.bindTexture(gl.TEXTURE_2D, smiley_texture);
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, smiley_texture.image);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.bindTexture(gl.TEXTURE_2D, null);
	}

	//Get MVP Uniform Location
	mvpUniform = gl.getUniformLocation(shaderProgramObject, "u_mvp_matrix");

	//Get Texture Uniform Location
	uniform_texture0_sampler = gl.getUniformLocation(shaderProgramObject, "u_texture0_sampler");

	//Vertices, Colors, Shader Attribs, Vao, Vbo Initializations
	var squareVertices = new Float32Array([
											1.0, 1.0, 0.0, //Right-Top Vertex
											-1.0, 1.0, 0.0, //Left-Top Vertex
											-1.0, -1.0, 0.0, //Left-Bottom Vertex
											1.0, -1.0, 0.0 //Right-Bottom Vertex
										]);

	//Create vao
	vao = gl.createVertexArray();
	gl.bindVertexArray(vao); //Binding with vao

	//----------Vbo for Position Starts----------
	vbo_position = gl.createBuffer(); //Create vbo_position
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_position); //Binding with vbo_position
	gl.bufferData(gl.ARRAY_BUFFER, squareVertices, gl.STATIC_DRAW);
	gl.vertexAttribPointer(WebGLMacros.NRK_ATTRIBUTE_VERTEX, 3, gl.FLOAT, false, 0, 0);
	gl.enableVertexAttribArray(WebGLMacros.NRK_ATTRIBUTE_VERTEX);
	gl.bindBuffer(gl.ARRAY_BUFFER, null); //Unbinding with vbo_position
	//----------Vbo for Position Ends----------
	
	//----------Vbo for Texture Starts----------
	vbo_texture = gl.createBuffer(); //Create vbo_texture
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_texture); //Binding with vbo_texture
	gl.bufferData(gl.ARRAY_BUFFER, squareTexCoords, gl.DYNAMIC_DRAW);
	gl.vertexAttribPointer(WebGLMacros.NRK_ATTRIBUTE_TEXTURE0, 2, gl.FLOAT, false, 0, 0);
	gl.enableVertexAttribArray(WebGLMacros.NRK_ATTRIBUTE_TEXTURE0);
	gl.bindBuffer(gl.ARRAY_BUFFER, null); //Unbinding with vbo_texture
	//----------Vbo for Texture Ends----------

	gl.bindVertexArray(null); //Unbinding with vao

	//Enable Depth Testing
	gl.enable(gl.DEPTH_TEST);

	//Depth Test to do
	gl.depthFunc(gl.LEQUAL);

	//Culling Back Faces (Always)
	//gl.enable(gl.CULL_FACE);

	gl.clearColor(0.0, 0.0, 0.0, 1.0); //Black Color

	//Initialize Perspective Matrix
	perspectiveProjectionMatrix = mat4.create();
}

function resize()
{
	if (bFullScreen == true) {
		canvas.width = window.innerWidth;
		canvas.height = window.innerHeight;
	}
	else {
		canvas.width = canvas_original_width;
		canvas.height = canvas_original_height;
	}

	gl.viewport(0, 0, canvas.width, canvas.height);

	mat4.perspective(perspectiveProjectionMatrix, 45.0, parseFloat(canvas.width) / parseFloat(canvas.height), 0.1, 100.0);
}

function draw()
{
	//Code
	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

	gl.useProgram(shaderProgramObject);

	var modelViewMatrix = mat4.create();
	var modelViewProjectionMatrix = mat4.create();

	//Translate Z-axis by -3.0
	mat4.translate(modelViewMatrix, modelViewMatrix, [0.0, 0.0, -3.0]);

	//Multiply ModelViewMatix and ProjectionMatrix
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

	gl.bindVertexArray(vao); //Bind with Vao

	//var squareTexCoords = new Float32Array();

	if(digitPressedIs == 1)
	{
		/*squareTexCoords = new Float32Array([
            0.0,0.0,
            1.0,0.0,
            1.0,1.0,
            0.0,1.0,]);*/
		//Right-Top Vertex
		squareTexCoords[0] = 0.5;
		squareTexCoords[1] = 0.5;

		//Left-Top Vertex
		squareTexCoords[2] = 0.0;
		squareTexCoords[3] = 0.5;

		//Left-Bottom Vertex
		squareTexCoords[4] = 0.0;
		squareTexCoords[5] = 0.0;

		//Right-Bottom Vertex
		squareTexCoords[6] = 0.5;
		squareTexCoords[7] = 0.0;
	}

	else if(digitPressedIs == 2)
	{
		//Right-Top Vertex
		squareTexCoords[0] = 1.0;
		squareTexCoords[1] = 1.0;

		//Left-Top Vertex
		squareTexCoords[2] = 0.0;
		squareTexCoords[3] = 1.0;

		//Left-Bottom Vertex
		squareTexCoords[4] = 0.0;
		squareTexCoords[5] = 0.0;

		//Right-Bottom Vertex
		squareTexCoords[6] = 1.0;
		squareTexCoords[7] = 0.0;
	}

	else if(digitPressedIs == 3)
	{
		//Right-Top Vertex
		squareTexCoords[0] = 2.0;
		squareTexCoords[1] = 2.0;

		//Left-Top Vertex
		squareTexCoords[2] = 0.0;
		squareTexCoords[3] = 2.0;

		//Left-Bottom Vertex
		squareTexCoords[4] = 0.0;
		squareTexCoords[5] = 0.0;

		//Right-Bottom Vertex
		squareTexCoords[6] = 2.0;
		squareTexCoords[7] = 0.0;
	}

	else if(digitPressedIs == 4)
	{
		//Right-Top Vertex
		squareTexCoords[0] = 0.5;
		squareTexCoords[1] = 0.5;

		//Left-Top Vertex
		squareTexCoords[2] = 0.5;
		squareTexCoords[3] = 0.5;

		//Left-Bottom Vertex
		squareTexCoords[4] = 0.5;
		squareTexCoords[5] = 0.5;

		//Right-Bottom Vertex
		squareTexCoords[6] = 0.5;
		squareTexCoords[7] = 0.5;
	}

	//----------Vbo for Texture Starts----------
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_texture); //Binding with vbo_texture
	gl.bufferData(gl.ARRAY_BUFFER, squareTexCoords, gl.DYNAMIC_DRAW);
	gl.bindBuffer(gl.ARRAY_BUFFER, null); //Unbinding with vbo_texture
	//----------Vbo for Texture Ends----------

	//Bind with Texture
	gl.bindTexture(gl.TEXTURE_2D, smiley_texture);
	gl.uniform1i(uniform_texture0_sampler, 0);

	gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
	gl.bindVertexArray(null); //Unbinding with Vao

	gl.useProgram(null);

	//Animation Loop
	requestAnimationFrame(draw, canvas);
}

function keyDown(event)
{
	//Code
	switch(event.keyCode)
	{
		case 27: //For Escape
			uninitialize();
			window.close();
			break;

		case 70: //For 'F' or 'f'
			toggleFullScreen();
			break;

		case 49: //For '1'
			//gl.enable(gl.TEXTURE_2D);
			digitPressedIs = 1;
			break;

		case 50: //For '2'
			//gl.enable(gl.TEXTURE_2D);
			digitPressedIs = 2;
			break;

		case 51: //For '3'
			//gl.enable(gl.TEXTURE_2D);
			digitPressedIs = 3;
			break;

		case 52: //For '4'
			//gl.enable(gl.TEXTURE_2D);
			digitPressedIs = 4;
			break;
	}
}

function mouseDown(event)
{
	//Code
}

function uninitialize()
{
	//Code
	if (smiley_texture)
	{
		gl.deleteTexture(smiley_texture);
		smiley_texture = 0;
	}

	if(vao)
	{
		gl.deleteVertexArray(vao);
		vao = null;
	}

	if(vbo_position)
	{
		gl.deleteBuffer(vbo_position);
		vbo_position = null;
	}

	if(vbo_texture)
	{
		gl.deleteBuffer(vbo_texture);
		vbo_texture = null;
	}

	if(shaderProgramObject)
	{
		if(vertexShaderObject)
		{
			gl.detachShader(shaderProgramObject, vertexShaderObject);
			gl.deleteShader(vertexShaderObject);
			vertexShaderObject = null;
		}

		if(fragmentShaderObject)
		{
			gl.detachShader(shaderProgramObject, fragmentShaderObject);
			gl.deleteShader(fragmentShaderObject);
			fragmentShaderObject = null;
		}

		gl.deleteProgram(shaderProgramObject);
		shaderProgramObject = null;
	}
}

