// On load function
var g_canvas = null;
var g1 = null;	//	Webgl context

var g_bFullScreen = false;
var canvas_original_height;
var canvas_original_width;

const WebGLMacros= 
					{
						VTG_ATTRIBUTE_VERTEX:0,
						VTG_ATTRIBUTE_COLOR:1,
						VTG_ATTRIBUTE_NORMAL:2,
						VTG_ATTRIBUTE_TEXTURE0:3,
					};

var shaderObjectVertex;
var shaderObjectFragment;
var shaderObjectProgram;

var vao;
var vbo;
var mvpUniform;

var orthographicProjectionMatrix;

//	To start animation : Have requestAnimationFrame() to be called "cross-browser" comapatible.  
var requestAnimationFrame = 
							window.requestAnimationFrame		||
							window.webkitRequestAnimationFrame	||
							window.mozRequestAnimationFrame		||
							window.oRequestAnimationFrame		||
							window.msRequestAnimationFrame;
							
//	To stop animation: To have cancelAnimationFrame() to be called "cross-browser" compatible.
var cancelAnimationFrame = 
							window.cancelAnimationFrame					||
							window.webkitCancelRequestAnimationFrame	||							
							window.mozCancelAnimationFrame				||
							window.mozCancelRequestAnimationFrame		||
							window.oCancelAnimationFrame				||
							window.oCancelRequestAnimationFrame			||
							window.msCancelAnimationFrame				||
							window.msCancelRequestAnimationFrame;							

function main()
{
	// Get <cnavas> element 
	g_canvas = document.getElementById("AMC");
	if (!g_canvas)
		console.log("Obtaining canvas failed\n");
	else
		console.log("Obtaining canvas succeeded");
	
	//	Print convas width and height
	console.log("canvas width "+g_canvas.width+" and canvas height "+g_canvas.height);
	
	canvas_original_height = g_canvas.height;
	canvas_original_width = g_canvas.width;

	//	Register keyboards keydown event handler 
	window.addEventListener("keydown", keyDown, false);
	window.addEventListener("click", mouseDown, false);
	window.addEventListener("resize", resize, false);

	//	Initialize Webgl.
	init();
	
	//	Start drawing here as warming up.
	resize();
	draw();
}

function toggleFullscreen()
{
	//	Code
	var fullscreen_element = document.fullscreenElement				||
							 document.webkitFullscreenElement		||	//	Apple browser
							 document.mozFullScreenElement			||	//	Mozilla browser
							 document.msFullscreenElement			||	//	Microsoft (IE)
							 null;
							 
	if (null == fullscreen_element)
	{
		if (g_canvas.requestFullscreen)
			g_canvas.requestFullscreen();
		else if (g_canvas.mozRequestFullScreen)
			g_canvas.mozRequestFullScreen();
		else if (g_canvas.webkitRequestFullscreen)
			g_canvas.webkitRequestFullscreen();
		else if (g_canvas.msRequestFullscreen)
			g_canvas.msRequestFullscreen();
		
		g_bFullScreen = true;
	}
	else	//	If already fullscreen 
	{
		if (document.exitFullscreen)
			document.exitFullscreen();
		else if (document.mozCancelFullScreen)
			document.mozCancelFullScreen();
		else if (document.webkitExitFullscreen)
			document.webkitExitFullscreen();
		else if (document.msExitFullscreen)
			document.msExitFullscreen();
		
		g_bFullScreen = false;
	}
}

function keyDown(event)
{
	switch(event.key)
	{
		case 'F':	//	for 'F' or 'f'
		case 'f':	//	for 'F' or 'f'
			toggleFullscreen();
			
			break;
			
		//case "Escape":	// Escape
		case 'E':
		case 'e':
			window.close();	//	May not work in firefox but works in safari and chrome.
			break;
	}
}

function mouseDown()
{
	
}


function init()
{
	//	Get 2D Context
	gl = g_canvas.getContext("webgl2");
	if (null == gl)
	{
		console.log("Failed to get rendering context for webgl 2\n");
		return;
	}
	
	gl.viewportWidth = g_canvas.width;
	gl.viewportHeight = g_canvas.height;

	var shaderSourceCodeVertex = 
	"#version 300 es"				+
	"\n"							+
	"in vec4 vPosition;"			+
	"uniform mat4 u_mvp_matrix;"	+
	"void main(void)"			+
	"{"			+
	"gl_Position = u_mvp_matrix * vPosition;"			+
	"}";
	
	shaderObjectVertex = gl.createShader(gl.VERTEX_SHADER);
	
	gl.shaderSource(shaderObjectVertex, shaderSourceCodeVertex);
	
	gl.compileShader(shaderObjectVertex);
	
	if (false == gl.getShaderParameter(shaderObjectVertex, gl.COMPILE_STATUS))
	{
		var err = gl.getShaderInfoLog(shaderObjectVertex);
		if (err.length > 0)
		{
			alert(err);
			uninitialize();
			//return;
		}
	}

	var shaderSourceCodeFragment = 
	"#version 300 es"				+
	"\n"							+
	"precision highp float;"			+
	"out vec4 FragColor;"	+
	"void main(void)"			+
	"{"			+
	"FragColor = vec4(1.0,1.0,1.0,1.0);"			+
	"}";
	
	shaderObjectFragment = gl.createShader(gl.FRAGMENT_SHADER);
	
	gl.shaderSource(shaderObjectFragment, shaderSourceCodeFragment);
	
	gl.compileShader(shaderObjectFragment);
	
	if (false == gl.getShaderParameter(shaderObjectFragment, gl.COMPILE_STATUS))
	{
		var err = gl.getShaderInfoLog(shaderObjectFragment);
		if (err.length > 0)
		{
			alert(err);
			uninitialize();
			//return;
		}
	}
	
	//	Create program.
	shaderObjectProgram = gl.createProgram();
	
	gl.attachShader(shaderObjectProgram, shaderObjectVertex);
	gl.attachShader(shaderObjectProgram, shaderObjectFragment);
	
	//	Pre-link binding of shader program object with vertex shader attributes.
	gl.bindAttribLocation(shaderObjectProgram,WebGLMacros.VTG_ATTRIBUTE_VERTEX,"vPosition");
	
	//	Linking
	gl.linkProgram(shaderObjectProgram);
	if (false == gl.getProgramParameter(shaderObjectProgram, gl.LINK_STATUS))
	{
		var err = gl.getPRogramInfoLog(shaderObjectProgram);
		if (err.length > 0)
		{
			alert(err);
			uninitialize();
			//return;
		}
	}
	
	//	Get MVP uniform.
	mvpUniform = gl.getUniformLocation(shaderObjectProgram, "u_mvp_matrix");
	
	///////////////////////////////////////////////////////////////////////////////
	//+	Vertices, color , shader attributes , VAO , VBO
	
	var triangleVertices = new Float32Array([
												0.0,50.0,0.0,
												-50.0,-50.0,0,
												50.0, -50.0,0
											]);											
	
	//+	Create VAO
	vao = gl.createVertexArray();
	gl.bindVertexArray(vao);
	
	//+	Create VBO
	vbo = gl.createBuffer(gl.ARRAY_BUFFER, vbo);
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
	gl.bufferData(gl.ARRAY_BUFFER, triangleVertices, gl.STATIC_DRAW);
	gl.vertexAttribPointer(WebGLMacros.VTG_ATTRIBUTE_VERTEX, 3, gl.FLOAT, false, 0,0);
	gl.enableVertexAttribArray(WebGLMacros.VTG_ATTRIBUTE_VERTEX);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	//-	Create VBO
	
	gl.bindVertexArray(null);
	//-	Create VAO
	
	//-	Vertices, color , shader attributes , VAO , VBO
	///////////////////////////////////////////////////////////////////////////////
	
	//	Set clear color
	gl.clearColor(0.0,0.0,1.0,1.0)	//	Blue
	
	//	Initialize projection matrix
	orthographicProjectionMatrix = mat4.create();
}


function resize()
{
	if (true == g_bFullScreen)
	{
		g_canvas.width = window.innerWidth;
		g_canvas.height = window.innerHeight;
	}
	else
	{
		g_canvas.width = canvas_original_width;
		g_canvas.height = canvas_original_height;
	}
	
	//	Set the viewport to match
	gl.viewport(0, 0, g_canvas.width, g_canvas.height);
	
	//	Orthographic projection => left,right,bottom,top,near,far.
	/*if (g_canvas.width > g_canvas.height)
	{
		mat4.ortho(orthographicProjectionMatrix, -100.0 * (g_canvas.width / g_canvas.height), 100.0 * (g_canvas.width / g_canvas.height), -100.0, 100.0, -100.0, 100.0);
	}
	else
	{
		mat4.ortho(orthographicProjectionMatrix, -100.0, 100.0, -100.0 * (g_canvas.height / g_canvas.width), 100.0 * (g_canvas.height / g_canvas.width), -100.0, 100.0);
	}*/
	
	if (g_canvas.width <= g_canvas.height)
        mat4.ortho(orthographicProjectionMatrix, -100.0, 100.0, (-100.0 * (g_canvas.height / g_canvas.width)), (100.0 * (g_canvas.height / g_canvas.width)), -100.0, 100.0);
    else
        mat4.ortho(orthographicProjectionMatrix, (-100.0 * (g_canvas.width / g_canvas.height)), (100.0 * (g_canvas.width / g_canvas.height)), -100.0, 100.0, -100.0, 100.0);
        
}


function draw()
{
	//	Code
	gl.clear(gl.COLOR_BUFFER_BIT);
	
	gl.useProgram(shaderObjectProgram);
	
	var modelViewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var modelViewProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.multiply(modelViewProjectionMatrix, orthographicProjectionMatrix, modelViewMatrix);
	
	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);
	
	gl.bindVertexArray(vao);
	
	gl.drawArrays(gl.TRIANGLES, 0, 3);
	
	gl.bindVertexArray(null);
	
	gl.useProgram(null);
	
	//	Animation loop
	requestAnimationFrame(draw, g_canvas);	
}


function uninitialize()
{
	if (vao)
	{
		gl.delateVertexArray(vao);
		vao = null;
	}
	
	if (vbo)
	{
		gl.delateBuffer(vbo);
		vbo = null;
	}
	
	if (shaderObjectProgram)
	{
		if (shaderObjectVertex)
		{
			gl.detachShaderObject(shaderObjectProgram, shaderObjectVertex);
			gl.deleteShader(shaderObjectVertex);
			shaderObjectVertex = null;
		}
		
		if (shaderObjectFragment)
		{
			gl.detachShaderObject(shaderObjectProgram, shaderObjectFragment);
			gl.deleteShader(shaderObjectFragment);
			shaderObjectFragment = null;
		}
		
		gl.deleteProgram(shaderObjectProgram);
		shaderObjectProgram = null;
	}
}
