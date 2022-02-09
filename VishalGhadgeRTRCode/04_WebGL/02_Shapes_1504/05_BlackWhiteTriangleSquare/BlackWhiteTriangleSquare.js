// On load function
var g_canvas = null;
var g1 = null;	//	Webgl context

var g_bFullScreen = false;
var canvas_original_height;
var canvas_original_width;

const WebGLMacros= 
					{
						VTG_ATTRIBUTE_POSITION:0,
						VTG_ATTRIBUTE_COLOR:1,
						VTG_ATTRIBUTE_NORMAL:2,
						VTG_ATTRIBUTE_TEXTURE0:3,
					};

var shaderObjectVertex;
var shaderObjectFragment;
var shaderObjectProgram;

var vaoTriangle;
var vaoSquare;
var vboPosition;
var vboColor;

var mvpUniform;

var perspectiveProjectionMatrix;

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
	
	var vertexShaderFileId = document.getElementById('vs');
	var fc = vertexShaderFileId.firstChild;
	var shaderSource = "";	//	preampble string
	while (fc)
	{
		if (3 == fc.nodeType)
		{
			shaderSource += fc.textContent;
		}
		fc = fc.nextSibling;
	}
	
	//alert(shaderSource);

	shaderObjectVertex = gl.createShader(gl.VERTEX_SHADER);
	
	//gl.shaderSource(shaderObjectVertex, shaderSourceCodeVertex);
	gl.shaderSource(shaderObjectVertex, shaderSource);
	
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
	
	var fragmentShaderFileId = document.getElementById('fs');
	fc = fragmentShaderFileId.firstChild;
	var fragmentSource = "";	//	preampble string
	while (fc)
	{
		if (3 == fc.nodeType)
		{
			fragmentSource += fc.textContent;
		}
		fc = fc.nextSibling;
	}
	
	//alert(fragmentSource);
	
	shaderObjectFragment = gl.createShader(gl.FRAGMENT_SHADER);
	
	//gl.shaderSource(shaderObjectFragment, shaderSourceCodeFragment);
	gl.shaderSource(shaderObjectFragment, fragmentSource);
	
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
	gl.bindAttribLocation(shaderObjectProgram,WebGLMacros.VTG_ATTRIBUTE_POSITION,"vPosition");
	
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
												0.0,1.0,0.0,
												-1.0,-1.0,0,
												1.0, -1.0,0
											]);
	
	var squareVertices = new Float32Array([
												-1.0,1.0,0.0,
												-1.0,-1.0,0.0,
												1.0, -1.0,0.0,
												1.0, 1.0,0.0
											]);
	//+	Create VAO-Triangle 
	vaoTriangle = gl.createVertexArray();
	gl.bindVertexArray(vaoTriangle);
	
	////////////////////////////////////////////////////////////////////
	//+ Vertex position

	vboPosition = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vboPosition);
	gl.bufferData(gl.ARRAY_BUFFER, triangleVertices, gl.STATIC_DRAW);
	gl.vertexAttribPointer(WebGLMacros.VTG_ATTRIBUTE_POSITION, 3, gl.FLOAT, false, 0,0);
	gl.enableVertexAttribArray(WebGLMacros.VTG_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	
	//- Vertex position	
	////////////////////////////////////////////////////////////////////

	gl.bindVertexArray(null);
	//-	Create VAO-Triangle
	
	//+	Create VAO-Square
	vaoSquare = gl.createVertexArray();
	gl.bindVertexArray(vaoSquare);
	
	////////////////////////////////////////////////////////////////////
	//+ Vertex position

	vboPosition = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vboPosition);
	gl.bufferData(gl.ARRAY_BUFFER, squareVertices, gl.STATIC_DRAW);
	gl.vertexAttribPointer(WebGLMacros.VTG_ATTRIBUTE_POSITION, 3, gl.FLOAT, false, 0,0);
	gl.enableVertexAttribArray(WebGLMacros.VTG_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	
	//- Vertex position	
	////////////////////////////////////////////////////////////////////

	gl.bindVertexArray(null);
	//-	Create VAO-Square
	
	//-	Vertices, color , shader attributes , VAO , VBO
	///////////////////////////////////////////////////////////////////////////////
	
	//	Set clear color
	gl.clearColor(0.0,0.0,0.0,1.0)	//	Black
	
	//	Initialize projection matrix
	perspectiveProjectionMatrix = mat4.create();
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
	
	//	perspective(float fovy, float aspect, float n, float f)
	mat4.perspective(perspectiveProjectionMatrix, 45, parseFloat(g_canvas.width)/parseFloat(g_canvas.height), 0.1, 100.0);        
}


function draw()
{
	var modelViewMatrix;
	var modelViewProjectionMatrix;
	
	//	Code
	gl.clear(gl.COLOR_BUFFER_BIT);
	
	gl.useProgram(shaderObjectProgram);
	
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Triangle

	modelViewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	modelViewProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelViewMatrix, modelViewMatrix, [-2.0, 0.0, -5.0]);
	
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);
	
	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);
	
	gl.bindVertexArray(vaoTriangle);
	
	gl.drawArrays(gl.TRIANGLES, 0, 3);
	
	gl.bindVertexArray(null);
	//-	Draw Triangle	
	/////////////////////////////////////////////////////////////////////////////////////////
	
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Square

	modelViewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	modelViewProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelViewMatrix, modelViewMatrix, [2.0, 0.0, -5.0]);
	
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);
	
	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);
	
	gl.bindVertexArray(vaoSquare);
	
	gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
	
	gl.bindVertexArray(null);
	//-	Draw Triangle	
	/////////////////////////////////////////////////////////////////////////////////////////
		
	gl.useProgram(null);
	
	//	Animation loop
	requestAnimationFrame(draw, g_canvas);	
}


function uninitialize()
{
	if (vaoTriangle)
	{
		gl.delateVertexArray(vaoTriangle);
		vaoTriangle = null;
	}
	
	if (vaoSquare)
	{
		gl.delateVertexArray(vaoSquare);
		vaoSquare = null;
	}
	
	if (vboPosition)
	{
		gl.delateBuffer(vboPosition);
		vboPosition = null;
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
