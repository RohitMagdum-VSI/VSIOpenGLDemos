// On load function
var g_canvas = null;
var g1 = null;	//	Webgl context

var g_bFullScreen = false;
var canvas_original_height;
var canvas_original_width;

var START_ANIMATE_I=-2.5;
var START_ANIMATE_N=2.0;
var START_ANIMATE_D=0.0;
var START_ANIMATE_I2=-2.5;
var START_ANIMATE_A=2.5;
var START_ANIMATE_FLAG=-4.0;

var g_fAnimateI = START_ANIMATE_I;
var g_fAnimateN = START_ANIMATE_N;
var g_fAnimateD = START_ANIMATE_D;
var g_fAnimateI2 = START_ANIMATE_I2;
var g_fAnimateA = START_ANIMATE_A;
var g_fAnimateFlag = START_ANIMATE_FLAG;
var g_fAnimateFlagX = START_ANIMATE_FLAG;

var g_audio;

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

var vaoI;
var vaoN;
var vaoD;
var vaoFlagLines;
var vaoA;

var vboPositionI;
var vboColorI;
var vboPositionN;
var vboColorN;
var vboPositionD;
var vboColorD;
var vboPositionA;
var vboColorA;
var vboPositionFlagLines;
var vboColorFlagLines;

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
	//toggleFullscreen();
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
			
		//case "Animate":
		case 'A':
		case 'a':
			g_fAnimateI = START_ANIMATE_I;
			g_fAnimateN = START_ANIMATE_N;
			g_fAnimateD = START_ANIMATE_D;
			g_fAnimateI2 = START_ANIMATE_I2;
			g_fAnimateA = START_ANIMATE_A;
			g_fAnimateFlag = START_ANIMATE_FLAG;
			g_fAnimateFlagX = START_ANIMATE_FLAG;
			
			g_audio.play();
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
	
	g_audio = document.getElementById("myAudio");
	if (null == g_audio)
	{
		console.log("Failed to get Audio \n");
	}
	
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
	gl.bindAttribLocation(shaderObjectProgram,WebGLMacros.VTG_ATTRIBUTE_COLOR,"vColor");
	
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
	
	var IVertices = new Float32Array([
												0.0,1.0,0.0,
												0.0,-1.0,0.0,
											]);											
	
	var IColor = new Float32Array([
												1.0,0.0,0.0,
												0.0,1.0,0.0,
											]);
	
	var NVertices = new Float32Array([
												0.0,1.0,0.0,
												0.0,-1.0,0.0,
												
												0.0,1.0,0.0,
												0.5,-1.0,0.0,
												
												0.5,1.0,0.0,
												0.5,-1.0,0.0,
											]);
	
	var NColor = new Float32Array([
												1.0,0.0,0.0,
												0.0,1.0,0.0,
												
												1.0,0.0,0.0,
												0.0,1.0,0.0,
												
												1.0,0.0,0.0,
												0.0,1.0,0.0,
											]);
	
	var DVertices = new Float32Array([
												0.0,1.0,0.0,
												0.0,-1.0,0.0,
												
												0.5,1.0,0.0,
												-0.02,1.0,0.0,
												
												0.5,1.0,0.0,
												0.5,-1.0,0.0,
												
												0.5,-1.0,0.0,
												-0.02,-1.0,0.0,
											]);
	
	var DColor = new Float32Array([
												1.0,0.0,0.0,
												0.0,1.0,0.0,
												
												1.0,0.0,0.0,
												1.0,0.0,0.0,
												
												1.0,0.0,0.0,
												0.0,1.0,0.0,
												
												0.0,1.0,0.0,
												0.0,1.0,0.0,
											]);
	
	var AVertices = new Float32Array([
												0.0,1.0,0.0,
												-0.4,-1.0,0.0,
												
												0.0,1.0,0.0,
												0.4,-1.0,0.0,
											]);
	
	var AColor = new Float32Array([
												1.0,0.0,0.0,
												0.0,1.0,0.0,
												
												1.0,0.0,0.0,
												0.0,1.0,0.0,
											]);
	
	var FlagVertices = new Float32Array([
												-0.2, 0.005, 0.0,
												0.2, 0.005, 0.0,
												
												-0.2, 0.0, 0.0,
												0.2, 0.0, 0.0,
												
												-0.2, -0.005, 0.0,
												0.2, -0.005, 0.0,
											]);
	
	var FlagColor = new Float32Array([
												1.0,0.0,0.0,
												1.0,0.0,0.0,
												
												1.0,1.0,1.0,
												1.0,1.0,1.0,
												
												0.0,1.0,0.0,
												0.0,1.0,0.0,
											]);
											

	//+	Create VAO-I 
	vaoI = gl.createVertexArray();
	gl.bindVertexArray(vaoI);
	
	////////////////////////////////////////////////////////////////////
	//+ Vertex position

	vboPositionI = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vboPositionI);
	gl.bufferData(gl.ARRAY_BUFFER, IVertices, gl.STATIC_DRAW);
	gl.vertexAttribPointer(WebGLMacros.VTG_ATTRIBUTE_POSITION, 3, gl.FLOAT, false, 0,0);
	gl.enableVertexAttribArray(WebGLMacros.VTG_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	
	//- Vertex position	
	////////////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////////////
	//+ Vertex Color

	vboColorI = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vboColorI);
	gl.bufferData(gl.ARRAY_BUFFER, IColor, gl.STATIC_DRAW);
	gl.vertexAttribPointer(WebGLMacros.VTG_ATTRIBUTE_COLOR, 3, gl.FLOAT, false, 0,0);
	gl.enableVertexAttribArray(WebGLMacros.VTG_ATTRIBUTE_COLOR);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	
	//- Vertex Color	
	////////////////////////////////////////////////////////////////////

	gl.bindVertexArray(null);
	//-	Create VAO-I 
	
	//+	Create VAO-N
	vaoN = gl.createVertexArray();
	gl.bindVertexArray(vaoN);
	
	////////////////////////////////////////////////////////////////////
	//+ Vertex position

	vboPositionN = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vboPositionN);
	gl.bufferData(gl.ARRAY_BUFFER, NVertices, gl.STATIC_DRAW);
	gl.vertexAttribPointer(WebGLMacros.VTG_ATTRIBUTE_POSITION, 3, gl.FLOAT, false, 0,0);
	gl.enableVertexAttribArray(WebGLMacros.VTG_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	
	//- Vertex position	
	////////////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////////////
	//+ Vertex Color

	vboColorN = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vboColorN);
	gl.bufferData(gl.ARRAY_BUFFER, NColor, gl.STATIC_DRAW);
	gl.vertexAttribPointer(WebGLMacros.VTG_ATTRIBUTE_COLOR, 3, gl.FLOAT, false, 0,0);
	gl.enableVertexAttribArray(WebGLMacros.VTG_ATTRIBUTE_COLOR);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	
	//- Vertex Color	
	////////////////////////////////////////////////////////////////////

	gl.bindVertexArray(null);
	//-	Create VAO-N
	
	//+	Create VAO-D
	vaoD = gl.createVertexArray();
	gl.bindVertexArray(vaoD);
	
	////////////////////////////////////////////////////////////////////
	//+ Vertex position

	vboPositionD = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vboPositionD);
	gl.bufferData(gl.ARRAY_BUFFER, DVertices, gl.STATIC_DRAW);
	gl.vertexAttribPointer(WebGLMacros.VTG_ATTRIBUTE_POSITION, 3, gl.FLOAT, false, 0,0);
	gl.enableVertexAttribArray(WebGLMacros.VTG_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	
	//- Vertex position	
	////////////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////////////
	//+ Vertex Color

	vboColorD = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vboColorD);
	gl.bufferData(gl.ARRAY_BUFFER, DColor, gl.DYNAMIC_DRAW);
	gl.vertexAttribPointer(WebGLMacros.VTG_ATTRIBUTE_COLOR, 3, gl.FLOAT, false, 0,0);
	gl.enableVertexAttribArray(WebGLMacros.VTG_ATTRIBUTE_COLOR);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	
	//- Vertex Color	
	////////////////////////////////////////////////////////////////////

	gl.bindVertexArray(null);
	//-	Create VAO-D
	
	//+	Create VAO-A
	vaoA = gl.createVertexArray();
	gl.bindVertexArray(vaoA);
	
	////////////////////////////////////////////////////////////////////
	//+ Vertex position

	vboPositionA = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vboPositionA);
	gl.bufferData(gl.ARRAY_BUFFER, AVertices, gl.STATIC_DRAW);
	gl.vertexAttribPointer(WebGLMacros.VTG_ATTRIBUTE_POSITION, 3, gl.FLOAT, false, 0,0);
	gl.enableVertexAttribArray(WebGLMacros.VTG_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	
	//- Vertex position	
	////////////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////////////
	//+ Vertex Color

	vboColorA = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vboColorA);
	gl.bufferData(gl.ARRAY_BUFFER, AColor, gl.STATIC_DRAW);
	gl.vertexAttribPointer(WebGLMacros.VTG_ATTRIBUTE_COLOR, 3, gl.FLOAT, false, 0,0);
	gl.enableVertexAttribArray(WebGLMacros.VTG_ATTRIBUTE_COLOR);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	
	//- Vertex Color	
	////////////////////////////////////////////////////////////////////

	gl.bindVertexArray(null);
	//-	Create VAO-A
	
	//+	Create VAO-Flag Lines 
	vaoFlagLines = gl.createVertexArray();
	gl.bindVertexArray(vaoFlagLines);
	
	////////////////////////////////////////////////////////////////////
	//+ Vertex position

	vboPositionFlagLines = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vboPositionFlagLines);
	gl.bufferData(gl.ARRAY_BUFFER, FlagVertices, gl.DYNAMIC_DRAW);
	gl.vertexAttribPointer(WebGLMacros.VTG_ATTRIBUTE_POSITION, 3, gl.FLOAT, false, 0,0);
	gl.enableVertexAttribArray(WebGLMacros.VTG_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	
	//- Vertex position	
	////////////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////////////
	//+ Vertex Color

	vboColorFlagLines = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vboColorFlagLines);
	gl.bufferData(gl.ARRAY_BUFFER, FlagColor, gl.STATIC_DRAW);
	gl.vertexAttribPointer(WebGLMacros.VTG_ATTRIBUTE_COLOR, 3, gl.FLOAT, false, 0,0);
	gl.enableVertexAttribArray(WebGLMacros.VTG_ATTRIBUTE_COLOR);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	
	//- Vertex Color	
	////////////////////////////////////////////////////////////////////

	gl.bindVertexArray(null);
	//-	Create VAO-Flag LINES
	
	//-	Vertices, color , shader attributes , VAO , VBO
	///////////////////////////////////////////////////////////////////////////////
	
	//	Set clear color
	gl.clearColor(0.0,0.0,0.0,1.0)	//	Black
	
	//+	Change 2 For 3D
	gl.clearDepth(1.0);

	gl.enable(gl.DEPTH_TEST);

	gl.depthFunc(gl.LEQUAL);

	//
	//	Optional.
	//
	//gl.shadeModel(gl.SMOOTH);	//	Not define webgl 
	//gl.hint(gl.PERSPECTIVE_CORRECTION_HINT, gl.NICEST); // In webgl , Error Invalid hint 

	//
	//	We will always cull back faces for better performance.
	//	We will this in case of 3-D rotation/graphics.
	//
	//glEnable(GL_CULL_FACE);

	//-	Change 2 For 3D
	
	//	Initialize projection matrix
	perspectiveProjectionMatrix = mat4.create();
	
	if (null != g_audio)
	{
		g_audio.play();
	}
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
	//+	Draw I

	modelViewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	modelViewProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	//mat4.translate(modelViewMatrix, modelViewMatrix, [-2.0, 0.0, -2.5]);
	mat4.translate(modelViewMatrix, modelViewMatrix, [g_fAnimateI, 0.0, -2.5]);
	
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);
	
	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);
	
	drawAlphabet('I');
	
	//-	Draw I	
	/////////////////////////////////////////////////////////////////////////////////////////
	
	if (g_fAnimateN == START_ANIMATE_N)
	{
		update();
		requestAnimationFrame(draw, g_canvas);
		return;
	}
	
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw N

	modelViewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	modelViewProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	//mat4.translate(modelViewMatrix, modelViewMatrix, [-1.5, 0.0, -2.5]);
	mat4.translate(modelViewMatrix, modelViewMatrix, [-1.5, g_fAnimateN, -2.5]);
	
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);
	
	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);
	
	drawAlphabet('N');
	//-	Draw N	
	/////////////////////////////////////////////////////////////////////////////////////////
	
	if (g_fAnimateD == START_ANIMATE_D)
	{
		update();
		requestAnimationFrame(draw, g_canvas);
		return;
	}
	
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw D

	modelViewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	modelViewProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelViewMatrix, modelViewMatrix, [-0.5, 0.0, -2.5]);
	
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);
	
	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);
	
	drawAlphabet('D');
	//-	Draw D
	/////////////////////////////////////////////////////////////////////////////////////////
	
	if (g_fAnimateI2 == START_ANIMATE_I2)
	{
		update();
		requestAnimationFrame(draw, g_canvas);
		return;
	}
	
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw I

	modelViewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	modelViewProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	//mat4.translate(modelViewMatrix, modelViewMatrix, [0.5, 0.0, -2.5]);
	mat4.translate(modelViewMatrix, modelViewMatrix, [0.5, g_fAnimateI2, -2.5]);
	
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);
	
	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);
	
	drawAlphabet('I');
	//-	Draw I
	/////////////////////////////////////////////////////////////////////////////////////////
	
	if (g_fAnimateA == START_ANIMATE_A)
	{
		update();
		requestAnimationFrame(draw, g_canvas);
		return;
	}
	
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw A

	modelViewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	modelViewProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	//mat4.translate(modelViewMatrix, modelViewMatrix, [1.5, 0.0, -2.5]);
	mat4.translate(modelViewMatrix, modelViewMatrix, [g_fAnimateA, 0.0, -2.5]);
	
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);
	
	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);
	
	drawAlphabet('A');
	//-	Draw A
	/////////////////////////////////////////////////////////////////////////////////////////
	
	if (g_fAnimateFlag == START_ANIMATE_FLAG)
	{
		update();
		requestAnimationFrame(draw, g_canvas);
		return;
	}
	
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Flag Lines 

	modelViewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	modelViewProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	//mat4.translate(modelViewMatrix, modelViewMatrix, [1.5, 0.0, -2.5]);
	mat4.translate(modelViewMatrix, modelViewMatrix, [g_fAnimateFlag, 0.0, -2.5]);
	
	mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);
	
	gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);
	
	drawFlagLines();
	//-	Draw Flag Lines 
	/////////////////////////////////////////////////////////////////////////////////////////
		
	gl.useProgram(null);
	
	update();
	//	Animation loop
	requestAnimationFrame(draw, g_canvas);
}


function drawAlphabet(ch)
{
	switch(ch)
	{
		case 'I':
		case 'i':
			gl.bindVertexArray(vaoI);
			gl.drawArrays(gl.LINES, 0, 2);
			gl.bindVertexArray(null);
			
			break;
			
		case 'N':
		case 'n':
			gl.bindVertexArray(vaoN);
			gl.drawArrays(gl.LINES, 0, 6);
			gl.bindVertexArray(null);
			
			break;
			
		case 'D':
		case 'd':	
			var DColor = new Float32Array([
											g_fAnimateD,0.0,0.0,
											0.0,g_fAnimateD,0.0,
											
											g_fAnimateD,0.0,0.0,
											g_fAnimateD,0.0,0.0,
											
											g_fAnimateD,0.0,0.0,
											0.0,g_fAnimateD,0.0,
											
											0.0,g_fAnimateD,0.0,
											0.0,g_fAnimateD,0.0,
										]);
			gl.bindVertexArray(vaoD);
			
			gl.bindBuffer(gl.ARRAY_BUFFER, vboColorD);
			gl.bufferData(gl.ARRAY_BUFFER, DColor, gl.DYNAMIC_DRAW);
			gl.bindBuffer(gl.ARRAY_BUFFER, null);
			
			gl.drawArrays(gl.LINES, 0, 8);
			gl.bindVertexArray(null);
			
			break;
			
		case 'A':
		case 'a':
			gl.bindVertexArray(vaoA);
			gl.drawArrays(gl.LINES, 0, 4);
			gl.bindVertexArray(null);
			
			break;		
	}
}


function drawFlagLines()
{	
	var FlagVertices = new Float32Array([
												-0.2, 0.005, 0.0,
												0.2, 0.005, 0.0,
												
												-0.2, 0.0, 0.0,
												0.2, 0.0, 0.0,
												
												-0.2, -0.005, 0.0,
												0.2, -0.005, 0.0,
											]);
	FlagVertices[0] = FlagVertices[6] = FlagVertices[12] = g_fAnimateFlagX;										
	gl.bindVertexArray(vaoFlagLines);
	
	gl.bindBuffer(gl.ARRAY_BUFFER, vboPositionFlagLines);
	gl.bufferData(gl.ARRAY_BUFFER, FlagVertices, gl.DYNAMIC_DRAW);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	
	gl.drawArrays(gl.LINES, 0, 6);
	
	gl.bindVertexArray(null);
}


function update()
{
	var SPEED = 0.005;

	if (g_fAnimateI <= -2.0)
	{
		g_fAnimateI = g_fAnimateI + SPEED;
		return;
	}

	if (g_fAnimateN >= 0.0)
	{
		g_fAnimateN = g_fAnimateN - SPEED;
		return;
	}

	if (g_fAnimateD < 1.0)
	{
		g_fAnimateD = g_fAnimateD + SPEED;// (0.001f);
		return;
	}

	if (g_fAnimateI2 <= 0.0)
	{
		g_fAnimateI2 = g_fAnimateI2 + SPEED;
		return;
	}

	if (g_fAnimateA >= 1.5)
	{
		g_fAnimateA = g_fAnimateA - SPEED;
		return;
	}

	if (g_fAnimateFlag <= 1.48)//1.5)
	{
		g_fAnimateFlag = g_fAnimateFlag + (SPEED * 3);
		return;
	}

	if (g_fAnimateFlagX <= -0.2)
	{
		g_fAnimateFlagX = g_fAnimateFlagX + (SPEED * 2);
		return;
	}	
}


function uninitialize()
{
	if (vaoI)
	{
		gl.delateVertexArray(vaoI);
		vaoI = null;
	}
	
	if (vaoN)
	{
		gl.delateVertexArray(vaoN);
		vaoN = null;
	}
	
	if (vaoD)
	{
		gl.delateVertexArray(vaoD);
		vaoD = null;
	}
	
	if (vaoA)
	{
		gl.delateVertexArray(vaoA);
		vaoA = null;
	}
	
	if (vaoFlagLines)
	{
		gl.delateVertexArray(vaoFlagLines);
		vaoFlagLines = null;
	}
	
	if (vboPositionI)
	{
		gl.delateBuffer(vboPositionI);
		vboPositionI = null;
	}
	
	if (vboColorI)
	{
		gl.delateBuffer(vboColorI);
		vboColorI = null;
	}
	
	if (vboPositionN)
	{
		gl.delateBuffer(vboPositionN);
		vboPositionN = null;
	}
	
	if (vboColorN)
	{
		gl.delateBuffer(vboColorN);
		vboColorN = null;
	}
	
	if (vboPositionD)
	{
		gl.delateBuffer(vboPositionD);
		vboPositionD = null;
	}
	
	if (vboColorD)
	{
		gl.delateBuffer(vboColorD);
		vboColorD = null;
	}
	
	if (vboPositionA)
	{
		gl.delateBuffer(vboPositionA);
		vboPositionA = null;
	}
	
	if (vboColorA)
	{
		gl.delateBuffer(vboColorA);
		vboColorA = null;
	}
	
	if (vboPositionFlagLines)
	{
		gl.delateBuffer(vboPositionFlagLines);
		vboPositionFlagLines = null;
	}
	
	if (vboColorFlagLines)
	{
		gl.delateBuffer(vboColorFlagLines);
		vboColorFlagLines = null;
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
