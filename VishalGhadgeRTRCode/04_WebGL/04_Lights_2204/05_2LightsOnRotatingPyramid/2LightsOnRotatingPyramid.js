// On load function
var g_canvas = null;
var g1 = null;	//	Webgl context

var g_bFullScreen = false;
var canvas_original_height;
var canvas_original_width;

var g_anglePyramid = 0.0;

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

var vaoPyramid;
var vboPosition;
var vboNormal;

var modelViewMatrixUniform;
var projectionMatrixUniform;
var laLUniform;
var ldLUniform;
var lsLUniform;
var lightPositionLUniform;

var laRUniform;
var ldRUniform;
var lsRUniform;
var lightPositionRUniform;

var kaUniform;
var kdUniform;
var ksUniform;
var materialShininessUniform;

var LKeyPressedUniform;

var bLight = false;
var bAnimate = false;

var arrLightRAmbient = [0.0,0.0,0.0]; 
var arrLightRDiffuse = [1.0,0.0,0.0]; 
var arrLightRSpecular = [1.0,0.0,0.0]; 
var arrLightRPosition = [2.0,1.0,1.0, 0.0];

var arrLightLAmbient = [0.0,0.0,0.0]; 
var arrLightLDiffuse = [0.0,0.0,1.0]; 
var arrLightLSpecular = [0.0,0.0,1.0]; 
var arrLightLPosition = [-2.0,1.0,1.0, 1.0];

var arrMaterialAmbient = [0.0,0.0,0.0]; 
var arrMaterialDiffuse = [1.0,1.0,1.0]; 
var arrMaterialSpecular = [1.0,1.0,1.0]; 
var materialShininess = 50.0;

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
			
		//case "L":
		case 'L':
		case 'l':
			if (false == bLight)
			{
				bLight = true;
			}
			else
			{
				bLight = false;
			}
			break;
			
		//case "A":
		case 'A':
		case 'a':
			if (false == bAnimate)
			{
				bAnimate = true;
			}
			else
			{
				bAnimate = false;
			}
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
	gl.bindAttribLocation(shaderObjectProgram,WebGLMacros.VTG_ATTRIBUTE_NORMAL,"vNormal");
	
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
	
	//	Get MV uniform.
	modelViewMatrixUniform = gl.getUniformLocation(shaderObjectProgram, "u_model_view_matrix");
	if (!modelViewMatrixUniform)
	{
		alert("failed to get modelViewMatrixUniform");
		uninitialize();
		return;
	}
	projectionMatrixUniform = gl.getUniformLocation(shaderObjectProgram, "u_projection_matrix");
	if (!projectionMatrixUniform)
	{
		alert("failed to get projectionMatrixUniform");
		uninitialize();
		return;
	}
	
	laLUniform = gl.getUniformLocation(shaderObjectProgram, "u_LaL");
	ldLUniform = gl.getUniformLocation(shaderObjectProgram, "u_LdL");
	lsLUniform = gl.getUniformLocation(shaderObjectProgram, "u_LsL");
	lightPositionLUniform = gl.getUniformLocation(shaderObjectProgram, "u_light_positionL");
	
	laRUniform = gl.getUniformLocation(shaderObjectProgram, "u_LaR");
	ldRUniform = gl.getUniformLocation(shaderObjectProgram, "u_LdR");
	lsRUniform = gl.getUniformLocation(shaderObjectProgram, "u_LsR");
	lightPositionRUniform = gl.getUniformLocation(shaderObjectProgram, "u_light_positionR");
	
	kaUniform = gl.getUniformLocation(shaderObjectProgram, "u_Ka");
	kdUniform = gl.getUniformLocation(shaderObjectProgram, "u_Kd");
	ksUniform = gl.getUniformLocation(shaderObjectProgram, "u_Ks");
	materialShininessUniform = gl.getUniformLocation(shaderObjectProgram, "u_material_shininess");
	
	LKeyPressedUniform = gl.getUniformLocation(shaderObjectProgram, "u_L_key_pressed");
	if (!LKeyPressedUniform)
	{
		alert("failed to get LKeyPressedUniform");
		uninitialize();
		return;
	}
	
	
	///////////////////////////////////////////////////////////////////////////////
	//+	Vertices, color , shader attributes , VAO , VBO
	
	var pyramidVertices = new Float32Array([
											//	Front face
											0.0, 1.0, 0.0,	//	apex
											-1.0, -1.0, 1.0,	//	left_bottom
											1.0, -1.0, 1.0,	//	right_bottom
											//	Right face
											0.0, 1.0, 0.0,	//	apex
											1.0, -1.0, 1.0,	//	left_bottom
											1.0, -1.0, -1.0,	//	right_bottom
											//	Back face
											0.0, 1.0, 0.0,	//	apex
											1.0, -1.0, -1.0,	//	left_bottom
											-1.0, -1.0, -1.0,	//	right_bottom
											//	Left face
											0.0, 1.0, 0.0,	//	apex
											-1.0, -1.0, -1.0,	//	left_bottom
											-1.0, -1.0, 1.0,	//	right_bottom	
											]);
    
    var pyramidNormals=new Float32Array([                                      
										//	Front face
										0.0, 0.447214, 0.894427,	//	apex
										0.0, 0.447214, 0.894427,	//	left_bottom
										0.0, 0.447214, 0.894427,	//	right_bottom

										//	Right face
										0.894427, 0.447214, 0.0,	//	apex
										0.894427, 0.447214, 0.0,	//	left_bottom
										0.894427, 0.447214, 0.0,	//	right_bottom

										//	Back face
										0.0, 0.447214, -0.894427,	//	apex
										0.0, 0.447214, -0.894427,	//	left_bottom
										0.0, 0.447214, -0.894427,	//	right_bottom

										//	Left face
										-0.894427, 0.447214, 0.0,	//	apex
										-0.894427, 0.447214, 0.0,	//	left_bottom
										-0.894427, 0.447214, 0.0,	//	right_bottom
                                      ]);
	
	//+	Create VAO-Square
	vaoPyramid = gl.createVertexArray();
	gl.bindVertexArray(vaoPyramid);
	
	////////////////////////////////////////////////////////////////////
	//+ Vertex position

	vboPosition = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vboPosition);
	gl.bufferData(gl.ARRAY_BUFFER, pyramidVertices, gl.STATIC_DRAW);
	gl.vertexAttribPointer(WebGLMacros.VTG_ATTRIBUTE_POSITION, 3, gl.FLOAT, false, 0,0);
	gl.enableVertexAttribArray(WebGLMacros.VTG_ATTRIBUTE_POSITION);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	
	//- Vertex position	
	////////////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////////////
	//+ Vertex Normals

	vboNormal = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vboNormal);
	gl.bufferData(gl.ARRAY_BUFFER, pyramidNormals, gl.STATIC_DRAW);
	gl.vertexAttribPointer(WebGLMacros.VTG_ATTRIBUTE_NORMAL, 3, gl.FLOAT, false, 0,0);
	gl.enableVertexAttribArray(WebGLMacros.VTG_ATTRIBUTE_NORMAL);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	
	//- Vertex Normals	
	////////////////////////////////////////////////////////////////////

	gl.bindVertexArray(null);
	//-	Create VAO-Square
	
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
	var ProjectionMatrix;
	
	//	Code
	gl.clear(gl.COLOR_BUFFER_BIT);
	
	gl.useProgram(shaderObjectProgram);
	
	if (bLight)
	{
		gl.uniform1i(LKeyPressedUniform, 1);
		
		gl.uniform3fv(laRUniform,arrLightRAmbient);
		gl.uniform3fv(ldRUniform,arrLightRDiffuse);
		gl.uniform3fv(lsRUniform,arrLightRSpecular);
		gl.uniform4fv(lightPositionRUniform,arrLightRPosition);
		
		gl.uniform3fv(laLUniform,arrLightLAmbient);
		gl.uniform3fv(ldLUniform,arrLightLDiffuse);
		gl.uniform3fv(lsLUniform,arrLightLSpecular);
		gl.uniform4fv(lightPositionLUniform,arrLightLPosition);
		
		gl.uniform3fv(kaUniform,arrMaterialAmbient);
		gl.uniform3fv(kdUniform,arrMaterialDiffuse);
		gl.uniform3fv(ksUniform,arrMaterialSpecular);
		gl.uniform1f(materialShininessUniform,materialShininess);
	}
	else
	{
		gl.uniform1i(LKeyPressedUniform, 0);
	}
	
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Cube

	modelViewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelViewMatrix, modelViewMatrix, [0.0, 0.0, -4.0]);
	
	mat4.rotateY(modelViewMatrix, modelViewMatrix, g_anglePyramid);	
	
	gl.uniformMatrix4fv(modelViewMatrixUniform, false, modelViewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	gl.bindVertexArray(vaoPyramid);
	
	gl.drawArrays(gl.TRIANGLES, 0, 12);
	
	gl.bindVertexArray(null);
	//-	Draw Cube	
	/////////////////////////////////////////////////////////////////////////////////////////
		
	gl.useProgram(null);
	
	if (bAnimate)
	{
		update();
	}
	
	//	Animation loop
	requestAnimationFrame(draw, g_canvas);	
}


function update()
{
	var speed = 0.03;
	
	g_anglePyramid = g_anglePyramid + speed;
	if (g_anglePyramid > 360)
	{
		g_anglePyramid = 0;
	}	
}


function uninitialize()
{
	if (vaoPyramid)
	{
		gl.delateVertexArray(vaoPyramid);
		vaoPyramid = null;
	}
	
	if (vboPosition)
	{
		gl.delateBuffer(vboPosition);
		vboPosition = null;
	}
	
	if (vboNormal)
	{
		gl.delateBuffer(vboNormal);
		vboNormal = null;
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
