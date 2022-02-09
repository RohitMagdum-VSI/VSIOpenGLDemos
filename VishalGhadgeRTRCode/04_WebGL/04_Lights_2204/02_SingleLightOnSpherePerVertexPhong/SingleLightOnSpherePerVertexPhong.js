// On load function
var g_canvas = null;
var g1 = null;	//	Webgl context

var g_bFullScreen = false;
var canvas_original_height;
var canvas_original_width;

var g_angleCube = 0.0;

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

/////////////////////////////////////////////////////////////////
//+Uniforms.

var modelMatrixUniform;
var viewMatrixUniform;
var projectionMatrixUniform;
var laUniform;
var ldUniform;
var lsUniform;
var lightPositionUniform;
var kaUniform;
var kdUniform;
var ksUniform;
var materialShininessUniform;

var LKeyPressedUniform;

//-Uniforms.
/////////////////////////////////////////////////////////////////

var arrLightAmbient = [0.0,0.0,0.0]; 
var arrLightDiffuse = [1.0,1.0,1.0]; 
var arrLightSpecular = [1.0,1.0,1.0]; 
var arrLightPosition = [100.0,100.0,100.0, 1.0];
 
var arrMaterialAmbient = [0.0,0.0,0.0]; 
var arrMaterialDiffuse = [1.0,1.0,1.0]; 
var arrMaterialSpecular = [1.0,1.0,1.0]; 
var fMaterialShininess = 50.0;

var g_sphere = null;

var bLight = false;
var bAnimate = false;

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
	modelMatrixUniform = gl.getUniformLocation(shaderObjectProgram, "u_model_matrix");
	if (!modelMatrixUniform)
	{
		alert("failed to get var modelMatrixUniform;");
		uninitialize();
		return;
	}
	viewMatrixUniform = gl.getUniformLocation(shaderObjectProgram, "u_view_matrix");
	if (!viewMatrixUniform)
	{
		alert("failed to get var viewMatrixUniform;");
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
	laUniform = gl.getUniformLocation(shaderObjectProgram, "u_La");
	if (!laUniform)
	{
		alert("failed to get laUniform");
		uninitialize();
		return;
	}
	ldUniform = gl.getUniformLocation(shaderObjectProgram, "u_Ld");
	if (!ldUniform)
	{
		alert("failed to get ldUniform");
		uninitialize();
		return;
	}
	lsUniform = gl.getUniformLocation(shaderObjectProgram, "u_Ls");
	if (!lsUniform)
	{
		alert("failed to get lsUniform");
		uninitialize();
		return;
	}
	lightPositionUniform = gl.getUniformLocation(shaderObjectProgram, "u_light_position");
	if (!lightPositionUniform)
	{
		alert("failed to get lightPositionUniform");
		uninitialize();
		return;
	}
	kaUniform = gl.getUniformLocation(shaderObjectProgram, "u_Ka");
	if (!kaUniform)
	{
		alert("failed to get kaUniform");
		uninitialize();
		return;
	}
	kdUniform = gl.getUniformLocation(shaderObjectProgram, "u_Kd");
	if (!kdUniform)
	{
		alert("failed to get kdUniform");
		uninitialize();
		return;
	}
	ksUniform = gl.getUniformLocation(shaderObjectProgram, "u_Ks");
	if (!ksUniform)
	{
		alert("failed to get ksUniform");
		uninitialize();
		return;
	}
	materialShininessUniform = gl.getUniformLocation(shaderObjectProgram, "u_material_shininess");
	if (!materialShininessUniform)
	{
		alert("failed to get materialShininessUniform");
		uninitialize();
		return;
	}
	LKeyPressedUniform = gl.getUniformLocation(shaderObjectProgram, "u_L_key_pressed");
	if (!LKeyPressedUniform)
	{
		alert("failed to get LKeyPressedUniform");
		uninitialize();
		return;
	}
	
	///////////////////////////////////////////////////////////////////////////////
	//+	Vertices, color , shader attributes , VAO , VBO
	
	g_sphere = new Mesh();
	makeSphere(g_sphere, 2.0, 50,50);
	
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
	//gl.enable(GL_CULL_FACE);

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
	//	Code
	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
	
	gl.useProgram(shaderObjectProgram);
	
	if (bLight)
	{
		gl.uniform1i(LKeyPressedUniform, 1);
		
		//	Setting light properties
		gl.uniform3fv(laUniform,arrLightAmbient);
		gl.uniform3fv(ldUniform,arrLightDiffuse);
		gl.uniform3fv(lsUniform,arrLightSpecular);
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		
		//	Setting material properties
		gl.uniform3fv(kaUniform,arrMaterialAmbient);
		gl.uniform3fv(kdUniform,arrMaterialDiffuse);
		gl.uniform3fv(ksUniform,arrMaterialSpecular);
		gl.uniform1f(materialShininessUniform,fMaterialShininess);		
	}
	else
	{
		gl.uniform1i(LKeyPressedUniform, 0);
	}

	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
		
	gl.useProgram(null);
	
	if (bAnimate)
	{
		//update();
	}
	
	//	Animation loop
	requestAnimationFrame(draw, g_canvas);	
}


function update()
{
	
}


function uninitialize()
{
	if (g_sphere)
	{
		g_sphere.deallocate();
		g_sphere = null;
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
