// On load function
var g_canvas = null;
var g1 = null;	//	Webgl context

var g_bFullScreen = false;
var canvas_original_height;
var canvas_original_width;

var g_angleRed = 0.0;
var g_angleBlue = 0.0;
var g_angleGreen = 0.0;

const WebGLMacros= 
					{
						VTG_ATTRIBUTE_POSITION:0,
						VTG_ATTRIBUTE_COLOR:1,
						VTG_ATTRIBUTE_NORMAL:2,
						VTG_ATTRIBUTE_TEXTURE0:3,
					};

const Light= 
					{
						PER_VERTEX_PHONG:0,
						PER_FRAGMENT_PHONG:1,
					};

var shaderObjectVertexPerVertex;
var shaderObjectFragmentPerVertex;
var shaderObjectProgramPerVertex;

var shaderObjectVertexPerFragment;
var shaderObjectFragmentPerFragment;
var shaderObjectProgramPerFragment;

/////////////////////////////////////////////////////////////////
//+Uniforms.

var modelMatrixUniform = new Array(2);
var viewMatrixUniform = new Array(2);
var projectionMatrixUniform = new Array(2);

var rotationRMatrixUniform = new Array(2);
var rotationGMatrixUniform = new Array(2);
var rotationBMatrixUniform = new Array(2);

var laRUniform = new Array(2);
var ldRUniform = new Array(2);
var lsRUniform = new Array(2);
var lightPositionRUniform = new Array(2);

var laGUniform = new Array(2);
var ldGUniform = new Array(2);
var lsGUniform = new Array(2);
var lightPositionGUniform = new Array(2);

var laBUniform = new Array(2);
var ldBUniform = new Array(2);
var lsBUniform = new Array(2);
var lightPositionBUniform = new Array(2);

var kaUniform = new Array(2);
var kdUniform = new Array(2);
var ksUniform = new Array(2);
var materialShininessUniform = new Array(2);

var LKeyPressedUniform = new Array(2);

//-Uniforms.
/////////////////////////////////////////////////////////////////

//	Red Light 
var arrLightRAmbient = [0.0,0.0,0.0]; 
var arrLightRDiffuse = [1.0,0.0,0.0]; 
var arrLightRSpecular = [1.0,0.0,0.0]; 
var arrLightRPosition = [0.0,0.0,0.0, 0.0];	//	Give position runtime

//	Green Light 
var arrLightGAmbient = [0.0,0.0,0.0]; 
var arrLightGDiffuse = [0.0,1.0,0.0]; 
var arrLightGSpecular = [0.0,1.0,0.0]; 
var arrLightGPosition = [0.0,0.0,0.0, 0.0];	//	Give position runtime

//	Blue Light 
var arrLightBAmbient = [0.0,0.0,0.0]; 
var arrLightBDiffuse = [0.0,0.0,1.0]; 
var arrLightBSpecular = [0.0,0.0,1.0]; 
var arrLightBPosition = [0.0,0.0,0.0, 0.0];	//	Give position runtime
 
var arrMaterialAmbient = [0.0,0.0,0.0]; 
var arrMaterialDiffuse = [1.0,1.0,1.0]; 
var arrMaterialSpecular = [1.0,1.0,1.0]; 
var fMaterialShininess = 50.0;

var g_sphere = null;

var bLight = false;
var bAnimate = false;
var g_LightType = 1;	//	1 for Vertex and 2 for Fragment

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
			
		//case "S":
		case 'S':
		case 's':
			if (1 == g_LightType)
			{
				 g_LightType = 2;
			}
			else
			{
				 g_LightType = 1;
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
	
	//////////////////////////////////////////////////////////////////////////////////
	//+	Per vertex Phong Light.
	
	var vertexShaderFileIdPerVertex = document.getElementById('vsPerVertex');
	var fcPerVertex = vertexShaderFileIdPerVertex.firstChild;
	var shaderSourcePerVertex = "";	//	preampble string
	while (fcPerVertex)
	{
		if (3 == fcPerVertex.nodeType)
		{
			shaderSourcePerVertex += fcPerVertex.textContent;
		}
		fcPerVertex = fcPerVertex.nextSibling;
	}
	
	//alert(shaderSourcePerVertex);

	shaderObjectVertexPerVertex = gl.createShader(gl.VERTEX_SHADER);
	
	//gl.shaderSource(shaderObjectVertex, shaderSourceCodeVertex);
	gl.shaderSource(shaderObjectVertexPerVertex, shaderSourcePerVertex);
	
	gl.compileShader(shaderObjectVertexPerVertex);
	
	if (false == gl.getShaderParameter(shaderObjectVertexPerVertex, gl.COMPILE_STATUS))
	{
		var err = gl.getShaderInfoLog(shaderObjectVertexPerVertex);
		if (err.length > 0)
		{
			alert(err);
			uninitialize();
			//return;
		}
	}
	
	var fragmentShaderFileIdPerVertex = document.getElementById('fsPerVertex');
	fcPerVertex = fragmentShaderFileIdPerVertex.firstChild;
	var fragmentSourcePerVertex = "";	//	preampble string
	while (fcPerVertex)
	{
		if (3 == fcPerVertex.nodeType)
		{
			fragmentSourcePerVertex += fcPerVertex.textContent;
		}
		fcPerVertex = fcPerVertex.nextSibling;
	}
	
	//alert(fragmentSourcePerVertex);
	
	shaderObjectFragmentPerVertex = gl.createShader(gl.FRAGMENT_SHADER);
	
	//gl.shaderSource(shaderObjectFragment, shaderSourceCodeFragment);
	gl.shaderSource(shaderObjectFragmentPerVertex, fragmentSourcePerVertex);
	
	gl.compileShader(shaderObjectFragmentPerVertex);
	
	if (false == gl.getShaderParameter(shaderObjectFragmentPerVertex, gl.COMPILE_STATUS))
	{
		var err = gl.getShaderInfoLog(shaderObjectFragmentPerVertex);
		if (err.length > 0)
		{
			alert(err);
			uninitialize();
			//return;
		}
	}
	
	//	Create program.
	shaderObjectProgramPerVertex = gl.createProgram();
	
	gl.attachShader(shaderObjectProgramPerVertex, shaderObjectVertexPerVertex);
	gl.attachShader(shaderObjectProgramPerVertex, shaderObjectFragmentPerVertex);
	
	//	Pre-link binding of shader program object with vertex shader attributes.
	gl.bindAttribLocation(shaderObjectProgramPerVertex,WebGLMacros.VTG_ATTRIBUTE_POSITION,"vPosition");
	gl.bindAttribLocation(shaderObjectProgramPerVertex,WebGLMacros.VTG_ATTRIBUTE_NORMAL,"vNormal");
	
	//	Linking
	gl.linkProgram(shaderObjectProgramPerVertex);
	if (false == gl.getProgramParameter(shaderObjectProgramPerVertex, gl.LINK_STATUS))
	{
		var err = gl.getPRogramInfoLog(shaderObjectProgramPerVertex);
		if (err.length > 0)
		{
			alert(err);
			uninitialize();
			//return;
		}
	}
	//-	Per vertex Phong Light.	
	//////////////////////////////////////////////////////////////////////////////////
	
	//////////////////////////////////////////////////////////////////////////////////
	//+	Per Fragment Phong Light.
	
	var vertexShaderFileIdPerFragment = document.getElementById('vsPerFragment');
	var fcPerFragment = vertexShaderFileIdPerFragment.firstChild;
	var shaderSourcePerFragment = "";	//	preampble string
	while (fcPerFragment)
	{
		if (3 == fcPerFragment.nodeType)
		{
			shaderSourcePerFragment += fcPerFragment.textContent;
		}
		fcPerFragment = fcPerFragment.nextSibling;
	}
	
	//alert(shaderSourcePerFragment);

	shaderObjectVertexPerFragment = gl.createShader(gl.VERTEX_SHADER);
	
	//gl.shaderSource(shaderObjectVertex, shaderSourceCodeVertex);
	gl.shaderSource(shaderObjectVertexPerFragment, shaderSourcePerFragment);
	
	gl.compileShader(shaderObjectVertexPerFragment);
	
	if (false == gl.getShaderParameter(shaderObjectVertexPerFragment, gl.COMPILE_STATUS))
	{
		var err = gl.getShaderInfoLog(shaderObjectVertexPerFragment);
		if (err.length > 0)
		{
			alert(err);
			uninitialize();
			//return;
		}
	}
	
	var fragmentShaderFileIdPerFragment = document.getElementById('fsPerFragment');
	fcPerFragment = fragmentShaderFileIdPerFragment.firstChild;
	var fragmentSourcePerFragment = "";	//	preampble string
	while (fcPerFragment)
	{
		if (3 == fcPerFragment.nodeType)
		{
			fragmentSourcePerFragment += fcPerFragment.textContent;
		}
		fcPerFragment = fcPerFragment.nextSibling;
	}
	
	//alert(fragmentSourcePerFragment);
	
	shaderObjectFragmentPerFragment = gl.createShader(gl.FRAGMENT_SHADER);
	
	//gl.shaderSource(shaderObjectFragment, shaderSourceCodeFragment);
	gl.shaderSource(shaderObjectFragmentPerFragment, fragmentSourcePerFragment);
	
	gl.compileShader(shaderObjectFragmentPerFragment);
	
	if (false == gl.getShaderParameter(shaderObjectFragmentPerFragment, gl.COMPILE_STATUS))
	{
		var err = gl.getShaderInfoLog(shaderObjectFragmentPerFragment);
		if (err.length > 0)
		{
			alert(err);
			uninitialize();
			//return;
		}
	}
	
	//	Create program.
	shaderObjectProgramPerFragment = gl.createProgram();
	
	gl.attachShader(shaderObjectProgramPerFragment, shaderObjectVertexPerFragment);
	gl.attachShader(shaderObjectProgramPerFragment, shaderObjectFragmentPerFragment);
	
	//	Pre-link binding of shader program object with vertex shader attributes.
	gl.bindAttribLocation(shaderObjectProgramPerFragment,WebGLMacros.VTG_ATTRIBUTE_POSITION,"vPosition");
	gl.bindAttribLocation(shaderObjectProgramPerFragment,WebGLMacros.VTG_ATTRIBUTE_NORMAL,"vNormal");
	
	//	Linking
	gl.linkProgram(shaderObjectProgramPerFragment);
	if (false == gl.getProgramParameter(shaderObjectProgramPerFragment, gl.LINK_STATUS))
	{
		var err = gl.getPRogramInfoLog(shaderObjectProgramPerFragment);
		if (err.length > 0)
		{
			alert(err);
			uninitialize();
			//return;
		}
	}
	//-	Per Fragment Phong Light.	
	//////////////////////////////////////////////////////////////////////////////////

	//	Get MV uniform.
	
	modelMatrixUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_model_matrix");
	if (!modelMatrixUniform)
	{
		alert("failed to get var modelMatrixUniform;");
		uninitialize();
		return;
	}
	viewMatrixUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_view_matrix");
	if (!viewMatrixUniform)
	{
		alert("failed to get var viewMatrixUniform;");
		uninitialize();
		return;
	}
	projectionMatrixUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_projection_matrix");
	
	rotationRMatrixUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_rotation_matrixR");
	rotationGMatrixUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_rotation_matrixG");
	rotationBMatrixUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_rotation_matrixB");
	
	laRUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_LaR");
	ldRUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_LdR");
	lsRUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_LsR");
	lightPositionRUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_light_positionR");
	
	laGUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_LaG");
	ldGUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_LdG");
	lsGUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_LsG");	
	lightPositionGUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_light_positionG");
	
	laBUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_LaB");
	ldBUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_LdB");
	lsBUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_LsB");
	lightPositionBUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_light_positionB");
	
	kaUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_Ka");
	if (!kaUniform)
	{
		alert("failed to get kaUniform");
		uninitialize();
		return;
	}
	kdUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_Kd");
	if (!kdUniform)
	{
		alert("failed to get kdUniform");
		uninitialize();
		return;
	}
	ksUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_Ks");
	if (!ksUniform)
	{
		alert("failed to get ksUniform");
		uninitialize();
		return;
	}
	materialShininessUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_material_shininess");
	if (!materialShininessUniform)
	{
		alert("failed to get materialShininessUniform");
		uninitialize();
		return;
	}
	LKeyPressedUniform[Light.PER_VERTEX_PHONG] = gl.getUniformLocation(shaderObjectProgramPerVertex, "u_L_key_pressed");
	if (!LKeyPressedUniform)
	{
		alert("failed to get LKeyPressedUniform");
		uninitialize();
		return;
	}
	
	//	Fragment UNIFORM Locations.
	
	modelMatrixUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_model_matrix");
	if (!modelMatrixUniform)
	{
		alert("failed to get var modelMatrixUniform;");
		uninitialize();
		return;
	}
	viewMatrixUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_view_matrix");
	if (!viewMatrixUniform)
	{
		alert("failed to get var viewMatrixUniform;");
		uninitialize();
		return;
	}
	projectionMatrixUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_projection_matrix");
	
	rotationRMatrixUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_rotation_matrixR");
	rotationGMatrixUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_rotation_matrixG");
	rotationBMatrixUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_rotation_matrixB");
	
	laRUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_LaR");
	ldRUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_LdR");
	lsRUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_LsR");
	lightPositionRUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_light_positionR");
	
	laGUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_LaG");
	ldGUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_LdG");
	lsGUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_LsG");	
	lightPositionGUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_light_positionG");
	
	laBUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_LaB");
	ldBUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_LdB");
	lsBUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_LsB");
	lightPositionBUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_light_positionB");
	
	kaUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_Ka");
	if (!kaUniform)
	{
		alert("failed to get kaUniform");
		uninitialize();
		return;
	}
	kdUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_Kd");
	if (!kdUniform)
	{
		alert("failed to get kdUniform");
		uninitialize();
		return;
	}
	ksUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_Ks");
	if (!ksUniform)
	{
		alert("failed to get ksUniform");
		uninitialize();
		return;
	}
	materialShininessUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_material_shininess");
	if (!materialShininessUniform)
	{
		alert("failed to get materialShininessUniform");
		uninitialize();
		return;
	}
	LKeyPressedUniform[Light.PER_FRAGMENT_PHONG] = gl.getUniformLocation(shaderObjectProgramPerFragment, "u_L_key_pressed");
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
	
	var index;
	if (1 == g_LightType)
	{
		gl.useProgram(shaderObjectProgramPerVertex);
		index = Light.PER_VERTEX_PHONG;
	}
	else
	{
		gl.useProgram(shaderObjectProgramPerFragment);
		index = Light.PER_FRAGMENT_PHONG;
	}
	
	if (bLight)
	{
		gl.uniform1i(LKeyPressedUniform[index], 1);
		
		//	Setting light properties
		
		//	Red Light 
		gl.uniform3fv(laRUniform[index],arrLightRAmbient);
		gl.uniform3fv(ldRUniform[index],arrLightRDiffuse);
		gl.uniform3fv(lsRUniform[index],arrLightRSpecular);
		gl.uniform4fv(lightPositionRUniform[index], arrLightRPosition);
		
		//	Green Light 
		gl.uniform3fv(laGUniform[index],arrLightGAmbient);
		gl.uniform3fv(ldGUniform[index],arrLightGDiffuse);
		gl.uniform3fv(lsGUniform[index],arrLightGSpecular);
		gl.uniform4fv(lightPositionGUniform[index], arrLightGPosition);
		
		//	Blue Light 
		gl.uniform3fv(laBUniform[index],arrLightBAmbient);
		gl.uniform3fv(ldBUniform[index],arrLightBDiffuse);
		gl.uniform3fv(lsBUniform[index],arrLightBSpecular);
		gl.uniform4fv(lightPositionBUniform[index], arrLightBPosition);
		
		//	Setting material properties
		gl.uniform3fv(kaUniform[index],arrMaterialAmbient);
		gl.uniform3fv(kdUniform[index],arrMaterialDiffuse);
		gl.uniform3fv(ksUniform[index],arrMaterialSpecular);
		gl.uniform1f(materialShininessUniform[index],fMaterialShininess);		
	}
	else
	{
		gl.uniform1i(LKeyPressedUniform[index], 0);
	}

	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	var rotationRMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationGMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationBMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform[index], false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform[index], false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform[index], false, perspectiveProjectionMatrix);
	
	mat4.rotateX(rotationRMatrix, rotationRMatrix, g_angleRed);	//	X-Axis rotation.
	arrLightRPosition[1] = g_angleRed;
	gl.uniform4fv(lightPositionRUniform[index], arrLightRPosition);
	gl.uniformMatrix4fv(rotationRMatrixUniform[index], false, rotationRMatrix);
	
	mat4.rotateY(rotationGMatrix, rotationGMatrix, g_angleGreen);	//	Y-Axis rotation.
	arrLightRPosition[0] = g_angleGreen;
	gl.uniform4fv(lightPositionGUniform[index], arrLightGPosition);
	gl.uniformMatrix4fv(rotationGMatrixUniform[index], false, rotationGMatrix);
	
	mat4.rotateZ(rotationBMatrix, rotationBMatrix, g_angleBlue);	//	Z-Axis rotation.
	arrLightBPosition[0] = g_angleBlue;
	gl.uniform4fv(lightPositionBUniform[index], arrLightBPosition);
	gl.uniformMatrix4fv(rotationBMatrixUniform[index], false, rotationBMatrix);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
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
	var speed = 0.1;
	
	g_angleRed = g_angleRed + speed;
	if (g_angleRed >= 360)
	{
		g_angleRed = 0.0;
	}
	
	g_angleGreen = g_angleGreen + speed;
	if (g_angleGreen >= 360)
	{
		g_angleGreen = 0.0;
	}
	
	g_angleBlue = g_angleBlue + speed;
	if (g_angleBlue >= 360)
	{
		g_angleBlue = 0.0;
	}
}


function uninitialize()
{
	if (g_sphere)
	{
		g_sphere.deallocate();
		g_sphere = null;
	}
	
	if (shaderObjectProgramPerVertex)
	{
		if (shaderObjectVertexPerVertex)
		{
			gl.detachShaderObject(shaderObjectProgramPerVertex, shaderObjectVertexPerVertex);
			gl.deleteShader(shaderObjectVertexPerVertex);
			shaderObjectVertexPerVertex = null;
		}
		
		if (shaderObjectFragmentPerVertex)
		{
			gl.detachShaderObject(shaderObjectProgramPerVertex, shaderObjectFragmentPerVertex);
			gl.deleteShader(shaderObjectFragmentPerVertex);
			shaderObjectFragmentPerVertex = null;
		}
		
		gl.deleteProgram(shaderObjectProgramPerVertex);
		shaderObjectProgramPerVertex = null;
	}
	
	if (shaderObjectProgramPerFragment)
	{
		if (shaderObjectVertexPerFragment)
		{
			gl.detachShaderObject(shaderObjectProgramPerFragment, shaderObjectVertexPerFragment);
			gl.deleteShader(shaderObjectVertexPerFragment);
			shaderObjectVertexPerFragment = null;
		}
		
		if (shaderObjectFragmentPerFragment)
		{
			gl.detachShaderObject(shaderObjectProgramPerFragment, shaderObjectFragmentPerFragment);
			gl.deleteShader(shaderObjectFragmentPerFragment);
			shaderObjectFragmentPerFragment = null;
		}
		
		gl.deleteProgram(shaderObjectProgramPerFragment);
		shaderObjectProgramPerFragment = null;
	}
}
