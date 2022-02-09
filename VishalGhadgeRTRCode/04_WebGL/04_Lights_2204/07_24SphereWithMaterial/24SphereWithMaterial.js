// On load function
var g_canvas = null;
var g1 = null;	//	Webgl context

var g_bFullScreen = false;
var canvas_original_height;
var canvas_original_width;

var g_fAngleLight = 0.0;
var chAnimationAxis = 'x';

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
var rotationMatrixUniform;
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
 
///////////////////////////////////////////////////////////////////////////
//+Material
//
//	Materail 00
//
var g_arrMaterial00Ambient = [ 0.0215, 0.1745, 0.0215]//];
var g_arrMaterial00Diffuse = [ 0.07568, 0.61424, 0.07568]//, 1.0];
var g_arrMaterial00Specular = [ 0.633, 0.727811, 0.633]//, 1.0];
var g_Material00Shininess = 0.6 * 128.0;

//
//	Materail 10
//
var g_arrMaterial10Ambient = [ 0.135, 0.2225, 0.1575];
var g_arrMaterial10Diffuse = [ 0.54, 0.89, 0.63];
var g_arrMaterial10Specular = [ 0.316228, 0.316228, 0.316228];
var g_Material10Shininess = 0.1 * 128.0;

//
//	Materail 20
//
var g_arrMaterial20Ambient = [ 0.05375, 0.05, 0.06625];
var g_arrMaterial20Diffuse = [ 0.18275, 0.17, 0.22525];
var g_arrMaterial20Specular = [ 0.332741, 0.328634, 0.346435];
var g_Material20Shininess = 0.3 * 128.0;

//
//	Materail 30
//
var g_arrMaterial30Ambient = [ 0.25, 0.20725, 0.20725];
var g_arrMaterial30Diffuse = [ 1.0, 0.829, 0.829];
var g_arrMaterial30Specular = [ 0.296648, 0.296648, 0.296648];
var g_Material30Shininess = 0.088 * 128.0;

//
//	Materail 40
//
var g_arrMaterial40Ambient = [ 0.1745, 0.01175, 0.01175];
var g_arrMaterial40Diffuse = [ 0.61424, 0.04136, 0.04136];
var g_arrMaterial40Specular = [ 0.727811, 0.626959, 0.626959];
var g_Material40Shininess = 0.6 * 128.0;

//
//	Materail 50
//
var g_arrMaterial50Ambient = [ 0.1, 0.18725, 0.1745];
var g_arrMaterial50Diffuse = [ 0.396, 0.74151, 0.69102];
var g_arrMaterial50Specular = [ 0.297254, 0.30829, 0.306678];
var g_Material50Shininess = 0.1 * 128.0;

//
//	Materail 01
//
var g_arrMaterial01Ambient = [ 0.329412, 0.223529, 0.027451];
var g_arrMaterial01Diffuse = [ 0.780392, 0.568627, 0.113725];
var g_arrMaterial01Specular = [ 0.992157, 0.941176, 0.807843];
var g_Material01Shininess = 0.21794872 * 128.0;

//
//	Materail 11
//
var g_arrMaterial11Ambient = [ 0.2125, 0.1275, 0.054];
var g_arrMaterial11Diffuse = [ 0.714, 0.4284, 0.18144];
var g_arrMaterial11Specular = [ 0.393548, 0.271906, 0.166721];
var g_Material11Shininess = 0.2 * 128.0;

//
//	Materail 21
//
var g_arrMaterial21Ambient = [ 0.25, 0.25, 0.25];
var g_arrMaterial21Diffuse = [ 0.4, 0.4, 0.4];
var g_arrMaterial21Specular = [ 0.774597, 0.774597, 0.774597];
var g_Material21Shininess = 0.6 * 128.0;

//
//	Materail 31
//
var g_arrMaterial31Ambient = [ 0.19125, 0.0735, 0.0225];
var g_arrMaterial31Diffuse = [ 0.7038, 0.27048, 0.0828];
var g_arrMaterial31Specular = [ 0.256777, 0.137622, 0.296648];
var g_Material31Shininess = 0.1 * 128.0;

//
//	Materail 41
//
var g_arrMaterial41Ambient = [ 0.24725, 0.1995, 0.0745];
var g_arrMaterial41Diffuse = [ 0.75164, 0.60648, 0.22648];
var g_arrMaterial41Specular = [ 0.628281, 0.555802, 0.366065];
var g_Material41Shininess = 0.4 * 128.0;

//
//	Materail 51
//
var g_arrMaterial51Ambient = [ 0.19225, 0.19225, 0.19225];
var g_arrMaterial51Diffuse = [ 0.50754, 0.50754, 0.50754];
var g_arrMaterial51Specular = [ 0.508273, 0.508273, 0.508273];
var g_Material51Shininess = 0.4 * 128.0;

//
//	Materail 02
//
var g_arrMaterial02Ambient = [ 0.0, 0.0, 0.0];
var g_arrMaterial02Diffuse = [ 0.0, 0.0, 0.0];
var g_arrMaterial02Specular = [ 0.0, 0.0, 0.0];
var g_Material02Shininess = 0.25 * 128.0;

//
//	Materail 12
//
var g_arrMaterial12Ambient = [ 0.0, 0.1, 0.06];
var g_arrMaterial12Diffuse = [ 0.0, 0.50980392, 0.50980392];
var g_arrMaterial12Specular = [ 0.50980392, 0.50980392, 0.50980392];
var g_Material12Shininess = 0.25 * 128.0;

//
//	Materail 22
//
var g_arrMaterial22Ambient = [ 0.0, 0.0, 0.0];
var g_arrMaterial22Diffuse = [ 0.1, 0.35, 0.1];
var g_arrMaterial22Specular = [ 0.45, 0.45, 0.45];
var g_Material22Shininess = 0.25 * 128.0;

//
//	Materail 32
//
var g_arrMaterial32Ambient = [ 0.0, 0.0, 0.0];
var g_arrMaterial32Diffuse = [ 0.5, 0.0, 0.0];
var g_arrMaterial32Specular = [ 0.7, 0.6, 0.6];
var g_Material32Shininess = 0.25 * 128.0;

//
//	Materail 42
//
var g_arrMaterial42Ambient = [ 0.0, 0.0, 0.0];
var g_arrMaterial42Diffuse = [ 0.55, 0.55, 0.55];
var g_arrMaterial42Specular = [ 0.70, 0.70, 0.70];
var g_Material42Shininess = 0.25 * 128.0;

//
//	Materail 52
//
var g_arrMaterial52Ambient = [ 0.0, 0.0, 0.0];
var g_arrMaterial52Diffuse = [ 0.5, 0.5, 0.0];
var g_arrMaterial52Specular = [ 0.60, 0.60, 0.50];
var g_Material52Shininess = 0.25 * 128.0;

//
//	Materail 03
//
var g_arrMaterial03Ambient = [ 0.02, 0.02, 0.02];
var g_arrMaterial03Diffuse = [ 0.01, 0.01, 0.01];
var g_arrMaterial03Specular = [ 0.4, 0.4, 0.4];
var g_Material03Shininess = 0.078125 * 128.0;

//
//	Materail 13
//
var g_arrMaterial13Ambient = [ 0.0, 0.05, 0.05];
var g_arrMaterial13Diffuse = [ 0.4, 0.5, 0.5];
var g_arrMaterial13Specular = [ 0.04, 0.7, 0.7];
var g_Material13Shininess = 0.078125 * 128.0;

//
//	Materail 23
//
var g_arrMaterial23Ambient = [ 0.0, 0.05, 0.0];
var g_arrMaterial23Diffuse = [ 0.4, 0.5, 0.4];
var g_arrMaterial23Specular = [ 0.04, 0.7, 0.04];
var g_Material23Shininess = 0.078125 * 128.0;

//
//	Materail 33
//
var g_arrMaterial33Ambient = [ 0.05, 0.0, 0.0];
var g_arrMaterial33Diffuse = [ 0.5, 0.4, 0.4];
var g_arrMaterial33Specular = [ 0.7, 0.04, 0.04];
var g_Material33Shininess = 0.078125 * 128.0;

//
//	Materail 43
//
var g_arrMaterial43Ambient = [ 0.05, 0.05, 0.05];
var g_arrMaterial43Diffuse = [ 0.5, 0.5, 0.5];
var g_arrMaterial43Specular = [ 0.7, 0.7, 0.7];
var g_Material43Shininess = 0.78125 * 128.0;

//
//	Materail 53
//
var g_arrMaterial53Ambient = [ 0.05, 0.05, 0.0];
var g_arrMaterial53Diffuse = [ 0.5, 0.5, 0.4];
var g_arrMaterial53Specular = [ 0.7, 0.7, 0.04];
var g_Material53Shininess = 0.078125 * 128.0;

//-Material
///////////////////////////////////////////////////////////////////////////

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
		case 'x':
		case 'X':
		case 'y':
		case 'Y':
		case 'z':
		case 'Z':
			chAnimationAxis = event.key;
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
	rotationMatrixUniform = gl.getUniformLocation(shaderObjectProgram, "u_rotation_matrix");
	if (!rotationMatrixUniform)
	{
		alert("failed to get rotationMatrixUniform");
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
	mat4.perspective(perspectiveProjectionMatrix, 45, parseFloat(g_canvas.width/4)/parseFloat(g_canvas.height/6), 0.1, 100.0);        
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
	}
	else
	{
		gl.uniform1i(LKeyPressedUniform, 0);
	}

	var fHeightMulti;
	var fWidthMulti;
	
	//	First column,
	fHeightMulti = 0.05;
	fWidthMulti = 0.07;
	
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere50();
	
	fHeightMulti = fHeightMulti + 0.15;
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere40();
	
	fHeightMulti = fHeightMulti + 0.15;
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere30();
	
	fHeightMulti = fHeightMulti + 0.15;
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere20();
	
	fHeightMulti = fHeightMulti + 0.15;
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere10();
	
	fHeightMulti = fHeightMulti + 0.15;
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere00();
	
	//	Second column,
	fHeightMulti = 0.05;
	fWidthMulti = fWidthMulti + 0.20;
	
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere51();
	
	fHeightMulti = fHeightMulti + 0.15;
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere41();
	
	fHeightMulti = fHeightMulti + 0.15;
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere31();
	
	fHeightMulti = fHeightMulti + 0.15;
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere21();
	
	fHeightMulti = fHeightMulti + 0.15;
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere11();
	
	fHeightMulti = fHeightMulti + 0.15;
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere01();
	
	//	Third column,
	fHeightMulti = 0.05;
	fWidthMulti = fWidthMulti + 0.20;
	
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere52();
	
	fHeightMulti = fHeightMulti + 0.15;
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere42();
	
	fHeightMulti = fHeightMulti + 0.15;
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere32();
	
	fHeightMulti = fHeightMulti + 0.15;
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere22();
	
	fHeightMulti = fHeightMulti + 0.15;
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere12();
	
	fHeightMulti = fHeightMulti + 0.15;
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere02();
	
	//	Fourth column,
	fHeightMulti = 0.05;
	fWidthMulti = fWidthMulti + 0.20;
	
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere53();
	
	fHeightMulti = fHeightMulti + 0.15;
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere43();
	
	fHeightMulti = fHeightMulti + 0.15;
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere33();
	
	fHeightMulti = fHeightMulti + 0.15;
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere23();
	
	fHeightMulti = fHeightMulti + 0.15;
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere13();
	
	fHeightMulti = fHeightMulti + 0.15;
	gl.viewport((g_canvas.width * fWidthMulti), (g_canvas.height * fHeightMulti), g_canvas.width/4, g_canvas.height/6); 
	Sphere03();
		
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
	g_fAngleLight = g_fAngleLight + 0.1;
	if (g_fAngleLight >= 720.0)
		//if (g_fAngleLight >= 360)
	{
		//	Fixme: Search proper way to avoid hitch in light animation.
		g_fAngleLight = 360.0;
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////
//+First Column
function Sphere00()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial00Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial00Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial00Specular);
	gl.uniform1f(materialShininessUniform, g_Material00Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}

function Sphere10()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial10Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial10Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial10Specular);
	gl.uniform1f(materialShininessUniform, g_Material10Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}

function Sphere20()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial20Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial20Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial20Specular);
	gl.uniform1f(materialShininessUniform, g_Material20Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}

function Sphere30()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial30Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial30Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial30Specular);
	gl.uniform1f(materialShininessUniform, g_Material30Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}

function Sphere40()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial40Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial40Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial40Specular);
	gl.uniform1f(materialShininessUniform, g_Material40Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}

function Sphere50()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial50Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial50Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial50Specular);
	gl.uniform1f(materialShininessUniform, g_Material50Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}
//-First Column
/////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
//+Second Column
function Sphere01()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial01Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial01Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial01Specular);
	gl.uniform1f(materialShininessUniform, g_Material01Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}

function Sphere11()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial11Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial11Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial11Specular);
	gl.uniform1f(materialShininessUniform, g_Material11Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}

function Sphere21()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial21Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial21Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial21Specular);
	gl.uniform1f(materialShininessUniform, g_Material21Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}

function Sphere31()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial31Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial31Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial31Specular);
	gl.uniform1f(materialShininessUniform, g_Material31Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}

function Sphere41()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial41Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial41Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial41Specular);
	gl.uniform1f(materialShininessUniform, g_Material41Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}

function Sphere51()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial51Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial51Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial51Specular);
	gl.uniform1f(materialShininessUniform, g_Material51Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}
//-Second Column
/////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
//+Third Column
function Sphere02()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial02Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial02Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial02Specular);
	gl.uniform1f(materialShininessUniform, g_Material02Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}

function Sphere12()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial12Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial12Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial12Specular);
	gl.uniform1f(materialShininessUniform, g_Material12Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}

function Sphere22()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial22Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial22Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial22Specular);
	gl.uniform1f(materialShininessUniform, g_Material22Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}

function Sphere32()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial32Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial32Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial32Specular);
	gl.uniform1f(materialShininessUniform, g_Material32Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}

function Sphere42()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial42Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial42Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial42Specular);
	gl.uniform1f(materialShininessUniform, g_Material42Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}

function Sphere52()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial52Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial52Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial52Specular);
	gl.uniform1f(materialShininessUniform, g_Material52Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}
//-Third Column
/////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
//+Fourth Column
function Sphere03()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial03Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial03Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial03Specular);
	gl.uniform1f(materialShininessUniform, g_Material03Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}

function Sphere13()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial13Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial13Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial13Specular);
	gl.uniform1f(materialShininessUniform, g_Material13Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}

function Sphere23()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial23Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial23Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial23Specular);
	gl.uniform1f(materialShininessUniform, g_Material23Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}

function Sphere33()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial33Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial33Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial33Specular);
	gl.uniform1f(materialShininessUniform, g_Material33Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}

function Sphere43()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial43Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial43Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial43Specular);
	gl.uniform1f(materialShininessUniform, g_Material43Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}

function Sphere53()
{
	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Sphere
	
	var modelMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var viewMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var ProjectionMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	var rotationMatrix = mat4.create();	//	Matrix creation + set to Identity matrix.
	
	mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -6.0]);
	
	gl.uniformMatrix4fv(modelMatrixUniform, false, modelMatrix);
	gl.uniformMatrix4fv(viewMatrixUniform, false, viewMatrix);
	gl.uniformMatrix4fv(projectionMatrixUniform, false, perspectiveProjectionMatrix);
	
	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		mat4.rotateX(rotationMatrix, rotationMatrix, g_fAngleLight);		//	X-axis rotation
		arrLightPosition[1] = g_fAngleLight;
		arrLightPosition[0] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		mat4.rotateY(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Y-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		mat4.rotateZ(rotationMatrix, rotationMatrix, g_fAngleLight);		//	Z-axis rotation
		arrLightPosition[0] = g_fAngleLight;
		arrLightPosition[1] = arrLightPosition[2] = 0.0;
		gl.uniform4fv(lightPositionUniform, arrLightPosition);
		gl.uniformMatrix4fv(rotationMatrixUniform, false, rotationMatrix);
	}
	else
	{
		arrLightPosition[0] = arrLightPosition[1] = arrLightPosition[2] = 0.0;
		arrLightPosition[2] = 1.0;
	}
	
	gl.uniform3fv(kaUniform, g_arrMaterial53Ambient);
	gl.uniform3fv(kdUniform, g_arrMaterial53Diffuse);
	gl.uniform3fv(ksUniform, g_arrMaterial53Specular);
	gl.uniform1f(materialShininessUniform, g_Material53Shininess);
	
	g_sphere.draw();
	
	//-	Draw Sphere	
	/////////////////////////////////////////////////////////////////////////////////////////
}
//-Fourth Column
/////////////////////////////////////////////////////////////////////////////////////////////

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
