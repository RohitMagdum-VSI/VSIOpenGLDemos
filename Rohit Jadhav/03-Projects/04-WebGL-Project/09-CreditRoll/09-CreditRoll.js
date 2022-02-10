var canvas;
var gl;


var vertexShaderObject_rvg_cr;
var fragmentShaderObject_rvg_cr;
var shaderProgramObject_rvg_cr;

var credTexture1;
var credTexture2;
var credTexture3;
var credTexture4;
var credTexture5;
var credTexture6;
var credTexture7;
var credTexture8;
var credTexture9;
var credTexture10;

var vao_square_rvg;
var vbo_square_position_rvg;
var vbo_square_texcoord_rvg;

var textureSamplerUniform_cr;
var blendAlphaUniform_cr;
var mvpUniform_rvg_cr;



var perspectiveProjectionMatrix_rvg_cr;


function initialize_CreditRoll(){

	// ****************** Vertex Shader ******************
	//Create Shader
	vertexShaderObject_rvg_cr = gl.createShader(gl.VERTEX_SHADER);

	//Provide Source Code to Shader
	var vertexShaderSource_rvg =
		"#version 300 es" +
		"\n" +
		"in vec4 vPosition;" +
		"in vec2 vTexcoord;" +
		"out vec2 out_Texcoord;" +
		"uniform mat4 u_mvpMatrix;" +
		"void main(void)" +
		"{"	+
			"gl_Position = u_mvpMatrix * vPosition;" +
			"out_Texcoord = vTexcoord;" +
		"}";

	//Feed to Shader Object
	gl.shaderSource(vertexShaderObject_rvg_cr, vertexShaderSource_rvg);

	//Compile Shader
	gl.compileShader(vertexShaderObject_rvg_cr);

	//Shader Compilation Error Checking
	if (gl.getShaderParameter(vertexShaderObject_rvg_cr, gl.COMPILE_STATUS) == false)
	{
		var error_rvg = gl.getShaderInfoLog(vertexShaderObject_rvg_cr);
		if (error_rvg.length > 0)
		{
			alert("09-CreditRoll : VS -> ",error_rvg);
			uninitialize();
		}
	}
	else
	{
		console.log("Vertex Shader Compilation is done Successfully.");
	}

	// ****************** Fragment Shader ******************
	//Create Shader
	fragmentShaderObject_rvg_cr = gl.createShader(gl.FRAGMENT_SHADER);

	//Provide Source Code to Shader
	var fragmentShaderSource_rvg =
		"#version 300 es" +
		"\n" +
		"precision highp float;" +
		"in vec2 out_Texcoord;" +
		"uniform float u_blendAlpha;" +
		"uniform highp sampler2D u_texture_sampler;" +
		"out vec4 fragColor;" +
		"void main(void)" +
		"{"	+
		"vec4 texColor = texture(u_texture_sampler, out_Texcoord);" +
		"if(texColor.a < 0.1)" +
		"	discard;  " +
		"fragColor =  vec4(1.0, 1.0, 1.0, u_blendAlpha);" +
		"fragColor = fragColor * texColor;" +
		"}";

	//Feed to Shader Object
	gl.shaderSource(fragmentShaderObject_rvg_cr, fragmentShaderSource_rvg);

	//Compile Shader
	gl.compileShader(fragmentShaderObject_rvg_cr);

	//Shader Compilation Error Checking
	if (gl.getShaderParameter(fragmentShaderObject_rvg_cr, gl.COMPILE_STATUS) == false)
	{
		var error_rvg = gl.getShaderInfoLog(fragmentShaderObject_rvg_cr);
		if (error_rvg.length > 0)
		{
			alert("09-CreditRoll : FS -> ",error_rvg);
			uninitialize();
		}
	}
	else
	{
		console.log("Fragment Shader Compilation is done Successfully.");
	}

	//Shader Linking Code
	//1)Create the Shader Program which capable of linking Shader.
	shaderProgramObject_rvg_cr = gl.createProgram();

	//2)Attach whichever Shaders you have to this Shader Program Object.
	gl.attachShader(shaderProgramObject_rvg_cr, vertexShaderObject_rvg_cr);
	gl.attachShader(shaderProgramObject_rvg_cr, fragmentShaderObject_rvg_cr);

	//3)(Pre-Linking) Bind with Shader Program Object with Attributes
	gl.bindAttribLocation(shaderProgramObject_rvg_cr, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
	gl.bindAttribLocation(shaderProgramObject_rvg_cr, WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0, "vTexcoord");

	//4)Then, Link all those attached Shader at once.
	gl.linkProgram(shaderProgramObject_rvg_cr);

	//Shader Linkage Error Checking
	if (!gl.getProgramParameter(shaderProgramObject_rvg_cr, gl.LINK_STATUS))
	{
		var error_rvg = gl.getProgramInfoLog(shaderProgramObject_rvg_cr);
		if (error_rvg.length > 0)
		{
			alert("09-CreditRoll : SP -> ",error_rvg);
			uninitialize();
		}
	}
	else
	{
		console.log("Program Shader Object Compilation is done Successfully.");
	}

	//Get MVP Uniform Location
	mvpUniform_rvg_cr = gl.getUniformLocation(shaderProgramObject_rvg_cr, "u_mvpMatrix");
	textureSamplerUniform_cr = gl.getUniformLocation(shaderProgramObject_rvg_cr, "u_texture_sampler");
	blendAlphaUniform_cr = gl.getUniformLocation(shaderProgramObject_rvg_cr, "u_blendAlpha");

	//Vertices Array Declaration
	var cubeVertices_rvg = new Float32Array([
												3.0, 1.7, 0.0,		//Top Right
												-3.0, 1.7, 0.0,		//Top Left
												-3.0, -1.7, 0.0,	//Bottom Left
												3.0, -1.7, 0.0		//Bottom Right	
			]);

	var cubeTexcoords_rvg = new Float32Array([
												1.0, 1.0,		//Front
												0.0, 1.0,		//Back
												0.0, 0.0,		//Left
												1.0, 0.0		//Right
					]);


	//Create Cassette to use Buffer Array in Display()
	vao_square_rvg = gl.createVertexArray();
	gl.bindVertexArray(vao_square_rvg);
			/////////////////////////////// FOR VERTEX ///////////////////////////////
			//Generate Buffer Array
			vbo_square_position_rvg = gl.createBuffer();

			//Bind Buffer Array
			gl.bindBuffer(gl.ARRAY_BUFFER, vbo_square_position_rvg);

			//Fill Data into Buffer Array
			gl.bufferData(gl.ARRAY_BUFFER, cubeVertices_rvg, gl.STATIC_DRAW);

			//Tell him How to Read Data from Buffer Array
			gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION, 3, gl.FLOAT, false, 0, 0);

			//Enable Attribute Pointer
			gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);

			//Unbind Buffer Array
			gl.bindBuffer(gl.ARRAY_BUFFER, null);

			/////////////////////////////// FOR TEXCOORD ///////////////////////////////
			//Generate Buffer Array
			vbo_square_texcoord_rvg = gl.createBuffer();

			//Bind Buffer Array
			gl.bindBuffer(gl.ARRAY_BUFFER, vbo_square_texcoord_rvg);

			//Fill Data into Buffer Array
			gl.bufferData(gl.ARRAY_BUFFER, cubeTexcoords_rvg, gl.STATIC_DRAW);

			//Tell him How to Read Data from Buffer Array
			gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0, 2, gl.FLOAT, false, 0, 0);

			//Enable Attribute Pointer
			gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0);

			//Unbind Buffer Array
			gl.bindBuffer(gl.ARRAY_BUFFER, null);

	//Stop Creating Cassette
	gl.bindVertexArray(null);


	

	//Set Texture
	credTexture1 = loadTexture("09-CreditRoll/Cred1.PNG");
	credTexture2 = loadTexture("09-CreditRoll/Cred2.PNG");
	credTexture3 = loadTexture("09-CreditRoll/Cred3.PNG");
	credTexture4 = loadTexture("09-CreditRoll/Cred4.PNG");
	credTexture5 = loadTexture("09-CreditRoll/Cred5.PNG");
	credTexture6 = loadTexture("09-CreditRoll/Cred6.PNG");
	credTexture7 = loadTexture("09-CreditRoll/Cred7.PNG");
	credTexture8 = loadTexture("09-CreditRoll/Cred8.PNG");
	credTexture9 = loadTexture("09-CreditRoll/Cred9.PNG");
	credTexture10 = loadTexture("09-CreditRoll/Cred10.PNG");



	perspectiveProjectionMatrix_rvg_cr = mat4.create();

	mat4.perspective(perspectiveProjectionMatrix_rvg_cr,
					45.0,
					parseFloat(canvas.width) / parseFloat(canvas.height),
					0.1,
					100.0);


	console.log("09-CreditRoll : initialize_CreditRoll() done");
}


function loadTexture(src)
{
	var credTexture = gl.createTexture();
	credTexture.image = new Image();
	credTexture.image.src = src;
	credTexture.image.onload = function()
	{
		gl.bindTexture(gl.TEXTURE_2D, credTexture);
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);

		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, credTexture.image);
		gl.generateMipmap(gl.TEXTURE_2D);
		gl.bindTexture(gl.TEXTURE_2D, null);
	}
	return credTexture;
}



function uninit()
{

	if (vao_square_rvg)
	{
		gl.deleteVertexArray(vao_square_rvg);
		vao_square_rvg = null;
	}

	if (vbo_square_position_rvg)
	{
		gl.deleteBuffer(vbo_square_position_rvg);
		vbo_square_position_rvg = null;
	}

	if (vbo_square_texcoord_rvg)
	{
		gl.deleteBuffer(vbo_square_texcoord_rvg);
		vbo_square_texcoord_rvg = null;
	}

	if (credTexture1)
	{
		gl.deleteTexture(credTexture1);
		credTexture1 = 0;
	}

	if (credTexture2)
	{
		gl.deleteTexture(credTexture2);
		credTexture2 = 0;
	}

	if (credTexture3)
	{
		gl.deleteTexture(credTexture3);
		credTexture3 = 0;
	}

	if (credTexture4)
	{
		gl.deleteTexture(credTexture4);
		credTexture4 = 0;
	}

	if (credTexture5)
	{
		gl.deleteTexture(credTexture5);
		credTexture5 = 0;
	}

	if (credTexture6)
	{
		gl.deleteTexture(credTexture6);
		credTexture6 = 0;
	}

	if (credTexture7)
	{
		gl.deleteTexture(credTexture7);
		credTexture7 = 0;
	}

	if (credTexture8)
	{
		gl.deleteTexture(credTexture8);
		credTexture8 = 0;
	}

	if (credTexture9)
	{
		gl.deleteTexture(credTexture9);
		credTexture9 = 0;
	}
}


function uninitialize_CreditRoll()
{
	//Code
	//////////////////////////////////////////////////////////////////////////////////////////////

	uninit();

	//////////////////////////////////////////////////////////////////////////////////////////////

	if (shaderProgramObject_rvg_cr)
	{
		if (fragmentShaderObject_rvg_cr)
		{
			gl.detachShader(shaderProgramObject_rvg_cr, fragmentShaderObject_rvg_cr);
			gl.deleteShader(fragmentShaderObject_rvg_cr);
			fragmentShaderObject_rvg_cr = null;
		}

		if (vertexShaderObject_rvg_cr)
		{
			gl.detachShader(shaderProgramObject_rvg_cr, vertexShaderObject_rvg_cr);
			gl.deleteShader(vertexShaderObject_rvg_cr);
			vertexShaderObject_rvg_cr = null;
		}

		gl.deleteProgram(shaderProgramObject_rvg_cr);
		shaderProgramObject_rvg_cr = null;
	}
}


function display_CreditRoll(frame, alphaValue)
{


	//Enabling Blend
	gl.enable(gl.BLEND);
	gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
	
	//Use Shader Program Object
	gl.useProgram(shaderProgramObject_rvg_cr);

	//Create Matrices
	var modelViewMatrix = mat4.create();
	var modelViewProjectionMatrix = mat4.create();
	var translateMatrix = mat4.create();

	//Translation fro PYRAMID
	mat4.translate(translateMatrix, translateMatrix, [0.0, 0.0, -4.0]);

	//Multiply Matrices
	modelViewMatrix = translateMatrix;

	mat4.multiply(modelViewProjectionMatrix,
					perspectiveProjectionMatrix_rvg_cr,
					modelViewMatrix);

	//Set Uniform
	gl.uniformMatrix4fv(mvpUniform_rvg_cr, false, modelViewProjectionMatrix);
	gl.uniform1f(blendAlphaUniform_cr, alphaValue);



	gl.activeTexture(gl.TEXTURE0);
	switch(frame){

		case 1:
			gl.bindTexture(gl.TEXTURE_2D, credTexture1);
			break;

		case 2:
			gl.bindTexture(gl.TEXTURE_2D, credTexture2);
			break;


		case 3:
			gl.bindTexture(gl.TEXTURE_2D, credTexture3);
			break;


		case 4:
			gl.bindTexture(gl.TEXTURE_2D, credTexture4);
			break;

		case 5:
			gl.bindTexture(gl.TEXTURE_2D, credTexture5);
			break;

		case 6:
			gl.bindTexture(gl.TEXTURE_2D, credTexture6);
			break;

		case 7:
			gl.bindTexture(gl.TEXTURE_2D, credTexture7);
			break;


		case 8:
			gl.bindTexture(gl.TEXTURE_2D, credTexture8);
			break;


		case 9:
			gl.bindTexture(gl.TEXTURE_2D, credTexture9);
			break;

		case 10:
			gl.bindTexture(gl.TEXTURE_2D, credTexture10);
			break;



	}		

	gl.uniform1i(textureSamplerUniform_cr, 0);

	//Play Cassette
	gl.bindVertexArray(vao_square_rvg);

	//Draw Object
	gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);

	//Stop Playing Cassette
	gl.bindVertexArray(null);

	//Stop using Shader Program Object
	gl.useProgram(null);

	gl.disable(gl.BLEND);
}





