var canvas;
var gl;
var global_viewMatrix;

var vertexShaderObject_rvg;
var fragmentShaderObject_rvg;
var shaderProgramObject_rvg;

var vao_star_rvg;
var vbo_star_position_rvg;
var vbo_star_color_rvg;

var t = 0.1;

var Stone_Texture;

var timeUniform;
var mvpUniform_rvg;
var textureSamplerUniform;

var perspectiveProjectionMatrix_rvg;

// Random number generator
var seed = 0x13371337;

const NUM_STARS = 1000;


function random_float()
{
    var res;
    var tmp;

    seed *= 16807;

    tmp = seed ^ (seed >> 4) ^ (seed << 15);

    res = (tmp >> 9) | 0x3F800000;

    return (res - 1.0);
}



function initialize_PointSprite()
{
			// ****************** Vertex Shader ******************
	//Create Shader
	vertexShaderObject_rvg = gl.createShader(gl.VERTEX_SHADER);

	//Provide Source Code to Shader
	var vertexShaderSource_rvg =
		"#version 300 es" +
		"\n" +
		"in vec4 vPosition;" +
		"in vec4 vColor;" +
		"uniform float time;" +
		

		"uniform mat4 u_model_mat;" +
		"uniform mat4 u_view_mat;" +
		"uniform mat4 u_proj_mat;" +


		"flat out vec4 starColor;" +
		"void main(void)" +
		"{"	+
			"vec4 newVertex = vPosition;" +
			
			"newVertex.y -= time;" +
			"newVertex.y = fract(newVertex.y);" +
			
			"float size = (15.0 * newVertex.y * newVertex.y);" +
			
			"starColor = smoothstep(1.0, 7.0, size) * vColor;" +
			
			"newVertex.y = (999.9 * newVertex.y) - 1000.0;" +

			"gl_Position = u_proj_mat * u_view_mat * u_model_mat * newVertex;" +
			
			"gl_PointSize = size;" +
		"}";		

	//Feed to Shader Object
	gl.shaderSource(vertexShaderObject_rvg, vertexShaderSource_rvg);

	//Compile Shader
	gl.compileShader(vertexShaderObject_rvg);

	//Shader Compilation Error Checking
	if (gl.getShaderParameter(vertexShaderObject_rvg, gl.COMPILE_STATUS) == false)
	{
		var error_rvg = gl.getShaderInfoLog(vertexShaderObject_rvg);
		if (error_rvg.length > 0)
		{
			alert(error_rvg);
			uninitialize();
		}
	}
	else
	{
		console.log("RVG : Vertex Shader Compilation is done Successfully.");
	}

	// ****************** Fragment Shader ******************
	//Create Shader
	fragmentShaderObject_rvg = gl.createShader(gl.FRAGMENT_SHADER);

	//Provide Source Code to Shader
	var fragmentShaderSource_rvg =
		"#version 300 es" +
		"\n" +
		"precision highp float;" +
		"uniform sampler2D u_texture_sampler;" +
		"flat in vec4 starColor;" +
		"out vec4 fragColor;" +
		"void main(void)" +
		"{"	+
			"vec2 p = gl_PointCoord * 2.0 - vec2(1.0);" +
			"if(dot(p,p) > 1.0)" +
			"{" +
				"discard;" +
			"}" +						
			"fragColor =  starColor * texture(u_texture_sampler, gl_PointCoord);" +
		"}";


	//Feed to Shader Object
	gl.shaderSource(fragmentShaderObject_rvg, fragmentShaderSource_rvg);

	//Compile Shader
	gl.compileShader(fragmentShaderObject_rvg);

	//Shader Compilation Error Checking
	if (gl.getShaderParameter(fragmentShaderObject_rvg, gl.COMPILE_STATUS) == false)
	{
		var error_rvg = gl.getShaderInfoLog(fragmentShaderObject_rvg);
		if (error_rvg.length > 0)
		{
			alert(error_rvg);
			uninitialize();
		}
	}
	else
	{
		console.log("RVG : Fragment Shader Compilation is done Successfully.");
	}

	//Shader Linking Code
	//1)Create the Shader Program which capable of linking Shader.
	shaderProgramObject_rvg = gl.createProgram();

	//2)Attach whichever Shaders you have to this Shader Program Object.
	gl.attachShader(shaderProgramObject_rvg, vertexShaderObject_rvg);
	gl.attachShader(shaderProgramObject_rvg, fragmentShaderObject_rvg);

	//3)(Pre-Linking) Bind with Shader Program Object with Attributes
	gl.bindAttribLocation(shaderProgramObject_rvg, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
	gl.bindAttribLocation(shaderProgramObject_rvg, WebGLMacros.AMC_ATTRIBUTE_COLOR, "vColor");

	//4)Then, Link all those attached Shader at once.
	gl.linkProgram(shaderProgramObject_rvg);

	//Shader Linkage Error Checking
	if (!gl.getProgramParameter(shaderProgramObject_rvg, gl.LINK_STATUS))
	{
		var error_rvg = gl.getProgramInfoLog(shaderProgramObject_rvg);
		if (error_rvg.length > 0)
		{
			alert(error_rvg);
			uninitialize();
		}
	}
	else
	{
		console.log("RVG : Program Shader Object Compilation is done Successfully.");
	}

	//Get MVP Uniform Location
	timeUniform = gl.getUniformLocation(shaderProgramObject_rvg, "time");
	mvpUniform_rvg = gl.getUniformLocation(shaderProgramObject_rvg, "u_mvpMatrix");
	textureSamplerUniform = gl.getUniformLocation(shaderProgramObject_rvg, "u_texture_sampler");

	//Set Texture
	//glEnable(GL_TEXTURE_2D);
	Stone_Texture = gl.createTexture();
	Stone_Texture.image = new Image();
	Stone_Texture.image.src = "08-PointSprite/Star.png";
	Stone_Texture.image.onload = function()
	{
		gl.bindTexture(gl.TEXTURE_2D, Stone_Texture);
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);

		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);

		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, Stone_Texture.image);
		gl.generateMipmap(gl.TEXTURE_2D);
		gl.bindTexture(gl.TEXTURE_2D, null);
	}	

	//Create Cassette to use Buffer Array in Display()
	vao_star_rvg = gl.createVertexArray();
	gl.bindVertexArray(vao_star_rvg);

	var starPosition = new Float32Array(NUM_STARS * 3);
	var starColor = new Float32Array(NUM_STARS * 3);

	for (var i = 0; i < NUM_STARS; i++)
	{
			starPosition[i * 3 + 0] = (Math.random() * 2.0 - 1.0) * 1024.0;
			starPosition[i * 3 + 1] = Math.random() - 300.0;
			starPosition[i * 3 + 2] = (Math.random() * 2.0 - 1.0) * 1024.0

			starColor[i * 3 + 0] = 0.8 + Math.random() * 0.2;
			starColor[i * 3 + 1] = 0.8 + Math.random() * 0.2;
			starColor[i * 3 + 2] = 0.8 + Math.random() * 0.2;
	}

	/////////////////////////////// FOR starPosition ///////////////////////////////
	//Generate Buffer Array
	vbo_star_position_rvg = gl.createBuffer();

	//Bind Buffer Array
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_star_position_rvg);

	//Fill Data into Buffer Array
	gl.bufferData(gl.ARRAY_BUFFER, starPosition, gl.STATIC_DRAW);

	//Tell him How to Read Data from Buffer Array
	gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION, 3, gl.FLOAT, false, 0, 0);
			
	//Enable Attribute Pointer
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);

	//Unbind Buffer Array
	gl.bindBuffer(gl.ARRAY_BUFFER, null);

	/////////////////////////////// FOR starColor ///////////////////////////////
	//Generate Buffer Array
	vbo_star_color_rvg = gl.createBuffer();

	//Bind Buffer Array
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo_star_color_rvg);

	//Fill Data into Buffer Array
	gl.bufferData(gl.ARRAY_BUFFER, starColor, gl.STATIC_DRAW);

	//Tell him How to Read Data from Buffer Array
	gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_COLOR, 3, gl.FLOAT, false, 0, 0);
			
	//Enable Attribute Pointer
	gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);

	//Unbind Buffer Array
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	
	//Stop Creating Cassette
	gl.bindVertexArray(null);


	perspectiveProjectionMatrix_rvg = mat4.create();

}


function uninitialize_PointSprite()
{
	if (vao_star_rvg)
	{
		gl.deleteVertexArray(vao_star_rvg);
		vao_star_rvg = null;
	}

	if (vbo_star_position_rvg)
	{
		gl.deleteBuffer(vbo_star_position_rvg);
		vbo_star_position_rvg = null;
	}

	if (vbo_texture_star)
	{
		gl.deleteBuffers(1, vbo_texture_star);
		vbo_texture_star = 0;
	}


	if (shaderProgramObject_rvg)
	{
		if (fragmentShaderObject_rvg)
		{
			gl.detachShader(shaderProgramObject_rvg, fragmentShaderObject_rvg);
			gl.deleteShader(fragmentShaderObject_rvg);
			fragmentShaderObject_rvg = null;
		}

		if (vertexShaderObject_rvg)
		{
			gl.detachShader(shaderProgramObject_rvg, vertexShaderObject_rvg);
			gl.deleteShader(vertexShaderObject_rvg);
			vertexShaderObject_rvg = null;
		}

		gl.deleteProgram(shaderProgramObject_rvg);
		shaderProgramObject_rvg = null;
	}
}



function draw_PointSprite()
{
	//Local Variable Declaration
	t -= 0.001;											//NEW	

	

	mat4.perspective(perspectiveProjectionMatrix_rvg, 45.0, 
		parseFloat(canvas.width) / parseFloat(canvas.height), 0.1, 4000.0);

	var modelMatrix = mat4.create();


	mat4.rotateZ(modelMatrix, modelMatrix, degToRad(180.0));

	//Use Shader Program Object
	gl.useProgram(shaderProgramObject_rvg);

	//Set Uniform
	gl.uniform1f(timeUniform, t);


	vec3.add(gCameraLookingAt, gCameraPosition, gCameraFront);
	mat4.lookAt(global_viewMatrix, gCameraPosition, gCameraLookingAt, gCameraUp);

	gl.uniformMatrix4fv(gl.getUniformLocation(shaderProgramObject_rvg, "u_model_mat"), false, modelMatrix);
	gl.uniformMatrix4fv(gl.getUniformLocation(shaderProgramObject_rvg, "u_view_mat"), false, global_viewMatrix);
	gl.uniformMatrix4fv(gl.getUniformLocation(shaderProgramObject_rvg, "u_proj_mat"), false, perspectiveProjectionMatrix_rvg);
	
	//Set Texture
	gl.activeTexture(gl.TEXTURE0);
	gl.bindTexture(gl.TEXTURE_2D, Stone_Texture);
	gl.uniform1i(textureSamplerUniform, 0);

	gl.enable(gl.BLEND);					//NEW
	gl.blendFunc(gl.ONE, gl.ONE);			//NEW

	//Play Cassette
	gl.bindVertexArray(vao_star_rvg);

	//Draw Object
	gl.drawArrays(gl.POINTS, 0, NUM_STARS);

	//Stop Playing Cassette
	gl.bindVertexArray(null);

	//Stop using Shader Program Object
	gl.useProgram(null);

	gl.disable(gl.BLEND);	
}


