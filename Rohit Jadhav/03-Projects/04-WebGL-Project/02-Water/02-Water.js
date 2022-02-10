//For External
var canvas;
var gl;
var global_viewMatrix;
var gGrid_StartX;
var gGrid_StartZ;
var gCameraPosition;


//For Water Reflection and Refraction
var gFBO_Reflection;
var gFBO_Reflection_Tex;
var gFBO_Reflection_DepthRenderbuffer;

var gFBO_Refraction;
var gFBO_Refraction_Tex;
var gFBO_Refraction_Depth_Tex;





//For Shader
var water_vertexShaderObject;
var water_fragmentShaderObject;
var water_shaderProgramObject;

var water_perspectiveProjectionMatrix;

//For Rect
var water_vao_Rect;
var water_vbo_Rect_Position;
var water_vbo_Rect_Texcoord;

var water_modelMatUniform;
var water_viewMatUniform;
var water_projMatUniform;
var water_samplerReflectionUniform;
var water_samplerRefractionUniform;
var water_samplerDuDvUniform;
var water_samplerDepth;
var water_dudvAnimationUniform;
var water_camposUniform;


var water_dudvTexture;
var water_animation = 0.0;
const water_animationFactor = 0.001;



function initialize_water(){


	water_vertexShaderObject = gl.createShader(gl.VERTEX_SHADER);
	var szVertexShaderSourceCode = 
		"#version 300 es" +
		"\n" +

		"precision mediump float;" +

		"in vec4 vPosition;" +
		"in vec2 vTex;" + 
		
		"uniform mat4 u_model_matrix;" +
		"uniform mat4 u_view_matrix;" +
		"uniform mat4 u_proj_matrix;" +
		"uniform vec3 u_cam_pos;" +

		"out vec2 outTexCoord;" +
		"out vec4 outClipSpaceCoordinate;" +
		"out vec3 outVertexToCam_Vector;" +


		

		
		"void main() {" +
			
			"outTexCoord = vTex * 4.0f;" +

			"vec4 newPos = vPosition;" +

			"newPos.y = 35.0f;" +

			"vec4 worldCoord = u_model_matrix * vec4(newPos.xyz, 1.0f);" +

			"outClipSpaceCoordinate = u_proj_matrix * u_view_matrix * worldCoord;" +

			"outVertexToCam_Vector = u_cam_pos - worldCoord.xyz;" +


			"gl_Position = outClipSpaceCoordinate;" +
		"}";

	gl.shaderSource(water_vertexShaderObject, szVertexShaderSourceCode);

	gl.compileShader(water_vertexShaderObject);

	var  shaderCompileStatus = gl.getShaderParameter(water_vertexShaderObject, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(water_vertexShaderObject);
		if(error.length > 0){
			alert("02-Water -> Vertex Shader Compilation Error: " + error);
			uninitialize_water();
			window.close();
		}
	}


	water_fragmentShaderObject = gl.createShader(gl.FRAGMENT_SHADER);
	var szFragmentShaderSourceCode = 
		"#version 300 es" +
		"\n" +
		"precision mediump float;" +
		"precision highp sampler2D;" +
	
		"in vec2 outTexCoord;" +
		"in vec4 outClipSpaceCoordinate;" +
		"in vec3 outVertexToCam_Vector;" +

		"uniform sampler2D u_sampler_reflection;" +
		"uniform sampler2D u_sampler_refraction;" +
		"uniform sampler2D u_sampler_dudv;" +
		"uniform sampler2D u_sampler_depth;" +

		"uniform float u_dudv_animationFactor;" +

		"const float fWaterStrength = 0.01f;" +
		"const float fNear = 0.1f;" +
		"const float fFar = 4000.0f;" +

		"uniform int u_light;" +
	
		"out vec4 v4FragColor;" +
		

		"void main(void)" +
		"{" +
			
			"vec2 normalizeDeviceCoord = (outClipSpaceCoordinate.xy / outClipSpaceCoordinate.w) / 2.0f + 0.5;" +


			//For Distortion in Water
			// "vec2 distortion1 = (texture(u_sampler_dudv, vec2(outTexCoord.x, outTexCoord.y + u_dudv_animationFactor)).rg * 2.0f - 1.0f) * fWaterStrength;" +
			// "vec2 distortion2 = (texture(u_sampler_dudv, vec2(outTexCoord.x + u_dudv_animationFactor, -outTexCoord.y + u_dudv_animationFactor)).rg * 2.0f - 1.0f) * fWaterStrength;" +

			"vec2 distortedTexCoord = (texture(u_sampler_dudv, vec2(outTexCoord.x + u_dudv_animationFactor, outTexCoord.y))).rg * 0.1f;" +
			"distortedTexCoord = outTexCoord + vec2(distortedTexCoord.x, distortedTexCoord.y + u_dudv_animationFactor);" +

			
			"vec2 reflectTexcoord = vec2(normalizeDeviceCoord.x, -normalizeDeviceCoord.y);" +
			"vec2 refractTexcoord = vec2(normalizeDeviceCoord.x, normalizeDeviceCoord.y);" +

			"float depth = texture(u_sampler_depth, refractTexcoord).r;" +
			"float floorDistance = 2.0f * fNear * fFar / (fFar + fNear - (2.0f * depth - 1.0f) * (fFar - fNear));" +

			"depth = gl_FragCoord.z;" +
			"float waterDistance = 2.0f * fNear * fFar / (fFar + fNear - (2.0f * depth - 1.0f) * (fFar - fNear));" +

			"float waterDepth = floorDistance - waterDistance;" +

			"vec2 totalDistortion = (texture(u_sampler_dudv, distortedTexCoord).rg * 2.0f - 1.0f) * fWaterStrength * clamp(waterDepth / 20.0f, 0.0f, 1.0f);" +
			// "vec2 totalDistortion = (distortion1 + distortion2) * clamp(waterDepth / 30.0f, 0.0f, 1.0f);" +
			// "vec2 totalDistortion = (distortion1 + distortion2); " + // * clamp(waterDepth / 50.0f, 0.0f, 1.0f);" +


			"reflectTexcoord += totalDistortion;" +
			"reflectTexcoord.x = clamp(reflectTexcoord.x, 0.001f, 0.999f);" +
			"reflectTexcoord.y = clamp(reflectTexcoord.y, -0.999f, -0.001f);" +


			"refractTexcoord += totalDistortion;" +
			"refractTexcoord = clamp(refractTexcoord, 0.001f, 0.999f);" +


			//For Normal Reflection
			"vec4 reflection = texture(u_sampler_reflection, reflectTexcoord);" +
			"vec4 refraction = texture(u_sampler_refraction, refractTexcoord);" +

			//For Fresnel Effect vec3(0.0f, 1.0f, 0.0f)-> Normal for the plane
			"vec3 normalize_VertexToCam_Vec = normalize(outVertexToCam_Vector);" +
			"float reflectionFactor = dot(normalize_VertexToCam_Vec, vec3(0.0f, 1.0f, 0.0f));" +
			"reflectionFactor = pow(reflectionFactor, 2.0f);" +


			"v4FragColor = mix(reflection, refraction, reflectionFactor);" +


			"if(u_light == 0)" + 	
				"v4FragColor = mix(v4FragColor, vec4(0.0f, 0.20f, 0.50f, 1.0f), 0.2f);" +
			

			"v4FragColor.a = clamp(waterDepth / 8.0f, 0.0f, 1.0f);" +
			

			// "v4FragColor = reflection;" +
 

		"}";

	gl.shaderSource(water_fragmentShaderObject, szFragmentShaderSourceCode);
	gl.compileShader(water_fragmentShaderObject);

	shaderCompileStatus = gl.getShaderParameter(water_fragmentShaderObject, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(water_fragmentShaderObject);
		if(error.length > 0){
			alert("02-Water -> Fragment Shader Compilation Error: "+ error);
			uninitialize_water();
			window.close();
		}
	}


	water_shaderProgramObject = gl.createProgram();

	gl.attachShader(water_shaderProgramObject, water_vertexShaderObject);
	gl.attachShader(water_shaderProgramObject, water_fragmentShaderObject);

	gl.bindAttribLocation(water_shaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
	gl.bindAttribLocation(water_shaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0, "vTex");

	gl.linkProgram(water_shaderProgramObject);

	var programLinkStatus = gl.getProgramParameter(water_shaderProgramObject, gl.LINK_STATUS);

	if(programLinkStatus == false){
		var error = gl.getProgramInfoLog(water_shaderProgramObject);
		if(error.length > 0){
			alert("02-Water -> Program Linkink Error: " + error);
			uninitialize_water();
			window.close();
		}
	}



	water_modelMatUniform = gl.getUniformLocation(water_shaderProgramObject, "u_model_matrix");
	water_viewMatUniform = gl.getUniformLocation(water_shaderProgramObject, "u_view_matrix");
	water_projMatUniform = gl.getUniformLocation(water_shaderProgramObject, "u_proj_matrix");
	water_samplerReflectionUniform = gl.getUniformLocation(water_shaderProgramObject, "u_sampler_reflection");
	water_samplerRefractionUniform = gl.getUniformLocation(water_shaderProgramObject, "u_sampler_refraction");
	water_samplerDuDvUniform = gl.getUniformLocation(water_shaderProgramObject, "u_sampler_dudv");
	water_dudvAnimationUniform = gl.getUniformLocation(water_shaderProgramObject, "u_dudv_animationFactor");
	water_camposUniform = gl.getUniformLocation(water_shaderProgramObject, "u_cam_pos");
	water_samplerDepth = gl.getUniformLocation(water_shaderProgramObject, "u_sampler_depth");


	var x = -gGrid_StartX;
	var z = -gGrid_StartZ;
	var rect_pos = new Float32Array([
			x, 0.0, -z,
			-x, 0.0, -z,
			-x, 0.0, z,
			x, 0.0, z,
		]);

		

	var rect_Texcoord = new Float32Array([
			1.0, 1.0,
			0.0, 1.0,
			0.0, 0.0,
			1.0, 0.0,
		]);



	/********* Rectangle *********/
	water_vao_Rect = gl.createVertexArray();
	gl.bindVertexArray(water_vao_Rect);

		/********* Position **********/
		water_vbo_Rect_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, water_vbo_Rect_Position);
		gl.bufferData(gl.ARRAY_BUFFER,  rect_pos, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION, 3, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Texture **********/
		water_vbo_Rect_Texcoord = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, water_vbo_Rect_Texcoord);
		gl.bufferData(gl.ARRAY_BUFFER, rect_Texcoord, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0, 2, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);

	water_perspectiveProjectionMatrix = mat4.create();





	// ***** Reflection Framebuffer *****
	gFBO_Reflection = gl.createFramebuffer();
	gl.bindFramebuffer(gl.FRAMEBUFFER, gFBO_Reflection);


		/********** Texture **********/
		gFBO_Reflection_Tex = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, gFBO_Reflection_Tex);

		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
		
		gl.texImage2D(gl.TEXTURE_2D, 0, 
				gl.RGBA, 
				gFBO_Water_Tex_Width, gFBO_Water_Tex_Height, 0,
				gl.RGBA, 
				gl.UNSIGNED_BYTE, null);

		gl.bindTexture(gl.TEXTURE_2D, null);
		
		gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, gFBO_Reflection_Tex, 0);

		


		/********** For Depth **********/
		gFBO_Reflection_DepthRenderbuffer = gl.createRenderbuffer();
		gl.bindRenderbuffer(gl.RENDERBUFFER, gFBO_Reflection_DepthRenderbuffer);
		gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, gFBO_Water_Tex_Width, gFBO_Water_Tex_Height);
		gl.bindRenderbuffer(gl.RENDERBUFFER, null);

		gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, gFBO_Reflection_DepthRenderbuffer);


		/********** Checking *********/
		if(gl.checkFramebufferStatus(gl.FRAMEBUFFER) != gl.FRAMEBUFFER_COMPLETE){
			alert("Water Reflection : checkFramebufferStatus: Failed");
			uninitialize_HeightMap();
			window.close();
		}

	gl.bindFramebuffer(gl.FRAMEBUFFER, null);




	// var a = gl.getSupportedExtensions();
	// //var a = gl.getExtension('WRBGL_depth_texture');
	// console.log(a);


	// ***** Refraction Framebuffer *****
	gFBO_Refraction = gl.createFramebuffer();
	gl.bindFramebuffer(gl.FRAMEBUFFER, gFBO_Refraction);


		/********** Color **********/
		gFBO_Refraction_Tex = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, gFBO_Refraction_Tex);

		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
		
		gl.texImage2D(gl.TEXTURE_2D, 0, 
				gl.RGBA, 
				gFBO_Water_Tex_Width, gFBO_Water_Tex_Height, 0,
				gl.RGBA, 
				gl.UNSIGNED_BYTE, null);

		gl.bindTexture(gl.TEXTURE_2D, null);
		
		gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, gFBO_Refraction_Tex, 0);

		

		// ********* Depth *********
		gFBO_Refraction_Depth_Tex = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, gFBO_Refraction_Depth_Tex);

		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		
		gl.texImage2D(gl.TEXTURE_2D, 0, 
				gl.DEPTH_COMPONENT32F, 
				gFBO_Water_Tex_Width, gFBO_Water_Tex_Height, 0,
				gl.DEPTH_COMPONENT, 
				gl.FLOAT, null);

		gl.bindTexture(gl.TEXTURE_2D, null);
		
		gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, gFBO_Refraction_Depth_Tex, 0);

		//gl.drawBuffers([gl.NONE]);
		


		/********** Checking *********/
		if(gl.checkFramebufferStatus(gl.FRAMEBUFFER) != gl.FRAMEBUFFER_COMPLETE){
			alert("Water Refraction : checkFramebufferStatus: Failed");
			uninitialize_HeightMap();
			window.close();
		}

	gl.bindFramebuffer(gl.FRAMEBUFFER, null);




	//Load Texture for DUDV
	water_dudvTexture = gl.createTexture();
	water_dudvTexture.image = new Image();
	water_dudvTexture.image.src = "00-Textures/waterDUDV.png";
	water_dudvTexture.image.onload = function(){
		gl.bindTexture(gl.TEXTURE_2D, water_dudvTexture);
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
		gl.texImage2D(gl.TEXTURE_2D,
					0,
					gl.RGBA,
					gl.RGBA,
					gl.UNSIGNED_BYTE,
					water_dudvTexture.image);
		gl.generateMipmap(gl.TEXTURE_2D);
		gl.bindTexture(gl.TEXTURE_2D, null);
	}





	console.log("02-Water -> initialize_water() complete");

}


function uninitialize_water(){

	//For DuDv Tex
	if(water_dudvTexture){
		gl.deleteTexture(water_dudvTexture);
		water_dudvTexture = 0;
	}



	//Refraction
	if(gFBO_Refraction_Tex){
		gl.deleteTexture(gFBO_Refraction_Tex);
		gFBO_Refraction_Tex = 0;
	}

	if(gFBO_Refraction){
		gl.deleteFramebuffer(gFBO_Refraction);
		gFBO_Refraction = 0;
	}


	//Reflection
	if(gFBO_Reflection_DepthRenderbuffer){
		gl.deleteRenderbuffer(gFBO_Reflection_DepthRenderbuffer);
		gFBO_Reflection_DepthRenderbuffer = 0;
	}

	if(gFBO_Reflection_Tex){
		gl.deleteTexture(gFBO_Reflection_Tex);
		gFBO_Reflection_Tex = 0;
	}	

	if(gFBO_Reflection){
		gl.deleteFramebuffer(gFBO_Reflection);
		gFBO_Reflection = 0;
	}




	if(water_vbo_Rect_Texcoord){
		gl.deleteBuffer(water_vbo_Rect_Texcoord);
		water_vbo_Rect_Texcoord = 0;
	}


	if(water_vbo_Rect_Position){
		gl.deleteBuffer(water_vbo_Rect_Position);
		water_vbo_Rect_Position = 0;
	}

	if(water_vao_Rect){
		gl.deleteVertexArray(water_vao_Rect);
		water_vao_Rect = 0;
	}


	if(water_shaderProgramObject){

		gl.useProgram(water_shaderProgramObject);

			if(water_fragmentShaderObject){
				gl.detachShader(water_shaderProgramObject, water_fragmentShaderObject);
				gl.deleteShader(water_fragmentShaderObject);
				water_fragmentShaderObject = 0;
			}

			if(water_vertexShaderObject){
				gl.detachShader(water_shaderProgramObject, water_vertexShaderObject);
				gl.deleteShader(water_vertexShaderObject);
				water_vertexShaderObject = 0;
			}

		gl.useProgram(null);
		gl.deleteProgram(water_shaderProgramObject);
		water_shaderProgramObject = 0;
	}

	console.log("02-Water -> uninitialize_water() complete");

}

function display_water(choice){


	var water_modelMatrix = mat4.create();


	//console.log(canvas.width, canvas.height);
	gl.viewport(0, 0, canvas.width, canvas.height);

	mat4.perspective(water_perspectiveProjectionMatrix, 45.0,
			parseFloat(canvas.width) / parseFloat(canvas.height),
			0.1, 4000.0);
	
	//gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);


	gl.useProgram(water_shaderProgramObject);


		gl.enable(gl.BLEND);
		gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

		/********** Rectangle **********/
		mat4.identity(water_modelMatrix);
		
		gl.uniformMatrix4fv(water_modelMatUniform, false, water_modelMatrix);
		gl.uniformMatrix4fv(water_viewMatUniform, false, global_viewMatrix);
		gl.uniformMatrix4fv(water_projMatUniform, false, water_perspectiveProjectionMatrix);

		gl.uniform1i(gl.getUniformLocation(water_shaderProgramObject, "u_light"), choice);


		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, gFBO_Reflection_Tex);
		gl.uniform1i(water_samplerReflectionUniform, 0);

		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, gFBO_Refraction_Tex);
		gl.uniform1i(water_samplerRefractionUniform, 1);

		gl.activeTexture(gl.TEXTURE2);
		gl.bindTexture(gl.TEXTURE_2D, water_dudvTexture);
		gl.uniform1i(water_samplerDuDvUniform, 2);

		gl.activeTexture(gl.TEXTURE3);
		gl.bindTexture(gl.TEXTURE_2D, gFBO_Refraction_Depth_Tex);
		gl.uniform1i(water_samplerDepth, 3);

		gl.uniform1f(water_dudvAnimationUniform, water_animation);

		gl.uniform3fv(water_camposUniform, gCameraPosition);



		
		gl.bindVertexArray(water_vao_Rect);

			gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);

		gl.bindVertexArray(null);



		gl.activeTexture(gl.TEXTURE3);
		gl.bindTexture(gl.TEXTURE_2D, null);

		gl.activeTexture(gl.TEXTURE2);
		gl.bindTexture(gl.TEXTURE_2D, null);

		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, null);
		
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, null);

		gl.disable(gl.BLEND);

		
	gl.useProgram(null);


	water_animation += water_animationFactor;
}