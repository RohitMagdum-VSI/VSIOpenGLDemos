//Global variables
var gl = null;

var canvas_AAW = null;
var isFullscreen_AAW = false;
var canvasOriginalWidth_AAW;
var canvasOriginalHeight_AAW;

var requestAnimationFrame_AAW = window.requestAnimationFrame ||
                                window.webkitRequestAnimationFrame ||
                                window.mozRequestAnimationFrame ||
                                window.oRequestAnimationFrame ||
                                window.msRequestAnimationFrame;

const webGLMacros_AAW = {AAW_ATTRIBUTE_VERTEX   : 0,
                         AAW_ATTRIBUTE_COLOR    : 1,
                         AAW_ATTRIBUTE_NORMAL   : 2,
                         AAW_ATTRIBUTE_TEXTURE0 : 3,
                         AAW_ATTRIBUTE_MODELMATRIX : 4};

var gShaderProgramObject_Scene_AAW;
var gShaderProgramObject_ShadowMap_AAW;

var gModelMatrixUniform_AAW;
var gViewMatrixUniform_AAW;
var gProjectionMatrixUniform_AAW;

var gLaUniform_AAW;
var gLdUniform_AAW;
var gLsUniform_AAW;
var gKaUniform_AAW;
var gKdUniform_AAW;
var gKsUniform_AAW;
var gLightPositionUniform_AAW;
var gMaterialShininessUniform_AAW;

var gLightAmbient_AAW = [0.1, 0.1, 0.1];
var gLightDiffuse_AAW = [1.0, 1.0, 1.0];
var gLightSpecular_AAW = [1.0, 1.0, 1.0];
var gLightPosition_AAW = [100.0, 100.0, 100.0, 0.0];

var gMaterialAmbient_AAW = [0.1, 0.0, 0.2];
var gMaterialDiffuse_AAW = [0.3, 0.3, 0.8];
var gMaterialSpecular_AAW = [1.0, 1.0, 1.0];
var gMaterialShininess_AAW = 70.0;

var gPerspectiveProjectionMatrix_AAW;
var gModel = null;

//Shadow Mapping
const DEPTH_TEXTURE_SIZE_AAW = 2048;
var gFrameBuffer_AAW;
var gShadowMapTexture_AAW;

var gVao_Ground_AAW;
var gVao_ShadowMap_AAW;

var gMVPMatrixUniform_AAW;
var gTextureSamplerUniform_AAW;
var gShadowMatrixUnifom_AAW;
var gAspect;

var gChoiceUniform_AAW;
var gChoiceUniform2_AAW;
const ASM_INSTANCED_COUNT = 10;

var keysPressed = [];
var xCameraPos = 0.0;
var yCameraPos = 0.0;
var zCameraPos = 0.0;

//Onload function
async function main(){

    //Get Canvas from DOM
    canvas_AAW = document.getElementById("AAW");
    if(!canvas_AAW)
        console.log("Failed to obatin Canvas\n");
    else
        console.log("Successfully obatained Canvas\n");

    canvasOriginalWidth_AAW = canvas_AAW.width;
    canvasOriginalHeight_AAW = canvas_AAW.height;

    //Add event listener
    window.addEventListener("keydown", keyDown, false);
    window.addEventListener('keyup', keyUp, false);
    window.addEventListener("click", mouseDown, false);
    window.addEventListener("resize", resize, false);

    initialize();
    resize();

    gModel = new Model();
    const response = await fetch('tree.obj');
    const objSource = await response.text();

    //New
    var ASM_modelMatrixValues = new Float32Array(16 * ASM_INSTANCED_COUNT * ASM_INSTANCED_COUNT);
	var ASM_translateMatrix = mat4.create();
	var ASM_scaleMatrix = mat4.create();
	var ASM_modelViewMatrix = mat4.create();		

	for (var i = 0; i < ASM_INSTANCED_COUNT; i++){
        for (var k = 0; k < ASM_INSTANCED_COUNT; k++){

            mat4.translate(ASM_translateMatrix, ASM_translateMatrix, [Math.random() * 600 - 300, -2.0, Math.random() * 600 - 300]);
		    mat4.multiply(ASM_modelViewMatrix, ASM_modelViewMatrix, ASM_translateMatrix);
		    for (var j = 0; j < 16; j++)
		    {
			    ASM_modelMatrixValues[((i * ASM_INSTANCED_COUNT + k) * 16) + j] = ASM_modelViewMatrix[j];	

		    }		
		    mat4.identity(ASM_translateMatrix);
		    mat4.identity(ASM_scaleMatrix);
		    mat4.identity(ASM_modelViewMatrix);
        }
	}
    gModel.parseOBJ(objSource, true, ASM_modelMatrixValues, ASM_INSTANCED_COUNT * ASM_INSTANCED_COUNT);

    display();
}

function toggleFullscreen(){

    //Code
    var fullscreenElement = document.fullscreenElement ||
                            document.webkitFullscreenElement ||
                            document.mozFullScreenElement ||
                            document.msFullscreenElement ||
                            null;

    if(fullscreenElement == null){
        if(canvas_AAW.requestFullscreen)
            canvas_AAW.requestFullscreen();
        else if(canvas_AAW.webkitRequestFullscreen)
            canvas_AAW.webkitRequestFullscreen();
        else if(canvas_AAW.mozRequestFullScreen)
            canvas_AAW.mozRequestFullScreen();
        else if(canvas_AAW.msRequestFullscreen)
            canvas_AAW.msRequestFullscreen();

        isFullscreen_AAW = true;
    }
    else{
        if(document.exitFullscreen)
            document.exitFullscreen();
        else if(document.webkitExitFullscreen)
            document.webkitExitFullscreen();
        else if(document.mozCancelFullScreen)
            document.mozCancelFullScreen();
        else if(document.msExitFullscreen)
            document.msExitFullscreen();

        isFullscreen_AAW = false;
    }
}

function keyDown(event){

    //Code
    keysPressed[event.keyCode] = true;
    switch(event.keyCode){
        case 70:
            toggleFullscreen();
            break;

        case 27:
            uninitialize();
            window.close();
            break;
    }
}

function keyUp(event){
    keysPressed[event.keyCode] = false;
}

function mouseDown(event){

}

function initialize(){

    //Code
    //Get WebGL drawing context from Canvas
    gl = canvas_AAW.getContext("webgl2");
    if(!gl){
        console.log("Failed to obtain WebGL 2.0 Context\n");
    }
    else{
        console.log("Successfully obtained WebGL 2.0 Context\n");
    }

    gl.viewportWidth = canvas_AAW.width;
    gl.viewportHeight == canvas_AAW.height;

    //initialize
    initializeShader_ShadowMap_AAW();
    initializeShader_Scene_AAW();
    initializeSceneData_AAW();
    initializeDepth_FrameBuffer_AAW();
    
    gl.clearDepth(1.0);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);

    gl.enable(gl.CULL_FACE);

    gl.clearColor(0.0, 0.0, 0.0, 1.0);

    gPerspectiveProjectionMatrix_AAW = mat4.create();
}

function resize(){

    //Code
    if(isFullscreen_AAW){
        canvas_AAW.width = window.innerWidth;
        canvas_AAW.height = window.innerHeight;
    }
    else{
        canvas_AAW.width = canvasOriginalWidth_AAW;
        canvas_AAW.height = canvasOriginalHeight_AAW;
    }

    gAspect = parseFloat(canvas_AAW.width) / parseFloat(canvas_AAW.height);
}

function display(){

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    displayScene_AAW();
    update();
    requestAnimationFrame_AAW(display, canvas_AAW);
}

function degToRad(degree){

    //Code
    return degree * Math.PI / 180.0;
}

function update(){

    //Code
    //"W"
    if(keysPressed[87]){
		zCameraPos += 0.5;
    }

    //"S"
    if(keysPressed[83]){
		zCameraPos -= 0.5;
    }

    //"D"
    if(keysPressed[68]){
		xCameraPos -= 0.5;
    }

    //"A"
    if(keysPressed[65]){
		xCameraPos += 0.5;
    }

    //"Space"
    if(keysPressed[32]){
		yCameraPos -= 0.5;
    }

    //"C"
    if(keysPressed[67]){
		yCameraPos += 0.5;
    }
}

function uninitialize(){

    //Code
    if(gModel){
        gModel.deallocate();
        gModel = null;
    }

    if(gShaderProgramObject_Scene_AAW){
        gl.useProgram(gShaderProgramObject_Scene_AAW);
        var shaderCount = gl.getProgramParameter(gShaderProgramObject_Scene_AAW, gl.GL_ATTACHED_SHADERS);
        var shaders = gl.getAttachedShaders(gShaderProgramObject_Scene_AAW);

        for(var i = 0; i < shaderCount; i++){
            gl.detachShader(gShaderProgramObject_Scene_AAW, shaders[i]);
            gl.deleteShader(shaders[i]);
            shaders[i] = null;
        }

        gl.useProgram(null);
        shaders = null;

        gl.deleteProgram(gShaderProgramObject_Scene_AAW);
        gShaderProgramObject_Scene_AAW = null;
    }

    if(gShaderProgramObject_ShadowMap_AAW){
        gl.useProgram(gShaderProgramObject_ShadowMap_AAW);
        var shaderCount = gl.getProgramParameter(gShaderProgramObject_ShadowMap_AAW, gl.GL_ATTACHED_SHADERS);
        var shaders = gl.getAttachedShaders(gShaderProgramObject_ShadowMap_AAW);

        for(var i = 0; i < shaderCount; i++){
            gl.detachShader(gShaderProgramObject_ShadowMap_AAW, shaders[i]);
            gl.deleteShader(shaders[i]);
            shaders[i] = null;
        }

        gl.useProgram(null);
        shaders = null;

        gl.deleteProgram(gShaderProgramObject_ShadowMap_AAW);
        gShaderProgramObject_ShadowMap_AAW = null;
    }
}

function initializeDepth_FrameBuffer_AAW(){

    //Code
    gFrameBuffer_AAW = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, gFrameBuffer_AAW);

        //Create a texture to store the shadow map
        gShadowMapTexture_AAW = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, gShadowMapTexture_AAW);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT32F, DEPTH_TEXTURE_SIZE_AAW, DEPTH_TEXTURE_SIZE_AAW, 0, gl.DEPTH_COMPONENT, gl.FLOAT, null);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_FUNC, gl.LEQUAL);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.bindTexture(gl.TEXTURE_2D, null);

        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, gShadowMapTexture_AAW, 0);
        gl.drawBuffers([gl.NONE]);
        var status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
        if(status != gl.FRAMEBUFFER_COMPLETE){
            console.log("Framebuffer Error : "+status);
        }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

function initializeShader_ShadowMap_AAW(){

    //Variable declarations
    var vertexShaderObject_AAW;
    var fragmentShaderObject_AAW;
    
    //Code
        //Vertex Shader
        var vertexShaderSourceCode = 
        "#version 300 es"+
        "\n"+
        "in         vec4    vPosition;" +
        "uniform    mat4    u_mvpMatrix;" +
        "in         mat4    modelMatrix;"+
        "uniform    int     u_choice;"+
        "void main(void){" +
        "   if(u_choice == 0){"+
        "       gl_Position = u_mvpMatrix * vPosition;"+
        "   }"+
        "   else if(u_choice == 1){"+
        "       gl_Position = u_mvpMatrix * modelMatrix * vPosition;" +
        "   }"+
        "}";

    vertexShaderObject_AAW = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShaderObject_AAW, vertexShaderSourceCode);
    gl.compileShader(vertexShaderObject_AAW);
    if(gl.getShaderParameter(vertexShaderObject_AAW, gl.COMPILE_STATUS) == false){
        var error = gl.getShaderInfoLog(vertexShaderObject_AAW);
        if(error.length > 0){
            alert("Shadow Vertex Shader Compilation Log : "+error);
            uninitialize();
        }
    }

    //Fragment Shader
    var fragmentShaderSourceCode = 
        "#version 300 es"+
        "\n"+
        "precision  highp   float;"+
        "out        vec4    FragColor;" +
        "void main(void){" +
        "   FragColor = vec4(1.0f);" +
        "}";

    fragmentShaderObject_AAW = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShaderObject_AAW, fragmentShaderSourceCode);
    gl.compileShader(fragmentShaderObject_AAW);
    if(gl.getShaderParameter(fragmentShaderObject_AAW, gl.COMPILE_STATUS) == false){
        var error = gl.getShaderInfoLog(fragmentShaderObject_AAW);
        if(error.length > 0){
            alert("Shadow Fragment Shader Compilation Log : "+error);
            uninitialize();
        }
    }

    //Shader Program
    gShaderProgramObject_ShadowMap_AAW = gl.createProgram();
    gl.attachShader(gShaderProgramObject_ShadowMap_AAW, vertexShaderObject_AAW);
    gl.attachShader(gShaderProgramObject_ShadowMap_AAW, fragmentShaderObject_AAW);

    gl.bindAttribLocation(gShaderProgramObject_ShadowMap_AAW, webGLMacros_AAW.AAW_ATTRIBUTE_VERTEX, "vPosition");
    gl.bindAttribLocation(gShaderProgramObject_ShadowMap_AAW, webGLMacros_AAW.AAW_ATTRIBUTE_MODELMATRIX, "modelMatrix");

    //Program Linking
    gl.linkProgram(gShaderProgramObject_ShadowMap_AAW);
    if(gl.getProgramParameter(gShaderProgramObject_ShadowMap_AAW, gl.LINK_STATUS) == false){
        var error = gl.getProgramInfoLog(gShaderProgramObject_ShadowMap_AAW);
        if(error.length > 0){
            alert(error);
            uninitialize();
        }
    }

    gMVPMatrixUniform_AAW = gl.getUniformLocation(gShaderProgramObject_ShadowMap_AAW, "u_mvpMatrix");
    gChoiceUniform_AAW = gl.getUniformLocation(gShaderProgramObject_ShadowMap_AAW, "u_choice");
}

function initializeShader_Scene_AAW(){

    //Variable declarations
    var vertexShaderObject_AAW;
    var fragmentShaderObject_AAW;

    //Code
    //Vertex Shader
    var vertexShaderSourceCode = 
        "#version 300 es"+
        "\n"+
        "in         vec4    vPosition;"+
        "in         vec3    vNormal;"+
        "uniform    mat4    u_modelMatrix;"+
        "uniform    mat4    u_viewMatrix;"+
        "uniform    mat4    u_projectionMatrix;"+
        "uniform    mat4    u_shadowMatrix;"+
        "uniform    vec4    u_lightPosition;"+
        "in         mat4    modelMatrix;"+
        "uniform    int     u_choice;"+
        "uniform    mediump int     u_lightKeyPressed;"+
        "out        vec3    out_transformedNormals;"+
        "out        vec3    out_lightDirection;"+
        "out        vec3    out_viewVector;"+
        "out        vec4    out_shadowCoord;"+
        "void main(void){"+
        "   if(u_choice == 0){"+
        "       vec4 eyeCoordinates     = u_viewMatrix * u_modelMatrix * vPosition;"+
        "       out_transformedNormals  = mat3(u_viewMatrix * u_modelMatrix) * vNormal;"+
        "       out_lightDirection      = vec3(u_lightPosition - eyeCoordinates);"+
        "       out_viewVector          = -eyeCoordinates.xyz;"+
        "       out_shadowCoord         = u_shadowMatrix * u_modelMatrix * vPosition;"+
        "       gl_Position = u_projectionMatrix * u_viewMatrix * u_modelMatrix * vPosition;"+
        "   }"+
        "   else if(u_choice == 1){"+
        "       vec4 eyeCoordinates     = u_viewMatrix * modelMatrix * vPosition;"+
        "       out_transformedNormals  = mat3(u_viewMatrix * modelMatrix) * vNormal;"+
        "       out_lightDirection      = vec3(u_lightPosition - eyeCoordinates);"+
        "       out_viewVector          = -eyeCoordinates.xyz;"+
        "       out_shadowCoord         = u_shadowMatrix * modelMatrix * vPosition;"+
        "       gl_Position = u_projectionMatrix * u_viewMatrix * modelMatrix * vPosition;"+
        "   }"+
        "}";

    vertexShaderObject_AAW = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShaderObject_AAW, vertexShaderSourceCode);
    gl.compileShader(vertexShaderObject_AAW);
    if(gl.getShaderParameter(vertexShaderObject_AAW, gl.COMPILE_STATUS) == false){
        var error = gl.getShaderInfoLog(vertexShaderObject_AAW);
        if(error.length > 0){
            alert("Scene Vertex Shader Compilation Log : "+error);
            uninitialize();
        }
    }

    //Fragment Shader
    var fragmentShaderSourceCode = 
        "#version 300 es"+
        "\n"+
        "precision  highp           float;"+
        "in         vec3            out_transformedNormals;"+
        "in         vec3            out_lightDirection;"+
        "in         vec3            out_viewVector;"+
        "in         vec4            out_shadowCoord;"+
        "uniform    highp   sampler2DShadow u_depthTextureSampler;"+
        "uniform    vec3            u_la;"+
        "uniform    vec3            u_ld;"+
        "uniform    vec3            u_ls;"+
        "uniform    vec3            u_ka;"+
        "uniform    vec3            u_kd;"+
        "uniform    vec3            u_ks;"+
        "uniform    float           u_materialShininess;"+
        "uniform    int             u_lightKeyPressed;"+
        "out        vec4            FragColor;"+
        "void main(void){"+
        "   vec3 Fong_ADS_Light;"+
        "   vec3 normalizedTransformedNormals   = normalize(out_transformedNormals);"+
        "   vec3 normalizedLightDirection       = normalize(out_lightDirection);"+
        "   vec3 normalizedViewVector           = normalize(out_viewVector);"+
        "   vec3 reflectionVector               = reflect(-normalizedLightDirection, normalizedTransformedNormals);"+
        "   float f          = textureProj(u_depthTextureSampler, out_shadowCoord);"+
        "   vec3 ambient    = u_la * u_ka;"+
        "   vec3 diffuse    = f * u_ld * u_kd * max(dot(normalizedLightDirection, normalizedTransformedNormals), 0.0f);"+
        "   vec3 specular   = f * u_ls * u_ks * pow(max(dot(normalizedViewVector, reflectionVector), 0.0f), u_materialShininess);"+
        "   Fong_ADS_Light  = ambient + diffuse + specular;"+
        "   FragColor = vec4(Fong_ADS_Light, 1.0f);"+
        "}";

    fragmentShaderObject_AAW = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShaderObject_AAW, fragmentShaderSourceCode);
    gl.compileShader(fragmentShaderObject_AAW);
    if(gl.getShaderParameter(fragmentShaderObject_AAW, gl.COMPILE_STATUS) == false){
        var error = gl.getShaderInfoLog(fragmentShaderObject_AAW);
        if(error.length > 0){
            alert("Scene Fragment Shader Compilation Log : "+error);
            uninitialize();
        }
    }

    //Shader Program
    gShaderProgramObject_Scene_AAW = gl.createProgram();
    gl.attachShader(gShaderProgramObject_Scene_AAW, vertexShaderObject_AAW);
    gl.attachShader(gShaderProgramObject_Scene_AAW, fragmentShaderObject_AAW);

    gl.bindAttribLocation(gShaderProgramObject_Scene_AAW, webGLMacros_AAW.AAW_ATTRIBUTE_VERTEX, "vPosition");
    gl.bindAttribLocation(gShaderProgramObject_Scene_AAW, webGLMacros_AAW.AAW_ATTRIBUTE_NORMAL, "vNormal");
    gl.bindAttribLocation(gShaderProgramObject_Scene_AAW, webGLMacros_AAW.AAW_ATTRIBUTE_MODELMATRIX, "modelMatrix");

    //Program Linking
    gl.linkProgram(gShaderProgramObject_Scene_AAW);
    if(gl.getProgramParameter(gShaderProgramObject_Scene_AAW, gl.LINK_STATUS) == false){
        var error = gl.getProgramInfoLog(gShaderProgramObject_Scene_AAW);
        if(error.length > 0){
            alert(error);
            uninitialize();
        }
    }

    gModelMatrixUniform_AAW          = gl.getUniformLocation(gShaderProgramObject_Scene_AAW, "u_modelMatrix");
    gViewMatrixUniform_AAW           = gl.getUniformLocation(gShaderProgramObject_Scene_AAW, "u_viewMatrix");
    gProjectionMatrixUniform_AAW     = gl.getUniformLocation(gShaderProgramObject_Scene_AAW, "u_projectionMatrix");
    gShadowMatrixUnifom_AAW          = gl.getUniformLocation(gShaderProgramObject_Scene_AAW, "u_shadowMatrix");
    gTextureSamplerUniform_AAW       = gl.getUniformLocation(gShaderProgramObject_Scene_AAW, "u_depthTextureSampler");
    gLaUniform_AAW                   = gl.getUniformLocation(gShaderProgramObject_Scene_AAW, "u_la");
    gLdUniform_AAW                   = gl.getUniformLocation(gShaderProgramObject_Scene_AAW, "u_ld");
    gLsUniform_AAW                   = gl.getUniformLocation(gShaderProgramObject_Scene_AAW, "u_ls");
    gKaUniform_AAW                   = gl.getUniformLocation(gShaderProgramObject_Scene_AAW, "u_ka");
    gKdUniform_AAW                   = gl.getUniformLocation(gShaderProgramObject_Scene_AAW, "u_kd");
    gKsUniform_AAW                   = gl.getUniformLocation(gShaderProgramObject_Scene_AAW, "u_ks");
    gLightPositionUniform_AAW        = gl.getUniformLocation(gShaderProgramObject_Scene_AAW, "u_lightPosition");
    gMaterialShininessUniform_AAW    = gl.getUniformLocation(gShaderProgramObject_Scene_AAW, "u_materialShininess");
    gChoiceUniform2_AAW              = gl.getUniformLocation(gShaderProgramObject_Scene_AAW, "u_choice");
}

function initializeSceneData_AAW(){

    //Variable declarations
    var vbo_GroundPosition = 0;
    var vbo_GroundNormals = 0;
    const groundVertices = new Float32Array([-300.0, -2.0, -300.0, 1.0,
                                             -300.0, -2.0,  300.0, 1.0,
                                              300.0, -2.0,  300.0, 1.0,
                                              300.0, -2.0, -300.0, 1.0]);
    const groundNormals = new Float32Array([0.0, 1.0, 0.0,
                                            0.0, 1.0, 0.0,
                                            0.0, 1.0, 0.0,
                                            0.0, 1.0, 0.0]);

    //Code
    gVao_Ground_AAW = gl.createVertexArray();
    gl.bindVertexArray(gVao_Ground_AAW);
    
        //Position
        vbo_GroundPosition = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo_GroundPosition);
            gl.bufferData(gl.ARRAY_BUFFER, groundVertices, gl.STATIC_DRAW);
            gl.vertexAttribPointer(webGLMacros_AAW.AAW_ATTRIBUTE_VERTEX, 4, gl.FLOAT, false, 0, 0);
            gl.enableVertexAttribArray(webGLMacros_AAW.AAW_ATTRIBUTE_VERTEX);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
    
        //Normals
        vbo_GroundNormals = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo_GroundNormals);
            gl.bufferData(gl.ARRAY_BUFFER, groundNormals, gl.STATIC_DRAW);
            gl.vertexAttribPointer(webGLMacros_AAW.AAW_ATTRIBUTE_NORMAL, 3, gl.FLOAT, false, 0, 0);
            gl.enableVertexAttribArray(webGLMacros_AAW.AAW_ATTRIBUTE_NORMAL);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
    
    gl.bindVertexArray(null);
}

function displayScene_AAW(){

    //Variable declaration
    var sceneModelMatrix = mat4.create();
    var sceneViewMatrix = mat4.create();
    var sceneProjectionMatrix = mat4.create();
    var shadowMatrix = mat4.create();

    var scaleBiasMatrix;

    var lightViewMatrix = mat4.create();
    var lightProjectionMatrix = mat4.create();
    var lightMVPMatrix = mat4.create();

    //Code
    mat4.lookAt(lightViewMatrix, gLightPosition_AAW, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    mat4.perspective(lightProjectionMatrix, 45.0, gAspect, 0.1, 600.0);

    mat4.multiply(lightMVPMatrix, lightMVPMatrix, lightProjectionMatrix);
    mat4.multiply(lightMVPMatrix, lightMVPMatrix, lightViewMatrix);

    gl.useProgram(gShaderProgramObject_ShadowMap_AAW);;
        gl.bindFramebuffer(gl.FRAMEBUFFER, gFrameBuffer_AAW);
            gl.viewport(0, 0, DEPTH_TEXTURE_SIZE_AAW, DEPTH_TEXTURE_SIZE_AAW);
            
            gl.clearDepth(1.0);
            gl.clear(gl.DEPTH_BUFFER_BIT);
            
            gl.enable(gl.POLYGON_OFFSET_FILL);
            gl.polygonOffset(2.0, 4.0);

            gl.uniform1i(gChoiceUniform_AAW, 1);
            gModel.drawModel();
            
            gl.uniform1i(gChoiceUniform_AAW, 0);
            mat4.multiply(lightMVPMatrix, lightMVPMatrix, sceneModelMatrix);
            gl.uniformMatrix4fv(gMVPMatrixUniform_AAW, false, lightMVPMatrix);
            
            gl.bindVertexArray(gVao_Ground_AAW);
                gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
            gl.bindVertexArray(null);
            
            gl.disable(gl.POLYGON_OFFSET_FILL);

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(null);



    scaleBiasMatrix = mat4.fromValues(0.5, 0.0, 0.0, 0.0,
                                      0.0, 0.5, 0.0, 0.0,
                                      0.0, 0.0, 0.5, 0.0,
                                      0.5, 0.5, 0.5, 1.0);
    mat4.translate(sceneViewMatrix, sceneViewMatrix, [xCameraPos, yCameraPos, -6.0 + zCameraPos]);
    mat4.perspective(sceneProjectionMatrix, 45.0, gAspect, 0.1, 100.0);

    gl.viewport(0, 0, canvas_AAW.width, canvas_AAW.height);
    gl.useProgram(gShaderProgramObject_Scene_AAW);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        mat4.multiply(shadowMatrix, shadowMatrix, scaleBiasMatrix);
        mat4.multiply(shadowMatrix, shadowMatrix, lightProjectionMatrix);
        mat4.multiply(shadowMatrix, shadowMatrix, lightViewMatrix);

        gl.uniformMatrix4fv(gModelMatrixUniform_AAW, false, sceneModelMatrix);
        gl.uniformMatrix4fv(gViewMatrixUniform_AAW, false, sceneViewMatrix);
        gl.uniformMatrix4fv(gProjectionMatrixUniform_AAW, false, sceneProjectionMatrix);
        gl.uniformMatrix4fv(gShadowMatrixUnifom_AAW, false, shadowMatrix);

        gl.uniform3fv(gLaUniform_AAW, gLightAmbient_AAW);
        gl.uniform3fv(gLdUniform_AAW, gLightDiffuse_AAW);
        gl.uniform3fv(gLsUniform_AAW, gLightSpecular_AAW);
        gl.uniform4fv(gLightPositionUniform_AAW, gLightPosition_AAW);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, gShadowMapTexture_AAW);
        gl.uniform1i(gTextureSamplerUniform_AAW, 0);

        gl.uniform3fv(gKaUniform_AAW, [0.1, 0.1, 0.1]);
        gl.uniform3fv(gKdUniform_AAW, [0.1, 0.5, 0.1]);
        gl.uniform3fv(gKsUniform_AAW, [0.1, 0.1, 0.1]);
        gl.uniform1f(gMaterialShininessUniform_AAW, 70.0);

        gl.uniform1i(gChoiceUniform2_AAW, 1);
        gModel.drawModel();

        
        gl.uniform3fv(gKaUniform_AAW, [0.0, 0.0, 0.0]);
        gl.uniform3fv(gKdUniform_AAW, [0.59, 0.42, 0.31]);
        gl.uniform3fv(gKsUniform_AAW, [0.59, 0.42, 0.31]);
        gl.uniform1f(gMaterialShininessUniform_AAW, 25.0);
        
        gl.uniform1i(gChoiceUniform2_AAW, 0);
        gl.bindVertexArray(gVao_Ground_AAW);
            gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
        gl.bindVertexArray(null);
    gl.useProgram(null);
}
