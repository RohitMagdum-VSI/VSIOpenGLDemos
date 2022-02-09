//global variables
var canvas=null;
var gl=null;//WebGL context
var bFullscreen=false;
var canvas_original_width;
var canvas_original_height;

//When whole WebGLMacros is const, means all inside it are automatically const
//This concept is called Named Value Pairing / Key Value Coding or Object Initializer
const WebGLMacros=
{
    HAD_ATTRIBUTE_POSITION:0,
    HAD_ATTRIBUTE_COLOR:1,
    HAD_ATTRIBUTE_NORMAL:2,
    HAD_ATTRIBUTE_TEXTURE0:3,
};

//Shader Object
var vertexShaderObject;
var fragmentShaderObject;

//Program Object
var shaderProgramObject;

var light_ambient=[0.0,0.0,0.0];
var light_diffuse=[1.0,1.0,1.0];
var light_specular=[1.0,1.0,1.0];
var light_position=[100.0,100.0,100.0,1.0];

var material_ambient=[0.0,0.0,0.0];
var material_diffuse=[1.0,1.0,1.0];
var material_specular=[1.0,1.0,1.0];
var material_shininess=50.0;

var sphere=null;

var IsLKeyPressed=false;
var IsAKeyPressed=false;

var modelMatrixUniform,viewMatrixUniform, projectionMatrixUniform;
var laUniform,ldUniform,lsUniform,lightPositionUniform;
var kaUniform,kdUniform,ksUniform,materialShininessUniform;
var lKeyPressedUniform;

var perspectiveProjectionMatrix;

var gAngle=0.0;

//To start Animation : to have requestAnimationFrame to be called "cross-browser" compatible
var requestAnimationFrame=window.requestAnimationFrame||
window.webkitRequestAnimationFrame||
window.mozRequestAnimationFrame||
window.oRequestAnimationFrame||
window.msRequestAnimationFrame;

var cancelAnimationFrame=window.cancelAnimationFrame||
window.webkitCancelAnimationFrame||
window.webkitCancelRequestAnimationFrame||
window.mozCancelRequestAnimationFrame||
window.mozCancelAnimmationFrame||
window.oCancelRequestAnimationFrame||
window.oCancelAnimationFrame||
window.msCancelRequestAnimationFrame||
window.msCancelAnimationFrame;

//onload function
function main()
{
    //get <canvas> element
    canvas = document.getElementById("HAD");
    if(!canvas)
        console.log("Obtaining Canvas Failed\n");
    else
        console.log("Obtaining Canvas Succeeded\n");

    canvas_original_width=canvas.width;
    canvas_original_height=canvas.height;
   
    //register keyboard's keydown event handler
    window.addEventListener("keydown",keyDown,false);
    window.addEventListener("click",mouseDown,false);
    window.addEventListener("resize",resize,false);

    //initialize WebGL
    init();

    //start drawing here as warming - up
    resize();
    draw();
}

function toggleFullScreen()
{
    var fullscreen_element=document.fullscreenElement||document.webkitFullscreenElement||document.mozFullScreenElement||document.msFullscreenElement||null;

    if(fullscreen_element==null)
    {
        if(canvas.requestFullscreen)
            canvas.requestFullscreen();
        else if(canvas.mozRequestFullScreen)
            canvas.mozRequestFullScreen();
        else if(canvas.webkitRequestFullscreen)
            canvas.webkitRequestFullscreen();
        else if(canvas.msRequestFullscreen)
            canvas.msRequestFullscreen();
        bFullscreen=true;
    }

    else
    {
        if(document.exitFullscreen)
            document.exitFullscreen();
        else if(document.mozCancelFullScreen)
            document.mozCancelFullScreen();
        else if(document.webkitExitFullscreen)
            document.webkitExitFullscreen();
        else if(document.msExitFullscreen)
            document.msExitFullscreen();
        bFullscreen=false;
    }
}

function init()
{
    //Get WebGL 2.0 Context
    gl=canvas.getContext("webgl2");
    if(gl==null)
    {
        console.log("Failed to get the rendering context for WebGL");
        return;
    }
    console.log("Rendering context for WebGL Obtained");

    gl.viewportWidth=canvas.width;
    gl.viewportHeight=canvas.height;

    //Vertex Shader
    var vertexShaderSourceCode=
    "#version 300 es"+
    "\n"+
    "in vec4 vPosition;"+
    "in vec3 vNormal;"+
    "uniform mat4 u_model_matrix;"+
    "uniform mat4 u_view_matrix;"+
    "uniform mat4 u_projection_matrix;"+
    "uniform mediump int u_lKeyPressed;"+
    "uniform vec3 u_La;"+
    "uniform vec3 u_Ld;"+
    "uniform vec3 u_Ls;"+
    "uniform vec4 u_light_position;"+
    "uniform vec3 u_Ka;"+
    "uniform vec3 u_Kd;"+
    "uniform vec3 u_Ks;"+
    "uniform float u_material_shininess;"+
    "out vec3 phong_ads_color;"+
    "void main(void)"+
    "{"+
    "if(u_lKeyPressed == 1)"+
    "{"+
    "vec4 eye_coordinates = u_view_matrix * u_model_matrix * vPosition;"+
    "vec3 transformed_normals = normalize(mat3(u_view_matrix * u_model_matrix) * vNormal);"+
    "vec3 light_direction = normalize(vec3(u_light_position) - eye_coordinates.xyz);"+
    "float tn_dot_ld = max(dot(transformed_normals, light_direction),0.0);"+
    "vec3 ambient = u_La * u_Ka;"+
    "vec3 diffuse = u_Ld * u_Kd * tn_dot_ld;"+
    "vec3 reflection_vector = reflect(-light_direction, transformed_normals);"+
    "vec3 viewer_vector = normalize(-eye_coordinates.xyz);"+
    "vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector,viewer_vector),0.0),u_material_shininess);"+
    "phong_ads_color = ambient + diffuse + specular;"+
    "}"+
    "else"+
    "{"+
    "phong_ads_color = vec3(1.0,1.0,1.0);"+
    "}"+
    "gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;"+
    "}";

    vertexShaderObject=gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShaderObject,vertexShaderSourceCode);
    gl.compileShader(vertexShaderObject);
    //Instead of glGetShaderiv() we need to use gl.getShaderParamter in WebGL
    if(gl.getShaderParameter(vertexShaderObject,gl.COMPILE_STATUS)==false)
    {
        var error=gl.getShaderInfoLog(vertexShaderObject);
        if(error.length>0)
        {
            alert(error);
            uninitialize();
        }
    }

    //Fragment Shader
    var fragmentShaderSourceCode=
    "#version 300 es"+
    "\n"+
    "precision highp float;"+
    "in vec3 phong_ads_color;"+
    "out vec4 FragColor;"+
    "void main(void)"+
    "{"+
    "FragColor=vec4(phong_ads_color,1.0);"+
    "}";

    fragmentShaderObject=gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShaderObject,fragmentShaderSourceCode);
    gl.compileShader(fragmentShaderObject);
    if(gl.getShaderParameter(fragmentShaderObject,gl.COMPILE_STATUS)==false)
    {
        var error = gl.getShaderInfoLog(fragmentShaderObject);
        if(error.length>0)
        {
            alert(error);
            uninitialize();
        }
    }

    //Shader Program
    shaderProgramObject=gl.createProgram();
    gl.attachShader(shaderProgramObject,vertexShaderObject);
    gl.attachShader(shaderProgramObject,fragmentShaderObject);

    //Pre-link binding of shader program object with shader's attributes 
    gl.bindAttribLocation(shaderProgramObject,WebGLMacros.HAD_ATTRIBUTE_POSITION,"vPosition");

    gl.bindAttribLocation(shaderProgramObject,WebGLMacros.HAD_ATTRIBUTE_NORMAL,"vNormal");

    //Linking
    gl.linkProgram(shaderProgramObject);
    if(!gl.getProgramParameter(shaderProgramObject,gl.LINK_STATUS))
    {
        var error = gl.getProgramInfoLog(shaderProgramObject);
        if(error.length>0)
        {
            alert(error);
            uninitialize();
        }
    }

    //Get Uniform Locations
    modelMatrixUniform = gl.getUniformLocation(shaderProgramObject,"u_model_matrix");

    viewMatrixUniform = gl.getUniformLocation(shaderProgramObject,"u_view_matrix");

    projectionMatrixUniform = gl.getUniformLocation(shaderProgramObject,"u_projection_matrix");

    lKeyPressedUniform = gl.getUniformLocation(shaderProgramObject,"u_lKeyPressed");

    laUniform=gl.getUniformLocation(shaderProgramObject,"u_La");

    ldUniform=gl.getUniformLocation(shaderProgramObject,"u_Ld");

    lsUniform=gl.getUniformLocation(shaderProgramObject,"u_Ls");

    lightPositionUniform=gl.getUniformLocation(shaderProgramObject,"u_light_position");

    kaUniform=gl.getUniformLocation(shaderProgramObject,"u_Ka");

    kdUniform=gl.getUniformLocation(shaderProgramObject,"u_Kd");

    ksUniform=gl.getUniformLocation(shaderProgramObject,"u_Ks");

    materialShininessUniform = gl.getUniformLocation(shaderProgramObject,"u_material_shininess");

    /*Vertices Array*/
    sphere = new Mesh();
    makeSphere(sphere,2.0,30,30);

    gl.clearColor(0.0,0.0,0.0,1.0);

    gl.enable(gl.DEPTH_TEST);
    //gl.clearDepth(1.0);
    gl.depthFunc(gl.LEQUAL);
    gl.enable(gl.CULL_FACE);

    perspectiveProjectionMatrix=mat4.create();
}

function resize()
{
    if(bFullscreen==true)
    {
        canvas.width=window.innerWidth;
        canvas.height=window.innerHeight;
    }
    else
    {
        canvas.width=canvas_original_width;
        canvas.height=canvas_original_height;
    }

    gl.viewport(0,0,canvas.width,canvas.height);

    //Orthographic Projection
    /*if(canvas.width <= canvas.height)
        mat4.ortho(orthographicProjectionMatrix,-100.0,100.0,(-100.0 * (canvas.height/canvas.width)),(100.0 * (canvas.height/canvas.width)),-100.0,100.0);
    else
        mat4.ortho(orthographicProjectionMatrix,(-100.0 * (canvas.width/canvas.height)),(100.0 * (canvas.width/canvas.height)),-100.0,100.0,-100.0,100.0);*/

    mat4.perspective(perspectiveProjectionMatrix,45.0,parseFloat(canvas.width)/parseFloat(canvas.height),0.1,100.0);
}

function draw()
{
    // code
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(shaderProgramObject);

    if(IsLKeyPressed == true)
    {
        gl.uniform1i(lKeyPressedUniform,1);

        gl.uniform3fv(laUniform,light_ambient);
        gl.uniform3fv(ldUniform,light_diffuse);
        gl.uniform3fv(lsUniform,light_specular);
        gl.uniform4fv(lightPositionUniform,light_position);

        gl.uniform3fv(kaUniform,material_ambient);
        gl.uniform3fv(kdUniform,material_diffuse);
        gl.uniform3fv(ksUniform,material_specular);
        gl.uniform1f(materialShininessUniform,material_shininess);
    }
    else
    {
        gl.uniform1i(lKeyPressedUniform,0);
    }

    var modelMatrix=mat4.create();
    var viewMatrix=mat4.create();
   // var translationMatrix=mat4.create();
    //var rotationMatrix=mat4.create();

    mat4.translate(modelMatrix,modelMatrix,[0.0,0.0,-6.0]);
    mat4.rotateY(modelMatrix,modelMatrix,degToRad(gAngle));

    gl.uniformMatrix4fv(modelMatrixUniform,false,modelMatrix);
    gl.uniformMatrix4fv(viewMatrixUniform,false,viewMatrix);
    gl.uniformMatrix4fv(projectionMatrixUniform,false,perspectiveProjectionMatrix);

    sphere.draw();

    gl.useProgram(null);

    if(IsAKeyPressed==true)
        update();
    
    // animation loop
    requestAnimationFrame(draw, canvas);
}

function update()
{
    gAngle=gAngle+1.0;
    if(gAngle>=360.0)
        gAngle = gAngle - 360.0;
}

function degToRad(degrees)
{
    return(degrees * Math.PI/180.0);
}

function keyDown(event)
{
    switch(event.keyCode)
    {
        case 70: //F or f
            toggleFullScreen();
            break;

        case 27: //ESC
            uninitialize();
            window.close();
            break;

        case 76: //L or l
            if(IsLKeyPressed == false)
                IsLKeyPressed=true;
            else
                IsLKeyPressed=false;
            break;

        case 65:
            if(IsAKeyPressed == false)
                IsAKeyPressed=true;
            else
                IsAKeyPressed=false;
            break;
    }
}

function mouseDown()
{
    
}

function uninitialize()
{
    if(sphere)
    {
        sphere.deallocate();
        sphere=null;
    }

    if(shaderProgramObject)
    {
        if(fragmentShaderObject)
        {
            gl.detachShader(shaderProgramObject,fragmentShaderObject);
            gl.deleteShader(fragmentShaderObject);
            fragmentShaderObject=null;
        }

        if(vertexShaderObject)
        {
            gl.detachShader(shaderProgramObject,vertexShaderObject);
            gl.deleteShader(vertexShaderObject);
            vertexShaderObject=null;
        }

        gl.deleteProgram(shaderProgramObject);
        shaderProgramObject=null;
    }
}