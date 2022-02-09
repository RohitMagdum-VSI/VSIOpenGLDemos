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
var light_position=[0.0,0.0,100.0,1.0];

var material_ambient=[0.0,0.0,0.0];
var material_diffuse=[1.0,1.0,1.0];
var material_specular=[1.0,1.0,1.0];
var material_shininess=50.0;

var sphere=null;

var IsLKeyPressed=false;
var IsXKeyPressed=false;
var IsYKeyPressed=false;
var IsZKeyPressed=false;

var modelMatrixUniform,viewMatrixUniform, projectionMatrixUniform;
var laUniform_White,ldUniform_White,lsUniform_White,lightPositionUniform_White;
var kaUniform,kdUniform,ksUniform,materialShininessUniform;
var lKeyPressedUniform;

var perspectiveProjectionMatrix;

var gAngle=0.0;


var material_ambient_1 = [ 0.0215, 0.1745, 0.0215];
var material_diffuse_1 = [ 0.07568, 0.61424, 0.07568];
var material_specular_1 = [ 0.633,0.727811,0.633];
var material_shininess_1 = 0.6 * 128.0;

var material_ambient_2 = [ 0.135,0.2225,0.1575];
var material_diffuse_2 = [ 0.54,0.8,0.63];
var material_specular_2 = [ 0.316228,0.316228,0.316228];
var material_shininess_2 = 0.1 * 128.0;

var material_ambient_3 = [ 0.05375,0.05,0.06625 ];
var material_diffuse_3 = [ 0.18275,0.17,0.22525 ];
var material_specular_3 = [ 0.332741,0.328634,0.346435 ];
var material_shininess_3 = 0.3 * 128.0;

var material_ambient_4 = [ 0.25,0.20725,0.20725];
var material_diffuse_4 = [ 1.0, 0.82, 0.82];
var material_specular_4 = [ 0.296648,0.296648,0.296648];
var material_shininess_4 = 0.088 * 128.0;

var material_ambient_5 = [ 0.1745,0.01175,0.01175];
var material_diffuse_5 = [ 0.61424,0.04136,0.04136];
var material_specular_5 = [ 0.727811,0.62695,0.62695];
var material_shininess_5 = 0.6 * 128.0;

var material_ambient_6 = [ 0.1,0.18725,0.1745];
var material_diffuse_6 = [ 0.396,0.74151,0.69102];
var material_specular_6 = [ 0.297254,0.3082,0.306678];
var material_shininess_6 = 0.1 * 128.0;

var material_ambient_7 = [ 0.329412,0.22352,0.027451];
var material_diffuse_7 = [ 0.780392,0.568627,0.113725];
var material_specular_7 = [ 0.992157,0.941176,0.807843];
var material_shininess_7 = 0.21794872 * 128.0;

var material_ambient_8 = [ 0.2125,0.1275,0.054];
var material_diffuse_8 = [ 0.714,0.4284,0.18144];
var material_specular_8 = [ 0.393548,0.271906,0.166721];
var material_shininess_8 = 0.2 * 128.0;

var material_ambient_9 = [ 0.25,0.25,0.25];
var material_diffuse_9 = [ 0.4,0.4,0.4];
var material_specular_9 = [ 0.774597,0.774597,0.774597];
var material_shininess_9 = 0.6 * 128.0;

var material_ambient_10 = [ 0.19125,0.0735,0.0225];
var material_diffuse_10 = [ 0.7038,0.27048,0.0828];
var material_specular_10 = [ 0.256777,0.137622,0.086014];
var material_shininess_10 = 0.1 * 128.0;

var material_ambient_11 = [ 0.24725,0.1995,0.0745];
var material_diffuse_11 = [ 0.75164,0.60648,0.22648];
var material_specular_11 = [ 0.628281,0.555802,0.366065];
var material_shininess_11 = 0.4 * 128.0;

var material_ambient_12 = [ 0.19225,0.19225,0.19225];
var material_diffuse_12 = [ 0.50754,0.50754,0.50754];
var material_specular_12 = [ 0.508273,0.508273,0.508273];
var material_shininess_12 = 0.4 * 128.0;

var material_ambient_13 = [ 0.0,0.0,0.0];
var material_diffuse_13 = [ 0.01,0.01,0.01];
var material_specular_13 = [ 0.5,0.5,0.5];
var material_shininess_13 = 0.25 * 128.0;

var material_ambient_14 = [ 0.0,0.1,0.06];
var material_diffuse_14 = [ 0.0,0.50980392,0.50980392];
var material_specular_14 = [ 0.50196078,0.50196078,0.50196078];
var material_shininess_14 = 0.25 * 128.0;

var material_ambient_15 = [ 0.0,0.0,0.0];
var material_diffuse_15 = [ 0.1,0.35,0.1];
var material_specular_15 = [ 0.45,0.55,0.45];
var material_shininess_15 = 0.25 * 128.0;

var material_ambient_16 = [ 0.0,0.0,0.0];
var material_diffuse_16 = [ 0.5,0.0,0.0];
var material_specular_16 = [ 0.7,0.6,0.6];
var material_shininess_16 = 0.25 * 128.0;

var material_ambient_17 = [ 0.0,0.0,0.0];
var material_diffuse_17 = [ 0.55,0.55,0.55];
var material_specular_17 = [ 0.70,0.70,0.70];
var material_shininess_17 = 0.25 * 128.0;

var material_ambient_18 = [ 0.0,0.0,0.0];
var material_diffuse_18 = [ 0.5,0.5,0.0];
var material_specular_18 = [ 0.6,0.6,0.5];
var material_shininess_18 = 0.25 * 128.0;

var material_ambient_19 = [ 0.02,0.02,0.02];
var material_diffuse_19 = [ 0.1,0.1,0.1];
var material_specular_19 = [ 0.4,0.4,0.4];
var material_shininess_19 = 0.078125 * 128.0;

var material_ambient_20 = [ 0.0,0.05,0.05];
var material_diffuse_20 = [ 0.4,0.5,0.5];
var material_specular_20 = [ 0.04,0.7,0.7];
var material_shininess_20 = 0.078125 * 128.0;

var material_ambient_21 = [ 0.0,0.05,0.0];
var material_diffuse_21 = [ 0.4,0.5,0.4];
var material_specular_21 = [ 0.04,0.7,0.04];
var material_shininess_21 = 0.078125 * 128.0;

var material_ambient_22 = [ 0.05,0.0,0.0];
var material_diffuse_22 = [ 0.5,0.4,0.4];
var material_specular_22 = [ 0.7,0.04,0.04];
var material_shininess_22 = 0.078125 * 128.0;

var material_ambient_23 = [ 0.05,0.05,0.05];
var material_diffuse_23 = [ 0.5,0.5,0.5];
var material_specular_23 = [ 0.7,0.7,0.7];
var material_shininess_23 = 0.078125 * 128.0;

var material_ambient_24 = [ 0.05,0.05,0.0];
var material_diffuse_24 = [ 0.5,0.5,0.4];
var material_specular_24 = [ 0.7,0.7,0.04];
var material_shininess_24 = 0.078125 * 128.0;


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
    "uniform vec4 u_light_position;"+
    "out vec3 transformed_normals;"+
    "out vec3 light_direction;"+
    "out vec3 viewer_vector;"+
    "void main(void)"+
    "{"+
    "if(u_lKeyPressed == 1)"+
    "{"+
    "vec4 eye_coordinates = u_view_matrix * u_model_matrix * vPosition;"+
    "transformed_normals = mat3(u_view_matrix * u_model_matrix) * vNormal;"+
    "light_direction = vec3(u_light_position) - eye_coordinates.xyz;"+
    "viewer_vector = -eye_coordinates.xyz;"+
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
    "in vec3 transformed_normals;"+
    "in vec3 light_direction;"+
    "in vec3 viewer_vector;"+
    "out vec4 FragColor;"+
    "uniform vec3 u_La;"+
    "uniform vec3 u_Ld;"+
    "uniform vec3 u_Ls;"+
    "uniform vec3 u_Ka;"+
    "uniform vec3 u_Kd;"+
    "uniform vec3 u_Ks;"+
    "uniform float u_material_shininess;"+
    "uniform int u_lKeyPressed;"+
    "void main(void)"+
    "{"+
    "vec3 phong_ads_color;"+
    "if(u_lKeyPressed == 1)"+
    "{"+
    "vec3 normalized_transformed_normals=normalize(transformed_normals);"+
    "vec3 normalized_light_direction = normalize(light_direction);"+
    "vec3 normalized_viewer_vector = normalize(viewer_vector);"+
    "vec3 ambient = u_La * u_Ka;"+
    "float tn_dot_ld = max(dot(normalized_transformed_normals,normalized_light_direction),0.0);"+
    "vec3 diffuse = u_Ld * u_Kd * tn_dot_ld;"+
    "vec3 reflection_vector = reflect(-normalized_light_direction,normalized_transformed_normals);"+
    "vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector,normalized_viewer_vector),0.0),u_material_shininess);"+
    "phong_ads_color=ambient+diffuse+specular;"+
    "}"+
    "else"+
    "{"+
    "phong_ads_color = vec3(1.0,1.0,1.0);"+
    "}"+
    "FragColor = vec4(phong_ads_color,1.0);"+
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

    laUniform_White=gl.getUniformLocation(shaderProgramObject,"u_La");

    ldUniform_White=gl.getUniformLocation(shaderProgramObject,"u_Ld");

    lsUniform_White=gl.getUniformLocation(shaderProgramObject,"u_Ls");

    lightPositionUniform_White=gl.getUniformLocation(shaderProgramObject,"u_light_position");

    kaUniform=gl.getUniformLocation(shaderProgramObject,"u_Ka");

    kdUniform=gl.getUniformLocation(shaderProgramObject,"u_Kd");

    ksUniform=gl.getUniformLocation(shaderProgramObject,"u_Ks");

    materialShininessUniform = gl.getUniformLocation(shaderProgramObject,"u_material_shininess");

    /*Vertices Array*/
    sphere = new Mesh();
    makeSphere(sphere,1.0,30,30);

    gl.clearColor(0.25,0.25,0.25,1.0);

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
        if(IsXKeyPressed==true)
        {
            light_position[0] = 0.0;
            light_position[1] = ((Math.sin(gAngle) * 10.0) - 6.0);
            light_position[2] = ((Math.cos(gAngle) * 10.0) - 6.0);
        }

        else if(IsYKeyPressed==true)
        {
            light_position[0] = ((Math.sin(gAngle) * 10.0) - 6.0);
            light_position[1] = 0.0;
            light_position[2] = ((Math.cos(gAngle) * 10.0) - 6.0);
        }

        else if(IsZKeyPressed==true)
        {
            light_position[0] = (Math.sin(gAngle) * 10.0);
            light_position[1] = (Math.cos(gAngle) * 10.0);
            light_position[2] = -6.0;
        }
        else
        {
            light_position[0] = 0.0;
            light_position[1] = 0.0;
            light_position[2] = 10.0;
        }

        gl.uniform1i(lKeyPressedUniform,1);

        gl.uniform3fv(laUniform_White,light_ambient);
        gl.uniform3fv(ldUniform_White,light_diffuse);
        gl.uniform3fv(lsUniform_White,light_specular);
        gl.uniform4fv(lightPositionUniform_White,light_position);
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
    //mat4.rotateY(modelMatrix,modelMatrix,degToRad(gAngle));

    gl.uniformMatrix4fv(modelMatrixUniform,false,modelMatrix);
    gl.uniformMatrix4fv(viewMatrixUniform,false,viewMatrix);
    gl.uniformMatrix4fv(projectionMatrixUniform,false,perspectiveProjectionMatrix);

    Draw_Sphere_1();
    Draw_Sphere_2();
    Draw_Sphere_3();
    Draw_Sphere_4();
    Draw_Sphere_5();
    Draw_Sphere_6();
    Draw_Sphere_7();
    Draw_Sphere_8();
    Draw_Sphere_9();
    Draw_Sphere_10();
    Draw_Sphere_11();
    Draw_Sphere_12();
    Draw_Sphere_13();
    Draw_Sphere_14();
    Draw_Sphere_15();
    Draw_Sphere_16();
    Draw_Sphere_17();
    Draw_Sphere_18();
    Draw_Sphere_19();
    Draw_Sphere_20();
    Draw_Sphere_21();
    Draw_Sphere_22();
    Draw_Sphere_23();
    Draw_Sphere_24();
    
    gl.useProgram(null);
    
    update();
    
    // animation loop
    requestAnimationFrame(draw, canvas);
}

function Draw_Sphere_1()
{
    gl.uniform3fv(kaUniform,material_ambient_1);
    gl.uniform3fv(kdUniform,material_diffuse_1);
    gl.uniform3fv(ksUniform,material_specular_1);
    gl.uniform1f(materialShininessUniform,material_shininess_1);

    gl.viewport(0,canvas.height * 5/6 - 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

function Draw_Sphere_2()
{
    gl.uniform3fv(kaUniform,material_ambient_2);
    gl.uniform3fv(kdUniform,material_diffuse_2);
    gl.uniform3fv(ksUniform,material_specular_2);
    gl.uniform1f(materialShininessUniform,material_shininess_2);

    gl.viewport(0,canvas.height * 4/6 - 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

function Draw_Sphere_3()
{
    gl.uniform3fv(kaUniform,material_ambient_3);
    gl.uniform3fv(kdUniform,material_diffuse_3);
    gl.uniform3fv(ksUniform,material_specular_3);
    gl.uniform1f(materialShininessUniform,material_shininess_3);

    gl.viewport(0,canvas.height * 3/6 - 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

function Draw_Sphere_4()
{
    gl.uniform3fv(kaUniform,material_ambient_4);
    gl.uniform3fv(kdUniform,material_diffuse_4);
    gl.uniform3fv(ksUniform,material_specular_4);
    gl.uniform1f(materialShininessUniform,material_shininess_4);

    gl.viewport(0,canvas.height * 2/6 - 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

function Draw_Sphere_5()
{
    gl.uniform3fv(kaUniform,material_ambient_5);
    gl.uniform3fv(kdUniform,material_diffuse_5);
    gl.uniform3fv(ksUniform,material_specular_5);
    gl.uniform1f(materialShininessUniform,material_shininess_5);

    gl.viewport(0,canvas.height * 1/6 - 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

function Draw_Sphere_6()
{
    gl.uniform3fv(kaUniform,material_ambient_6);
    gl.uniform3fv(kdUniform,material_diffuse_6);
    gl.uniform3fv(ksUniform,material_specular_6);
    gl.uniform1f(materialShininessUniform,material_shininess_6);

    gl.viewport(0,- 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

/************************/
function Draw_Sphere_7()
{
    gl.uniform3fv(kaUniform,material_ambient_7);
    gl.uniform3fv(kdUniform,material_diffuse_7);
    gl.uniform3fv(ksUniform,material_specular_7);
    gl.uniform1f(materialShininessUniform,material_shininess_7);

    gl.viewport(canvas.width / 4,canvas.height * 5/6 - 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

function Draw_Sphere_8()
{
    gl.uniform3fv(kaUniform,material_ambient_8);
    gl.uniform3fv(kdUniform,material_diffuse_8);
    gl.uniform3fv(ksUniform,material_specular_8);
    gl.uniform1f(materialShininessUniform,material_shininess_8);

    gl.viewport(canvas.width / 4,canvas.height * 4/6 - 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

function Draw_Sphere_9()
{
    gl.uniform3fv(kaUniform,material_ambient_9);
    gl.uniform3fv(kdUniform,material_diffuse_9);
    gl.uniform3fv(ksUniform,material_specular_9);
    gl.uniform1f(materialShininessUniform,material_shininess_9);

    gl.viewport(canvas.width / 4,canvas.height * 3/6 - 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

function Draw_Sphere_10()
{
    gl.uniform3fv(kaUniform,material_ambient_10);
    gl.uniform3fv(kdUniform,material_diffuse_10);
    gl.uniform3fv(ksUniform,material_specular_10);
    gl.uniform1f(materialShininessUniform,material_shininess_10);

    gl.viewport(canvas.width / 4,canvas.height * 2/6 - 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

function Draw_Sphere_11()
{
    gl.uniform3fv(kaUniform,material_ambient_11);
    gl.uniform3fv(kdUniform,material_diffuse_11);
    gl.uniform3fv(ksUniform,material_specular_11);
    gl.uniform1f(materialShininessUniform,material_shininess_11);

    gl.viewport(canvas.width / 4,canvas.height * 1/6 - 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

function Draw_Sphere_12()
{
    gl.uniform3fv(kaUniform,material_ambient_12);
    gl.uniform3fv(kdUniform,material_diffuse_12);
    gl.uniform3fv(ksUniform,material_specular_12);
    gl.uniform1f(materialShininessUniform,material_shininess_12);

    gl.viewport(canvas.width / 4,- 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

/************************/
function Draw_Sphere_13()
{
    gl.uniform3fv(kaUniform,material_ambient_13);
    gl.uniform3fv(kdUniform,material_diffuse_13);
    gl.uniform3fv(ksUniform,material_specular_13);
    gl.uniform1f(materialShininessUniform,material_shininess_13);

    gl.viewport(canvas.width / 2,canvas.height * 5/6 - 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

function Draw_Sphere_14()
{
    gl.uniform3fv(kaUniform,material_ambient_14);
    gl.uniform3fv(kdUniform,material_diffuse_14);
    gl.uniform3fv(ksUniform,material_specular_14);
    gl.uniform1f(materialShininessUniform,material_shininess_14);

    gl.viewport(canvas.width / 2,canvas.height * 4/6 - 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

function Draw_Sphere_15()
{
    gl.uniform3fv(kaUniform,material_ambient_15);
    gl.uniform3fv(kdUniform,material_diffuse_15);
    gl.uniform3fv(ksUniform,material_specular_15);
    gl.uniform1f(materialShininessUniform,material_shininess_15);

    gl.viewport(canvas.width / 2,canvas.height * 3/6 - 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

function Draw_Sphere_16()
{
    gl.uniform3fv(kaUniform,material_ambient_16);
    gl.uniform3fv(kdUniform,material_diffuse_16);
    gl.uniform3fv(ksUniform,material_specular_16);
    gl.uniform1f(materialShininessUniform,material_shininess_16);

    gl.viewport(canvas.width / 2,canvas.height * 2/6 - 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

function Draw_Sphere_17()
{
    gl.uniform3fv(kaUniform,material_ambient_17);
    gl.uniform3fv(kdUniform,material_diffuse_17);
    gl.uniform3fv(ksUniform,material_specular_17);
    gl.uniform1f(materialShininessUniform,material_shininess_17);

    gl.viewport(canvas.width / 2,canvas.height * 1/6 - 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

function Draw_Sphere_18()
{
    gl.uniform3fv(kaUniform,material_ambient_18);
    gl.uniform3fv(kdUniform,material_diffuse_18);
    gl.uniform3fv(ksUniform,material_specular_18);
    gl.uniform1f(materialShininessUniform,material_shininess_18);

    gl.viewport(canvas.width / 2,- 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

/************************/
function Draw_Sphere_19()
{
    gl.uniform3fv(kaUniform,material_ambient_19);
    gl.uniform3fv(kdUniform,material_diffuse_19);
    gl.uniform3fv(ksUniform,material_specular_19);
    gl.uniform1f(materialShininessUniform,material_shininess_19);

    gl.viewport((canvas.width/2)+(canvas.width/4),canvas.height * 5/6 - 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

function Draw_Sphere_20()
{
    gl.uniform3fv(kaUniform,material_ambient_20);
    gl.uniform3fv(kdUniform,material_diffuse_20);
    gl.uniform3fv(ksUniform,material_specular_20);
    gl.uniform1f(materialShininessUniform,material_shininess_20);

    gl.viewport((canvas.width/2)+(canvas.width/4),canvas.height * 4/6 - 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

function Draw_Sphere_21()
{
    gl.uniform3fv(kaUniform,material_ambient_21);
    gl.uniform3fv(kdUniform,material_diffuse_21);
    gl.uniform3fv(ksUniform,material_specular_21);
    gl.uniform1f(materialShininessUniform,material_shininess_21);

    gl.viewport((canvas.width/2)+(canvas.width/4),canvas.height * 3/6 - 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

function Draw_Sphere_22()
{
    gl.uniform3fv(kaUniform,material_ambient_22);
    gl.uniform3fv(kdUniform,material_diffuse_22);
    gl.uniform3fv(ksUniform,material_specular_22);
    gl.uniform1f(materialShininessUniform,material_shininess_22);

    gl.viewport((canvas.width/2)+(canvas.width/4),canvas.height * 2/6 - 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

function Draw_Sphere_23()
{
    gl.uniform3fv(kaUniform,material_ambient_23);
    gl.uniform3fv(kdUniform,material_diffuse_23);
    gl.uniform3fv(ksUniform,material_specular_23);
    gl.uniform1f(materialShininessUniform,material_shininess_23);

    gl.viewport((canvas.width/2)+(canvas.width/4),canvas.height * 1/6 - 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

function Draw_Sphere_24()
{
    gl.uniform3fv(kaUniform,material_ambient_24);
    gl.uniform3fv(kdUniform,material_diffuse_24);
    gl.uniform3fv(ksUniform,material_specular_24);
    gl.uniform1f(materialShininessUniform,material_shininess_24);

    gl.viewport((canvas.width/2)+(canvas.width/4),- 30,canvas.width/4,canvas.height/4);
    sphere.draw();
}

function update()
{
    gAngle=gAngle+0.03;
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

        case 88: //X or x
            if(IsXKeyPressed == false)
            {
                IsXKeyPressed=true;
                IsYKeyPressed=false;
                IsZKeyPressed=false;
            }
            else
                IsXKeyPressed=false;
            break;

        case 89: //Y or y
            if(IsYKeyPressed == false)
            {
                IsXKeyPressed=false;
                IsYKeyPressed=true;
                IsZKeyPressed=false;
            }
            else
                IsYKeyPressed=false;
            break;

        case 90: //Z or z
            if(IsZKeyPressed == false)
            {
                IsXKeyPressed=false;
                IsYKeyPressed=false;
                IsZKeyPressed=true;
            }
            else
                IsZKeyPressed=false;
            break;

        default:
            IsXKeyPressed=false;
            IsYKeyPressed=false;
            IsZKeyPressed=false;
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