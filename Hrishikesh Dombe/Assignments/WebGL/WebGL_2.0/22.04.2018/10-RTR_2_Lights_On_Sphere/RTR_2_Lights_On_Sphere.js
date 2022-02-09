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

var light_ambient_Blue=[0.0,0.0,0.0];
var light_diffuse_Blue=[0.0,0.0,1.0];
var light_specular_Blue=[0.0,0.0,1.0];
var light_position_Blue=[100.0,100.0,100.0,1.0];

var light_ambient_Red=[0.0,0.0,0.0];
var light_diffuse_Red=[1.0,0.0,0.0];
var light_specular_Red=[1.0,0.0,0.0];
var light_position_Red=[-100.0,100.0,100.0,1.0];

var material_ambient=[0.0,0.0,0.0];
var material_diffuse=[1.0,1.0,1.0];
var material_specular=[1.0,1.0,1.0];
var material_shininess=50.0;

var sphere=null;

var IsLKeyPressed=false;
var IsAKeyPressed=false;
var IsTKeyPressed=false;

var modelMatrixUniform,viewMatrixUniform, projectionMatrixUniform;
var laUniform_Red,ldUniform_Red,lsUniform_Red,lightPositionUniform_Red;
var laUniform_Blue,ldUniform_Blue,lsUniform_Blue,lightPositionUniform_Blue;
var kaUniform,kdUniform,ksUniform,materialShininessUniform;
var lKeyPressedUniform,toggleshaderUniform;

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
    "uniform mediump int toggleshader;"+
    "uniform vec4 u_light_position_Red;"+
    "uniform vec3 u_La_Red;"+
    "uniform vec3 u_Ld_Red;"+
    "uniform vec3 u_Ls_Red;"+
    "uniform vec4 u_light_position_Blue;"+
    "uniform vec3 u_La_Blue;"+
    "uniform vec3 u_Ld_Blue;"+
    "uniform vec3 u_Ls_Blue;"+
    "uniform vec3 u_Ka;"+
    "uniform vec3 u_Kd;"+
    "uniform vec3 u_Ks;"+
    "uniform float u_material_shininess;"+
    "out vec3 transformed_normals;"+
    "out vec3 light_direction_Red;"+
    "out vec3 light_direction_Blue;"+
    "out vec3 viewer_vector;"+
    "out vec3 phong_ads_color_vertex;"+
    "void main(void)"+
    "{"+
    "if(u_lKeyPressed == 1)"+
    "{"+
    "vec4 eye_coordinates = u_view_matrix * u_model_matrix * vPosition;"+
    "transformed_normals = mat3(u_view_matrix * u_model_matrix) * vNormal;"+
    "light_direction_Red = vec3(u_light_position_Red) - eye_coordinates.xyz;"+
    "light_direction_Blue = vec3(u_light_position_Blue) - eye_coordinates.xyz;"+
    "viewer_vector = -eye_coordinates.xyz;"+
    "if(toggleshader == 1)" +
    "{" +
    "vec3 normalized_transformed_normals = normalize(transformed_normals);"+
    "vec3 normalized_viewer_vector = normalize(viewer_vector);"+
    "vec3 normalized_light_direction_Red = normalize(light_direction_Red);"+
    "vec3 normalized_light_direction_Blue = normalize(light_direction_Blue);"+
    "float tn_dot_ld_Red = max(dot(normalized_transformed_normals,normalized_light_direction_Red),0.0);"+
    "float tn_dot_ld_Blue = max(dot(normalized_transformed_normals,normalized_light_direction_Blue),0.0);"+
    "vec3 ambient = u_La_Red * u_Ka + u_La_Blue * u_Ka;"+
    "vec3 diffuse = u_Ld_Red * u_Kd * tn_dot_ld_Red + u_Ld_Blue * u_Kd * tn_dot_ld_Blue;"+
    "vec3 reflection_vector_Red = reflect(-normalized_light_direction_Red,normalized_transformed_normals);"+
    "vec3 reflection_vector_Blue = reflect(-normalized_light_direction_Blue,normalized_transformed_normals);"+
    "vec3 specular = u_Ls_Red * u_Ks * pow(max(dot(reflection_vector_Red,normalized_viewer_vector),0.0),u_material_shininess) + u_Ls_Blue * u_Ks * pow(max(dot(reflection_vector_Blue,normalized_viewer_vector),0.0),u_material_shininess);"+
    "phong_ads_color_vertex = ambient + diffuse + specular;"+
    "}" +
    "}"+
    "else"+
    "{"+
    "phong_ads_color_vertex = vec3(1.0,1.0,1.0);"+
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
    "in vec3 light_direction_Red;"+
    "in vec3 light_direction_Blue;"+
    "in vec3 viewer_vector;"+
    "in vec3 phong_ads_color_vertex;"+
    "out vec4 FragColor;"+
    "uniform vec3 u_La_Red;"+
    "uniform vec3 u_Ld_Red;"+
    "uniform vec3 u_Ls_Red;"+
    "uniform vec3 u_La_Blue;"+
    "uniform vec3 u_Ld_Blue;"+
    "uniform vec3 u_Ls_Blue;"+
    "uniform vec3 u_Ka;"+
    "uniform vec3 u_Kd;"+
    "uniform vec3 u_Ks;"+
    "uniform float u_material_shininess;"+
    "uniform int u_lKeyPressed;"+
    "uniform mediump int toggleshader;"+
    "void main(void)"+
    "{"+
    "vec3 phong_ads_color;"+
    "if(u_lKeyPressed == 1)"+
    "{"+
    "if(toggleshader == 0)"+
    "{"+
    "vec3 normalized_transformed_normals=normalize(transformed_normals);"+
    "vec3 normalized_light_direction_Red = normalize(light_direction_Red);"+
    "vec3 normalized_light_direction_Blue = normalize(light_direction_Blue);"+
    "vec3 normalized_viewer_vector = normalize(viewer_vector);"+
    "vec3 ambient = u_La_Red * u_Ka + u_La_Blue * u_Ka;"+
    "float tn_dot_ld_Red = max(dot(normalized_transformed_normals,normalized_light_direction_Red),0.0);"+
    "float tn_dot_ld_Blue = max(dot(normalized_transformed_normals,normalized_light_direction_Blue),0.0);"+
    "vec3 diffuse = u_Ld_Red * u_Kd * tn_dot_ld_Red + u_Ld_Blue * u_Kd * tn_dot_ld_Blue;"+
    "vec3 reflection_vector_Red = reflect(-normalized_light_direction_Red,normalized_transformed_normals);"+
    "vec3 reflection_vector_Blue = reflect(-normalized_light_direction_Blue,normalized_transformed_normals);"+
    "vec3 specular = u_Ls_Red * u_Ks * pow(max(dot(reflection_vector_Red,normalized_viewer_vector),0.0),u_material_shininess) +  u_Ls_Blue * u_Ks * pow(max(dot(reflection_vector_Blue,normalized_viewer_vector),0.0),u_material_shininess);"+
    "phong_ads_color=ambient+diffuse+specular;"+
    "}"+
    "else"+
    "{"+
    "phong_ads_color = phong_ads_color_vertex;"+
    "}"+
    "}"+
    "else"+
    "{"+
    "phong_ads_color=phong_ads_color_vertex;"+
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

    laUniform_Red=gl.getUniformLocation(shaderProgramObject,"u_La_Red");

    ldUniform_Red=gl.getUniformLocation(shaderProgramObject,"u_Ld_Red");

    lsUniform_Red=gl.getUniformLocation(shaderProgramObject,"u_Ls_Red");

    lightPositionUniform_Red=gl.getUniformLocation(shaderProgramObject,"u_light_position_Red");
    
    laUniform_Blue=gl.getUniformLocation(shaderProgramObject,"u_La_Blue");

    ldUniform_Blue=gl.getUniformLocation(shaderProgramObject,"u_Ld_Blue");

    lsUniform_Blue=gl.getUniformLocation(shaderProgramObject,"u_Ls_Blue");

    lightPositionUniform_Blue=gl.getUniformLocation(shaderProgramObject,"u_light_position_Blue");

    kaUniform=gl.getUniformLocation(shaderProgramObject,"u_Ka");

    kdUniform=gl.getUniformLocation(shaderProgramObject,"u_Kd");

    ksUniform=gl.getUniformLocation(shaderProgramObject,"u_Ks");

    materialShininessUniform = gl.getUniformLocation(shaderProgramObject,"u_material_shininess");

    toggleshaderUniform = gl.getUniformLocation(shaderProgramObject,"toggleshader");

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

        gl.uniform3fv(laUniform_Red,light_ambient_Red);
        gl.uniform3fv(ldUniform_Red,light_diffuse_Red);
        gl.uniform3fv(lsUniform_Red,light_specular_Red);
        gl.uniform4fv(lightPositionUniform_Red,light_position_Red);

        gl.uniform3fv(laUniform_Blue,light_ambient_Blue);
        gl.uniform3fv(ldUniform_Blue,light_diffuse_Blue);
        gl.uniform3fv(lsUniform_Blue,light_specular_Blue);
        gl.uniform4fv(lightPositionUniform_Blue,light_position_Blue);

        gl.uniform3fv(kaUniform,material_ambient);
        gl.uniform3fv(kdUniform,material_diffuse);
        gl.uniform3fv(ksUniform,material_specular);
        gl.uniform1f(materialShininessUniform,material_shininess);

        if(IsTKeyPressed==true)
            gl.uniform1i(toggleshaderUniform,1);
        else
            gl.uniform1i(toggleshaderUniform,0);
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

        case 65://A or a
            if(IsAKeyPressed == false)
                IsAKeyPressed=true;
            else
                IsAKeyPressed=false;
            break;

        case 84://T or t
            if(IsTKeyPressed == false)
                IsTKeyPressed=true;
            else
                IsTKeyPressed=false;
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