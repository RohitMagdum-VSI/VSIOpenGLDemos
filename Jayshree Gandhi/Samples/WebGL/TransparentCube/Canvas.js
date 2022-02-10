// onload function
var canvas = null;
var gl = null;
var bFullscreen = false;
var canvas_original_width;
var canvas_original_height;

var angleCube = 0.0;

const WebGLMacros = {
    AMC_ATTRIBUTE_POSITION: 0,
    AMC_ATTRIBUTE_COLOR: 1,
    AMC_ATTRIBUTE_NORMAL: 2,
    AMC_ATTRIBUTE_TEXTURE0: 3
};

var gShaderProgramObject;
var gVertexShaderObject;
var gFragmentShaderObject;

var vao_cube;
var vbo_position_cube;
var vbo_tex_cube;

var mvpUniform;
var resolutionUniform;
var tickUniform;

var perspectiveProjectionMatrix;

var requestAnimationFrame =
    window.requestAnimationFrame ||
    window.webkitRequestAnimationFrame ||
    window.mozRequestAnimationFrame ||
    window.oRequestAnimationFrame ||
    window.msRequestAnimationFrame;

var cancelAnimationFrame =
    window.cancelAnimationFrame ||
    window.webkitCancelRequestAnimationFrame ||
    window.webkitCancelAnimationFrame ||
    window.mozCancelRequestAnimationFrame ||
    window.mozCancelAnimationFrame ||
    window.oCancelRequestAnimationFrame ||
    window.oCancelAnimationFrame ||
    window.msCancelRequestAnimationFrame ||
    window.msCancelAnimationFrame;

function main() {
    //S1: Get canvas
    canvas = document.getElementById("AMC");
    if (!canvas) console.log("Obtaining Canvas failed!\n");
    else console.log("Obtaining Canvas successful!\n");

    canvas_original_width = canvas.width;
    canvas_original_height = canvas.height;

    console.log("Canvas width : " + canvas.width + "Canvas height :" + canvas.height);

    //register keybord and mouse event handler
    window.addEventListener("keydown", keyDown, false);
    window.addEventListener("click", mouseDown, false);
    window.addEventListener("resize", resize, false);

    //initialize WebGL
    init();

    //start drawing and warming up
    resize();
    draw();
}

function toggleFullScreen() {
    var fullscreen_element =
        document.fullscreenElement ||
        document.webkitFullscreenElement ||
        document.mozFullScreenElement ||
        document.msFullscreenElement ||
        null;

    if (fullscreen_element == null) {
        if (canvas.requestFullscreen) canvas.requestFullscreen();
        else if (canvas.mozRequestFullScreen) canvas.mozRequestFullScreen();
        else if (canvas.webkitFullscreen) canvas.webkitFullscreen();
        else if (canvas.msRequestFullscreen) canvas.msRequestFullscreen();
        bFullscreen = true;
    } else {
        if (document.exitFullscreen) document.exitFullscreen();
        else if (document.mozCancelFullScreen) document.mozCancelFullScreen();
        else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
        else if (document.msExitFullscreen) document.msExitFullscreen();
        bFullscreen = false;
    }
}

function init() {
    //S2: Get webgl 2.0 context
    gl = canvas.getContext("webgl2");
    if (!gl) {
        console.log("Failed to get rendering context for WebGL\n");
        return;
    } else console.log("Got rendering context for WebGL!\n");

    //set viewport width and height
    gl.viewportWidth = canvas.width;
    gl.viewportHeight = canvas.height;

    //vertex shader
    var vertexShaderSourceCode =
        "#version 300 es" +
        "\n" +
        "precision mediump float;" +
        "in vec4 vPosition;" +
        "in vec2 vTexCoord;" +
        "uniform mat4 u_mvp_matrix;" +
        "out vec2 out_texCoord;" +
        "out float out_depth;" +
        "void main(void)" +
        "{" +
        "   out_texCoord = vTexCoord;" +
        "   out_depth = (u_mvp_matrix * vPosition).z;" +
        "   gl_Position = u_mvp_matrix * vPosition;" +
        "}";

    gVertexShaderObject = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(gVertexShaderObject, vertexShaderSourceCode);
    gl.compileShader(gVertexShaderObject);
    if (gl.getShaderParameter(gVertexShaderObject, gl.COMPILE_STATUS) == false) {
        var error = gl.getShaderInfoLog(gVertexShaderObject);
        if (error.length > 0) {
            alert("\nVertex Shader compilation log : " + error);
            uninitialize();
        }
    }

    //fragment shader
    var fragmentShaderSourceCode =
        "#version 300 es" +
        "\n" +
        "precision mediump float;" +
        "const float PI2 = 6.28318530718;" +
        "vec4 radialRainbow(vec2 st, float tick)" +
        "{" +
        "   vec2 toCenter = vec2(0.5) - st;" +
        "   float angle = mod((atan(toCenter.y, toCenter.x) / PI2) + 0.5 + sin(tick), 1.0);" +
        "   vec4 a = vec4(0.15, 0.58, 0.96, 1.0);" +
        "   vec4 b = vec4(0.29, 1.00, 0.55, 1.0);" +
        "   vec4 c = vec4(1.00, 0.0, 0.85, 1.0);" +
        "   vec4 d = vec4(0.92, 0.20, 0.14, 1.0);" +
        "   vec4 e = vec4(1.0, 0.96, 0.32, 1.0);" +
        "   float step = 1.0 / 10.0;" +
        "   vec4 color = a;" +
        "   color = mix(color, b, smoothstep(step * 1.0, step * 2.0, angle));" +
        "   color = mix(color, a, smoothstep(step * 2.0, step * 3.0, angle));" +
        "   color = mix(color, b, smoothstep(step * 3.0, step * 4.0, angle));" +
        "   color = mix(color, c, smoothstep(step * 4.0, step * 5.0, angle));" +
        "   color = mix(color, d, smoothstep(step * 5.0, step * 6.0, angle));" +
        "   color = mix(color, c, smoothstep(step * 6.0, step * 7.0, angle));" +
        "   color = mix(color, d, smoothstep(step * 7.0, step * 8.0, angle));" +
        "   color = mix(color, e, smoothstep(step * 8.0, step * 9.0, angle));" +
        "   color = mix(color, a, smoothstep(step * 9.0, step * 10.0, angle));" +
        "   return color;" +
        "}" +
        "float borders(vec2 uv, float strokeWidth)" +
        "{" +
        "   vec2 borderBottomLeft = smoothstep(vec2(0.0), vec2(strokeWidth), uv);" +
        "   vec2 borderTopRight = smoothstep(vec2(0.0), vec2(strokeWidth), 1.0 - uv);" +
        "   return 1.0 - borderBottomLeft.x * borderBottomLeft.y * borderTopRight.x * borderTopRight.y;" +
        "}" +
        "in vec2 out_texCoord;" +
        "in float out_depth;" +
        "uniform vec2 u_resolution;" +
        "uniform float u_tick;" +
        "out vec4 FragColor;" +
        "void main(void)" +
        "{" +
        "   vec2 st = gl_FragCoord.xy / u_resolution;" +
        "   vec4 bordersColor = radialRainbow(st, u_tick);" +
        "   float depth = clamp(smoothstep(-1.0, 1.0, out_depth), 0.6, 0.9);" +
        "   bordersColor *= vec4(borders(out_texCoord, 0.011)) * depth;" +
        "   FragColor = bordersColor;" +
        "}";

    gFragmentShaderObject = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(gFragmentShaderObject, fragmentShaderSourceCode);
    gl.compileShader(gFragmentShaderObject);
    if (gl.getShaderParameter(gFragmentShaderObject, gl.COMPILE_STATUS) == false) {
        var error = gl.getShaderInfoLog(gFragmentShaderObject);
        if (error.length > 0) {
            alert("\nFragment Shader compilation log : " + error);
            uninitialize();
        }
    }

    //shader program
    gShaderProgramObject = gl.createProgram();
    gl.attachShader(gShaderProgramObject, gVertexShaderObject);
    gl.attachShader(gShaderProgramObject, gFragmentShaderObject);

    //pre-linking
    gl.bindAttribLocation(gShaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
    gl.bindAttribLocation(gShaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_TEXTURE0, "vTexCoord");

    //linking
    gl.linkProgram(gShaderProgramObject);
    if (gl.getProgramParameter(gShaderProgramObject, gl.LINK_STATUS) == false) {
        var error = gl.getProgramInfoLog(gShaderProgramObject);
        if (error.length > 0) {
            alert("\nProgram linking log : " + error);
            uninitialize();
        }
    }

    //get uniform location
    mvpUniform = gl.getUniformLocation(gShaderProgramObject, "u_mvp_matrix");
    resolutionUniform = gl.getUniformLocation(gShaderProgramObject, "u_resolution");
    tickUniform = gl.getUniformLocation(gShaderProgramObject, "u_tick");

    //vertices, color, texture, vao, vbo , shader attribs

    var cubeVertices = new Float32Array([
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0
    ]);

    /*var cubeColor = new Float32Array([
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0
    ]);
    */

    var cubeTexCoord = new Float32Array([
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0
    ]);

    //create vao

    //rectangle
    vao_cube = gl.createVertexArray();
    gl.bindVertexArray(vao_cube);

    //create vbo
    vbo_position_cube = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo_position_cube);
    gl.bufferData(gl.ARRAY_BUFFER, cubeVertices, gl.STATIC_DRAW);
    gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    vbo_tex_cube = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo_tex_cube);
    gl.bufferData(gl.ARRAY_BUFFER, cubeTexCoord, gl.STATIC_DRAW);
    gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_TEXTURE0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_TEXTURE0);

    //unbind vbo and vao
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindVertexArray(null);

    //depth
    gl.enable(gl.DEPTH_TEST);

    gl.enable(gl.CULL_FACE);

    //set clear color
    gl.clearColor(0.0, 0.0, 0.0, 1.0);

    perspectiveProjectionMatrix = mat4.create();
}

function resize() {
    if (bFullscreen == true) {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    } else {
        canvas.width = canvas_original_width;
        canvas.height = canvas_original_height;
    }

    gl.viewport(0, 0, canvas.width, canvas.height);

    //perspective
    mat4.perspective(
        perspectiveProjectionMatrix,
        45.0,
        parseFloat(canvas.width) / parseFloat(canvas.height),
        0.1,
        100.0
    );
}

function draw() {
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(gShaderProgramObject);

    //matrices
    var modelViewMatrix = mat4.create();
    var modelViewProjectionMatrix = mat4.create();

    //triangle
    //identity
    mat4.identity(modelViewMatrix);
    mat4.identity(modelViewProjectionMatrix);

    //transformations
    mat4.translate(modelViewMatrix, modelViewMatrix, [0.0, 0.0, -3.0]);
    mat4.rotateX(modelViewMatrix, modelViewMatrix, degToRad(angleCube));
    mat4.rotateY(modelViewMatrix, modelViewMatrix, degToRad(angleCube));
    mat4.rotateZ(modelViewMatrix, modelViewMatrix, degToRad(angleCube));
    mat4.scale(modelViewMatrix, modelViewMatrix, [0.75, 0.75, 0.75]);
    mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

    gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);
    gl.uniform2f(resolutionUniform, canvas.width, canvas.height);
    gl.uniform1f(tickUniform, 1.0);

    gl.cullFace(gl.BACK);
    gl.bindVertexArray(vao_cube);

    gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
    gl.drawArrays(gl.TRIANGLE_FAN, 4, 4);
    gl.drawArrays(gl.TRIANGLE_FAN, 8, 4);
    gl.drawArrays(gl.TRIANGLE_FAN, 12, 4);
    gl.drawArrays(gl.TRIANGLE_FAN, 16, 4);
    gl.drawArrays(gl.TRIANGLE_FAN, 20, 4);

    gl.bindVertexArray(null);

    gl.cullFace(gl.FRONT);
    gl.bindVertexArray(vao_cube);

    gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
    gl.drawArrays(gl.TRIANGLE_FAN, 4, 4);
    gl.drawArrays(gl.TRIANGLE_FAN, 8, 4);
    gl.drawArrays(gl.TRIANGLE_FAN, 12, 4);
    gl.drawArrays(gl.TRIANGLE_FAN, 16, 4);
    gl.drawArrays(gl.TRIANGLE_FAN, 20, 4);

    gl.bindVertexArray(null);

    gl.useProgram(null);

    //animation loop
    update();
    requestAnimationFrame(draw, canvas);
}

function update() {
    angleCube = angleCube - 1.0;
    if (angleCube <= -360.0) {
        angleCube = 0.0;
    }
}

function uninitialize() {
    if (vao_cube) {
        gl.deleteVertexArray(vao_cube);
        vao_cube = null;
    }

    if (vbo_position_cube) {
        gl.deleteBuffer(vbo_position_cube);
        vbo = null;
    }

    if (vbo_tex_cube) {
        gl.deleteBuffer(vbo_tex_cube);
        vbo_tex_cube = null;
    }

    if (gShaderProgramObject) {
        if (gFragmentShaderObject) {
            gl.detachShader(gShaderProgramObject, gFragmentShaderObject);
            gl.deleteShader(gFragmentShaderObject);
            gFragmentShaderObject = null;
        }
        if (gVertexShaderObject) {
            gl.detachShader(gShaderProgramObject, gVertexShaderObject);
            gl.deleteShader(gVertexShaderObject);
            gVertexShaderObject = null;
        }

        gl.deleteProgram(gShaderProgramObject);
        gShaderProgramObject = null;
    }
}

function keyDown(event) {
    switch (event.keyCode) {
        case 27:
            uninitialize();
            window.close();
            break;

        case 70:
            toggleFullScreen();
            break;
    }
}

function mouseDown() {}

function degToRad(degrees) {
    var d = Math.PI / 180;
    return degrees * d;
}
