// global variables
var canvas = null;
var gl = null; //  webgl context
var bFullscreen = false;
var canvas_original_width;
var canvas_original_height;

const WebGLMacros = {
    VDG_ATTRIBUTE_VERTEX: 0,
    VDG_ATTRIBUTE_COLOR: 1,
    VDG_ATTRIBUTE_NORMAL: 2,
    VDG_ATTRIBUTE_TEXTURE0: 3
};

var vertexShaderObject;
var fragmentShaderObject;
var shaderProgramObject;

var vao_rect;
var vbo_position_rect;
var vbo_texture_rect;
var mvpUniform;
var texture0_samplerUniform;

var perspectiveProjectionMatrix;

var angleCube = 0.0;

var texImage;
var textureCanvas;

// To start animation : To have requestAnimationFram() to be called "cross browser" compatible
var requestAnimationFrame =
    window.requestAnimationFrame ||
    window.webkitRequestAnimationFrame ||
    window.mozRequestAnimationFrame ||
    window.oRequestAnimationFrame ||
    window.msRequestAnimationFrame;

// To stop animation : To have cancelAnimationFrame() to be called "cross browser" compatible
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

// onload function
function main() {
    // get <canvas> element
    canvas = document.getElementById("AMC");
    if (!canvas) console.log("Obtaining Canvas Failed\n");
    else console.log("Obtaining Canvas Succeeded\n");
    canvas_original_width = canvas.width;
    canvas_original_height = canvas.height;

    // register keyboard's keydown event handler
    window.addEventListener("keydown", keyDown, false);
    window.addEventListener("click", mouseDown, false);
    window.addEventListener("resize", resize, false);

    // initialize WebGL
    init();

    // start drawing here as warming up
    resize();
    draw();
}

function toggleFullScreen() {
    // code
    var fullscreen_element =
        document.fullscreenElement ||
        document.webkitFullscreenElement ||
        document.mozFullScreenElement ||
        document.msFullscreenElement ||
        null;

    // if not fullscreen
    if (fullscreen_element == null) {
        if (canvas.requestFullscreen) canvas.requestFullscreen();
        else if (canvas.mozRequestFullScreen) canvas.mozRequestFullScreen();
        else if (canvas.webkitRequestFullscreen) canvas.webkitRequestFullscreen();
        else if (canvas.msRequestFullscreen) canvas.msRequestFullscreen();
        bFullscreen = true;
    } // if already fullscreen
    else {
        if (document.exitFullscreen) document.exitFullscreen();
        else if (document.mozCancelFullScreen) document.mozCancelFullScreen();
        else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
        else if (document.msExitFullscreen) document.msExitFullscreen();
        bFullscreen = false;
    }
}

function init() {
    // code
    // get WebGL 2.0 Context
    gl = canvas.getContext("webgl2");
    if (gl == null) {
        // failed to get context
        console.log("Failed to get the rendering context for WebGL");
        return;
    }
    gl.viewportWidth = canvas.width;
    gl.viewportHeight = canvas.height;

    // vertex shader
    var vertexShaderSourceCode =
        "#version 300 es" +
        "\n" +
        "in vec4 vPosition;" +
        "in vec2 vTexture0_Coord;" +
        "uniform mat4 u_mvp_matrix;" +
        "out vec2 out_texture0_coord;" +
        "void main(void)" +
        "{" +
        "gl_Position = u_mvp_matrix * vPosition;" +
        "out_texture0_coord = vTexture0_Coord;" +
        "}";

    vertexShaderObject = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShaderObject, vertexShaderSourceCode);
    gl.compileShader(vertexShaderObject);
    if (gl.getShaderParameter(vertexShaderObject, gl.COMPILE_STATUS) == false) {
        var error = gl.getShaderInfoLog(vertexShaderObject);
        if (error.length > 0) {
            alert(error);
            uninitialize();
        }
    }
    // fragment shader
    var fragmentShaderSourceCode =
        "#version 300 es" +
        "\n" +
        "precision highp float;" +
        "in vec2 out_texture0_coord;" +
        "uniform highp sampler2D u_texture0_sampler;" +
        "out vec4 FragColor;" +
        "void main(void)" +
        "{" +
        "FragColor = texture(u_texture0_sampler, out_texture0_coord);" +
        "}";

    fragmentShaderObject = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShaderObject, fragmentShaderSourceCode);
    gl.compileShader(fragmentShaderObject);
    if (gl.getShaderParameter(fragmentShaderObject, gl.COMPILE_STATUS) == false) {
        var error = gl.getShaderInfoLog(fragmentShaderObject);
        if (error.length > 0) {
            alert(error);
            uninitialize();
        }
    }

    // shader program
    shaderProgramObject = gl.createProgram();
    gl.attachShader(shaderProgramObject, vertexShaderObject);
    gl.attachShader(shaderProgramObject, fragmentShaderObject);

    // pe-linking binding of the shader program object with vertex shader attributes
    gl.bindAttribLocation(shaderProgramObject, WebGLMacros.VDG_ATTRIBUTE_VERTEX, "vPosition");
    gl.bindAttribLocation(shaderProgramObject, WebGLMacros.VDG_ATTRIBUTE_TEXTURE0, "vTexture0_Coord");

    // linking
    gl.linkProgram(shaderProgramObject);
    if (!gl.getProgramParameter(shaderProgramObject, gl.LINK_STATUS)) {
        var error = gl.getProgramInfoLog(shaderProgramObject);
        if (error.length > 0) {
            alert(error);
            uninitialize();
        }
    }

    // get MVP uniform loction
    mvpUniform = gl.getUniformLocation(shaderProgramObject, "u_mvp_matrix");
    texture0_samplerUniform = gl.getUniformLocation(shaderProgramObject, "u_texture0_sampler");

    // load cube text texture
    //text
    textureCanvas = createTextCanvas("AMC");

    texImage = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texImage);
    //gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, textureCanvas);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindTexture(gl.TEXTURE_2D, null);

    // *** vertices, colors, shader attribs, vbo, va initializations ***
    var rectangleVertices = new Float32Array([1.0, -1.0, 0.0, -1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
    var rectangleTexCoords = new Float32Array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);

    vao_rect = gl.createVertexArray();
    gl.bindVertexArray(vao_rect);

    vbo_position_rect = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo_position_rect);
    gl.bufferData(gl.ARRAY_BUFFER, rectangleVertices, gl.STATIC_DRAW);
    gl.vertexAttribPointer(WebGLMacros.VDG_ATTRIBUTE_VERTEX, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(WebGLMacros.VDG_ATTRIBUTE_VERTEX);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    vbo_texture_rect = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo_texture_rect);
    gl.bufferData(gl.ARRAY_BUFFER, rectangleTexCoords, gl.STATIC_DRAW);
    gl.vertexAttribPointer(WebGLMacros.VDG_ATTRIBUTE_TEXTURE0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(WebGLMacros.VDG_ATTRIBUTE_TEXTURE0);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    gl.bindVertexArray(null);

    // set clear color
    gl.clearColor(0.5, 0.5, 0.5, 1.0);

    // initialize projection matrix
    perspectiveProjectionMatrix = mat4.create();

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.TEXTURE_2D);

    //gl.enable(gl.CULL_FACE);
}

function resize() {
    // code
    if (bFullscreen == true) {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    } else {
        canvas.width = canvas_original_width;
        canvas.height = canvas_original_height;
    }

    // set the viewport to match
    gl.viewport(0, 0, canvas.width, canvas.height);

    mat4.perspective(
        perspectiveProjectionMatrix,
        45.0,
        parseFloat(canvas.width) / parseFloat(canvas.height),
        0.1,
        100.0
    );
}

function draw() {
    // code
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(shaderProgramObject);

    var modelViewMatrix = mat4.create();
    var modelViewProjectionMatrix = mat4.create();

    mat4.identity(modelViewMatrix);
    mat4.identity(modelViewProjectionMatrix);

    mat4.translate(modelViewMatrix, modelViewMatrix, [0.0, 0.0, -5.0]);
    /*mat4.rotateX(modelViewMatrix, modelViewMatrix, degToRad(angleCube));
    mat4.rotateY(modelViewMatrix, modelViewMatrix, degToRad(angleCube));
    mat4.rotateZ(modelViewMatrix, modelViewMatrix, degToRad(angleCube));*/
    mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

    gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texImage);
    gl.uniform1i(texture0_samplerUniform, 0);

    gl.bindVertexArray(vao_rect);
    gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);

    gl.bindVertexArray(null);

    gl.bindTexture(gl.TEXTURE_2D, null);

    gl.useProgram(null);

    // animation loop
    requestAnimationFrame(draw, canvas);
}

function uninitalize() {
    // code
    if (texImage) {
        gl.deleteTexture(texImage);
    }
    if (vao_rect) {
        gl.deleteVertexArray(vao_rect);
        vao_rect = null;
    }
    if (vbo_position_rect) {
        gl.deleteBuffer(vbo_position_rect);
        vbo_position_rect = null;
    }
    if (vbo_texture_rect) {
        gl.deleteBuffer(vbo_texture_rect);
        vbo_texture_rect = null;
    }
    if (shaderProgramObject) {
        if (fragmentShaderObject) {
            gl.detachShader(shaderProgramObject, fragmentShaderObject);
            gl.deleteShader(fragmentShaderObject);
            fragmentShaderObject = null;
        }
        if (vertexShaderObject) {
            gl.detachShader(shaderProgramObject, vertexShaderObject);
            gl.deleteShader(vertexShaderObject);
            vertexShaderObject = null;
        }
        gl.deleteProgram(shaderProgramObject);
        shaderProgramObject = null;
    }
}

function keyDown(event) {
    switch (event.keyCode) {
        case 27: // ESCAPE key
            uninitalize();
            window.close();
            break;
        case 70:
            toggleFullScreen();
            break;
    }
}

function mouseDown() {
    // code
}

function createTextCanvas(s) {
    const textCanvas = document.createElement("canvas");
    textCanvas.style.display = "none";
    const textCtx = textCanvas.getContext("2d");

    let fontSize = 156;
    textCtx.font = fontSize + "px monospace";
    const textmetrics = textCtx.measureText(s);

    let tWidth = textmetrics.width;
    let tHeight = fontSize;

    textCanvas.width = tWidth;
    textCanvas.height = tHeight;
    textCanvas.style.width = tWidth + "px";
    textCanvas.style.height = tHeight + "px";
    textCtx.font = fontSize + "px monospace";
    textCtx.textAlign = "center";
    textCtx.textBaseline = "middle";

    //textCtx.fillStyle = "";
    textCtx.fillRect(0, 0, textCtx.canvas.width, textCtx.canvas.height);

    textCtx.fillStyle = "green";
    textCtx.fillText(s, tWidth / 2, tHeight / 2);

    return textCanvas;
}
