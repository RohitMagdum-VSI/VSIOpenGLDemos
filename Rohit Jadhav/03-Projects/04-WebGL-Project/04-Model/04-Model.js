var canvas;
var gl;


var gShaderProgramObject_Model_AAW;



var gPerspectiveProjectionMatrix_AAW;



//Shadow Mapping
const DEPTH_TEXTURE_SIZE_AAW = 1024;
var gFrameBuffer_DepthMap_AAW;
var gShadowMapTexture_AAW;


var gModelMatrixUniform_AAW;
var gViewMatrixUniform_AAW;
var gProjectionMatrixUniform_AAW;
var gTextureSamplerUniform_AAW;
var gShadowMatrixUnifom_AAW;
var gChoiceUniform_AAW;
var gShadowChoiceUniform_AAW;



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
var gLightPosition_AAW = [1130.0, 1200.0, 1130.0, 1.0];

var gMaterialAmbient_AAW = [0.0, 0.0, 0.0];
var gMaterialDiffuse_AAW = [0.5, 0.5, 0.5];
var gMaterialSpecular_AAW = [1.0, 1.0, 1.0];
var gMaterialShininess_AAW = 50.0;

var gTreeTexture_AAW;
var gLampTexture_AAW;
var gTextureSampler_AAW;



function initializeShader_Model_AAW(){

    //Variable declarations
    var vertexShaderObject_AAW;
    var fragmentShaderObject_AAW;
    
    //Code
        //Vertex Shader
        var vertexShaderSourceCode = 
        "#version 300 es"+
        "\n"+
        "in         vec4    vPosition;" +
        "in         vec3    vNormal;"+
        "in         vec2    vTexCoord;"+

        "uniform mat4 u_model_matrix;" +
        "uniform mat4 u_view_matrix;" +
        "uniform mat4 u_proj_matrix;" +


        "in         mat4    modelMatrix;"+
        "uniform    int     u_choice;"+
        "uniform    int     u_shadowChoice;" +

        "uniform    mat4    u_shadowMatrix;"+
        "uniform    vec4    u_lightPosition;"+

        "out        vec3    out_transformedNormals;"+
        "out        vec3    out_lightDirection;"+
        "out        vec3    out_viewVector;"+
        "out        vec4    out_shadowCoord;"+
        "out        vec2    out_vTexCoord;"+
        "flat out int out_u_shadowChoice;" +

        //For Fog
        "out float out_fogCoord;" +


        "void main(void){" +
        
            "mat4 m4ModelMatrix;" +

            "if(u_choice == 0){" +
                "m4ModelMatrix = u_model_matrix;" +
            "}" +

            "else if(u_choice == 1){" +
                "m4ModelMatrix = modelMatrix;" +
            "}" +

            "vec4 eyeCoordinates     = u_view_matrix * m4ModelMatrix * vPosition;"+

            //"if(u_shadowChoice == 1){" +
                
                "out_transformedNormals  = mat3(u_view_matrix * m4ModelMatrix) * vNormal;"+
                "out_lightDirection      = vec3(u_lightPosition - eyeCoordinates);"+
                "out_viewVector          = -eyeCoordinates.xyz;"+
                "out_shadowCoord         = u_shadowMatrix * m4ModelMatrix * vPosition;"+
            //"}" +

            "out_u_shadowChoice = u_shadowChoice;" +

            "out_fogCoord = abs(eyeCoordinates.z);" +

            "gl_Position = u_proj_matrix * u_view_matrix * m4ModelMatrix * vPosition;" +

            "out_vTexCoord = vTexCoord;"+
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
        "flat in    int             out_u_shadowChoice;" +
        "in         vec3            out_transformedNormals;"+
        "in         vec3            out_lightDirection;"+
        "in         vec3            out_viewVector;"+
        "in         vec4            out_shadowCoord;"+
        "in         vec2            out_vTexCoord;"+
        "uniform    highp           sampler2DShadow     u_depthTextureSampler;"+
        "uniform    highp           sampler2D   u_textureSampler;"+
        "uniform    vec3            u_la;"+
        "uniform    vec3            u_ld;"+
        "uniform    vec3            u_ls;"+
        "uniform    vec3            u_ka;"+
        "uniform    vec3            u_kd;"+
        "uniform    vec3            u_ks;"+
        "uniform    float           u_materialShininess;"+
        "uniform    int             u_lightKeyPressed;"+
        

        //For Fog
        "in float out_fogCoord;" +


        "out        vec4            FragColor;" +
        

        "void main(void){" +

            "if(out_u_shadowChoice == 0){" +

                // // *** For Fog ***
                // "float f = exp(-0.001 * out_fogCoord);" +

                // "vec3 color = mix(vec3(1.0, 1.0f, 1.0f), vec3(0.0f, 0.5f, 0.0f), f);" +

                // "FragColor = vec4(color, 1.0);" +


                "   vec3 Fong_ADS_Light = vec3(0.0f);"+
                "   vec3 normalizedTransformedNormals   = normalize(out_transformedNormals);"+
                "   vec3 normalizedLightDirection       = normalize(out_lightDirection);"+
                "   vec3 normalizedViewVector           = normalize(out_viewVector);"+
                "   vec3 reflectionVector               = reflect(-normalizedLightDirection, normalizedTransformedNormals);"+
                "   vec3 ambient    = u_la * u_ka;"+
                "   vec3 diffuse    = u_ld * u_kd * max(dot(normalizedLightDirection, normalizedTransformedNormals), 0.0f);"+
                "   vec3 specular   = u_ls * u_ks * pow(max(dot(normalizedViewVector, reflectionVector), 0.0f), u_materialShininess);"+
                "   Fong_ADS_Light  = ambient + diffuse;"+
                // "   FragColor = texture(u_textureSampler, out_vTexCoord);"+
                "   FragColor = vec4(diffuse, 1.0) * texture(u_textureSampler, out_vTexCoord);" + 

            "}" +

            "else if(out_u_shadowChoice == 1){" +

                "   vec3 Fong_ADS_Light = vec3(0.0f);"+
                "   vec3 normalizedTransformedNormals   = normalize(out_transformedNormals);"+
                "   vec3 normalizedLightDirection       = normalize(out_lightDirection);"+
                "   vec3 normalizedViewVector           = normalize(out_viewVector);"+
                "   vec3 reflectionVector               = reflect(-normalizedLightDirection, normalizedTransformedNormals);"+
                "   float f          = textureProj(u_depthTextureSampler, out_shadowCoord);"+
                "   if(f < 0.2){"+
                "       f = 0.2;"+
                "   }"+
                "   vec3 ambient    = u_la * u_ka;"+
                "   vec3 diffuse    = u_ld * u_kd * max(dot(normalizedLightDirection, normalizedTransformedNormals), 0.0f);"+
                "   vec3 specular   = u_ls * u_ks * pow(max(dot(normalizedViewVector, reflectionVector), 0.0f), u_materialShininess);"+
                "   Fong_ADS_Light  = ambient + diffuse;"+
                "   FragColor = vec4(diffuse, 1.0) * texture(u_textureSampler, out_vTexCoord);"+

                //  // *** For Fog ***
                // "float fog = exp(-0.001 * out_fogCoord);" +

                // "vec3 color = mix(vec3(1.0, 1.0f, 1.0f), Fong_ADS_Light.xyz, fog);" +

                // "FragColor = vec4(color, 1.0);" +

            "}" +


             "if(out_u_shadowChoice == 2){" +

                "   vec3 Fong_ADS_Light = vec3(0.0f);"+
                "   vec3 normalizedTransformedNormals   = normalize(out_transformedNormals);"+
                "   vec3 normalizedLightDirection       = normalize(out_lightDirection);"+
                "   vec3 normalizedViewVector           = normalize(out_viewVector);"+
                "   vec3 reflectionVector               = reflect(-normalizedLightDirection, normalizedTransformedNormals);"+
                "   vec3 ambient    = u_la * u_ka;"+
                "   vec3 diffuse    = u_ld * u_kd * max(dot(normalizedLightDirection, normalizedTransformedNormals), 0.0f);"+
                "   vec3 specular   = u_ls * u_ks * pow(max(dot(normalizedViewVector, reflectionVector), 0.0f), u_materialShininess);"+
                "   Fong_ADS_Light  = ambient + diffuse;"+
                "   FragColor = texture(u_textureSampler, out_vTexCoord);"+
               

            "}" +


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
    gShaderProgramObject_Model_AAW = gl.createProgram();
    gl.attachShader(gShaderProgramObject_Model_AAW, vertexShaderObject_AAW);
    gl.attachShader(gShaderProgramObject_Model_AAW, fragmentShaderObject_AAW);

    gl.bindAttribLocation(gShaderProgramObject_Model_AAW, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
    gl.bindAttribLocation(gShaderProgramObject_Model_AAW, WebGLMacros.AMC_ATTRIBUTE_NORMAL, "vNormal");
    gl.bindAttribLocation(gShaderProgramObject_Model_AAW, WebGLMacros.AMC_ATTRIBUTE_MODELMATRIX, "modelMatrix");
    gl.bindAttribLocation(gShaderProgramObject_Model_AAW, WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0, "vTexCoord");

    //Program Linking
    gl.linkProgram(gShaderProgramObject_Model_AAW);
    if(gl.getProgramParameter(gShaderProgramObject_Model_AAW, gl.LINK_STATUS) == false){
        var error = gl.getProgramInfoLog(gShaderProgramObject_Model_AAW);
        if(error.length > 0){
            alert(error);
            uninitialize();
        }
    }

    gModelMatrixUniform_AAW = gl.getUniformLocation(gShaderProgramObject_Model_AAW, "u_model_matrix");
    gViewMatrixUniform_AAW = gl.getUniformLocation(gShaderProgramObject_Model_AAW, "u_view_matrix");
    gProjectionMatrixUniform_AAW = gl.getUniformLocation(gShaderProgramObject_Model_AAW, "u_proj_matrix");
    gChoiceUniform_AAW = gl.getUniformLocation(gShaderProgramObject_Model_AAW, "u_choice");
    gShadowChoiceUniform_AAW = gl.getUniformLocation(gShaderProgramObject_Model_AAW, "u_shadowChoice");

    gShadowMatrixUnifom_AAW          = gl.getUniformLocation(gShaderProgramObject_Model_AAW, "u_shadowMatrix");
    gTextureSamplerUniform_AAW       = gl.getUniformLocation(gShaderProgramObject_Model_AAW, "u_depthTextureSampler");
    gLaUniform_AAW                   = gl.getUniformLocation(gShaderProgramObject_Model_AAW, "u_la");
    gLdUniform_AAW                   = gl.getUniformLocation(gShaderProgramObject_Model_AAW, "u_ld");
    gLsUniform_AAW                   = gl.getUniformLocation(gShaderProgramObject_Model_AAW, "u_ls");
    gKaUniform_AAW                   = gl.getUniformLocation(gShaderProgramObject_Model_AAW, "u_ka");
    gKdUniform_AAW                   = gl.getUniformLocation(gShaderProgramObject_Model_AAW, "u_kd");
    gKsUniform_AAW                   = gl.getUniformLocation(gShaderProgramObject_Model_AAW, "u_ks");
    gLightPositionUniform_AAW        = gl.getUniformLocation(gShaderProgramObject_Model_AAW, "u_lightPosition");
    gMaterialShininessUniform_AAW    = gl.getUniformLocation(gShaderProgramObject_Model_AAW, "u_materialShininess");
    gTextureSampler_AAW              = gl.getUniformLocation(gShaderProgramObject_Model_AAW, "u_textureSampler");

   

    initializeDepth_FrameBuffer_AAW();
}



function initializeDepth_FrameBuffer_AAW(){

    //Code
    gFrameBuffer_DepthMap_AAW = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, gFrameBuffer_DepthMap_AAW);

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


function uninitialize_Model(){

    if(gShaderProgramObject_Model_AAW){
        gl.useProgram(gShaderProgramObject_Model_AAW);
        var shaderCount = gl.getProgramParameter(gShaderProgramObject_Model_AAW, gl.GL_ATTACHED_SHADERS);
        var shaders = gl.getAttachedShaders(gShaderProgramObject_Model_AAW);

        for(var i = 0; i < shaderCount; i++){
            gl.detachShader(gShaderProgramObject_Model_AAW, shaders[i]);
            gl.deleteShader(shaders[i]);
            shaders[i] = null;
        }

        gl.useProgram(null);
        shaders = null;

        gl.deleteProgram(gShaderProgramObject_Model_AAW);
        gShaderProgramObject_Model_AAW = null;
    }
}



function Model(){

    let vao;
    let vbo_position;
    let vbo_normal;
    let vbo_texture;
    let vbo_index;
    let vbo_modelMatrix;

    let numElements;

    let finalVertices;
    let finalTexCoords;
    let finalNormals;
    let finalElements;

    let vertices = [];
    let texCoords = [];
    let normals = [];
    let indices = [];

    let gIsInstanced;
    let gInstanceCount;

    let vertexAttributesArray = [];

    this.parseOBJ=function(objSource, isInstanced, instancedModelMatrix, instanceCount){
    
        //Code

        gIsInstanced = isInstanced;
        gInstanceCount = instanceCount;     

        let lines = objSource.split('\n');
        for(let lineNum = 0; lineNum < lines.length; lineNum++){
            if(lines[lineNum].startsWith("v ")){
                vertices.push(lines[lineNum].replace("v ", '').split(' ').map(Number));

                let tempObject = {index: vertexAttributesArray.length,
                                  nextIndex: -1,
                                  isSet: false,
                                  vertexIndex: -1,
                                  textureIndex: -1,
                                  normalIndex: -1};

                vertexAttributesArray.push(tempObject);
            }
            if(lines[lineNum].startsWith("vt ")){
                texCoords.push(lines[lineNum].replace("vt ", '').split(' ').map(Number));
            }
            if(lines[lineNum].startsWith("vn ")){
                normals.push(lines[lineNum].replace("vn ", '').split(' ').map(Number));
            }
            if(lines[lineNum].startsWith("f ")){
                let myLine = lines[lineNum].replace("f ", '').split(' ').map(chunk => chunk.split('/').map(Number));
                for(let i = 0; i < 3; i++){
                    let vi = myLine[i][0] - 1;
                    let ti = myLine[i][1] - 1;
                    let ni = myLine[i][2] - 1;

                    if(vertexAttributesArray[vi].isSet == false){

                        vertexAttributesArray[vi].vertexIndex = vi;
                        vertexAttributesArray[vi].textureIndex = ti;
                        vertexAttributesArray[vi].normalIndex = ni;
                        vertexAttributesArray[vi].isSet = true;

                        indices.push(vi);
                    }
                    else{
                        if(vertexAttributesArray[vi].textureIndex == ti && vertexAttributesArray[vi].normalIndex == ni){
                            indices.push(vi);
                        }
                        else{
                            let isFound = false;

                            while(vertexAttributesArray[vi].nextIndex != -1){
                                isFound = true;
                                indices.push(vertexAttributesArray[vi].index);
                                break;
                            }

                            if(!isFound){
                                let tempObject = {index: vertexAttributesArray.length,
                                                  nextIndex: -1,
                                                  isSet: true,
                                                  vertexIndex: vi,
                                                  textureIndex: ti,
                                                  normalIndex: ni};
                                                  
                                vertexAttributesArray[vi].nextIndex = tempObject.index;

                                vertexAttributesArray.push(tempObject);
                                indices.push(tempObject.index);
                            }
                        }
                    }
                }
            }
        }
        lines = null;
        // console.log(vertices);
        // console.log("##########################################");
        // console.log(texCoords);
        // console.log("##########################################");
        // console.log(normals);
        // console.log("##########################################");
        // console.log(vertexIndices);
        // console.log("##########################################");
        // console.log(texCoordIndices);
        // console.log("##########################################");
        // console.log(normalIndices);
    
    
        finalVertices = new Float32Array(vertexAttributesArray.length * 3);
        finalTexCoords = new Float32Array(vertexAttributesArray.length * 2);
        finalNormals = new Float32Array(vertexAttributesArray.length * 3);
        finalElements = new Uint32Array(indices);
    
        for(let i = 0; i < vertexAttributesArray.length; i++){
            let vi = vertexAttributesArray[i].vertexIndex;
            let ti = vertexAttributesArray[i].textureIndex;
            let ni = vertexAttributesArray[i].normalIndex;
    
            finalVertices[3 * i + 0] = vertices[vi][0];
            finalVertices[3 * i + 1] = vertices[vi][1];
            finalVertices[3 * i + 2] = vertices[vi][2];
    
            finalTexCoords[2 * i + 0] = texCoords[ti][0];
            finalTexCoords[2 * i + 1] = texCoords[ti][1];
    
            finalNormals[3 * i + 0] = normals[ni][0];
            finalNormals[3 * i + 1] = normals[ni][1];
            finalNormals[3 * i + 2] = normals[ni][2];
        }

        numElements = finalElements.length;
        //console.log(finalElements);
    
        //VAO & VBO
        vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        
            vbo_position = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, vbo_position);
                gl.bufferData(gl.ARRAY_BUFFER, finalVertices, gl.STATIC_DRAW);
                gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION, 3, gl.FLOAT, false, 0, 0);
                gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
            gl.bindBuffer(gl.ARRAY_BUFFER, null);
        
            vbo_normal = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, vbo_normal);
                gl.bufferData(gl.ARRAY_BUFFER, finalNormals, gl.STATIC_DRAW);
                gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_NORMAL, 3, gl.FLOAT, false, 0, 0);
                gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_NORMAL);
            gl.bindBuffer(gl.ARRAY_BUFFER, null);
        
            vbo_texture = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, vbo_texture);
                gl.bufferData(gl.ARRAY_BUFFER, finalTexCoords, gl.STATIC_DRAW);
                gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0, 2, gl.FLOAT, false, 0, 0);
                gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0);
            gl.bindBuffer(gl.ARRAY_BUFFER, null);
        
            vbo_index = gl.createBuffer();
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vbo_index);
                gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, finalElements, gl.STATIC_DRAW);
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

            if(gIsInstanced == true)
            {
                vbo_modelMatrix= gl.createBuffer();
                 gl.bindBuffer(gl.ARRAY_BUFFER, vbo_modelMatrix);
        
                 gl.bufferData(gl.ARRAY_BUFFER, instancedModelMatrix, gl.STATIC_DRAW);

                 for (var i = 0; i < 4; i++)
                 {
                    gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_MODELMATRIX + i, 4, gl.FLOAT, false, 64, (16 * i));
                    gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_MODELMATRIX + i);
                    gl.vertexAttribDivisor(WebGLMacros.AMC_ATTRIBUTE_MODELMATRIX + i, 1);
                }

                gl.bindBuffer(gl.ARRAY_BUFFER, null);

            }

        gl.bindVertexArray(null);
        
        cleanUp();
    }
    
    this.drawModel=function(){

        //Code
        gl.bindVertexArray(vao);  

            if(gIsInstanced == true)
            {

                gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vbo_index);
                gl.drawElementsInstanced(gl.TRIANGLES, numElements, gl.UNSIGNED_INT, 0, gInstanceCount);
                gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
            }
            else
            {

                gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vbo_index);
                gl.drawElements(gl.TRIANGLES, numElements, gl.UNSIGNED_INT, 0);
                gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);            

            }      
          gl.bindVertexArray(null);  
    }

    function cleanUp(){

        //Code
        if(finalVertices != null){
            finalVertices = null;
        }
        if(finalTexCoords != null){
            finalTexCoords = null;
        }
        if(finalNormals != null){
            finalNormals = null;
        }
        if(finalElements != null){
            finalElements = null;
        }
        if(vertices != null){
            vertices = null;
        }
        if(texCoords != null){
            texCoords = null;
        }
        if(normals != null){
            normals = null;
        }
        if(indices != null){
            indices = null;
        }
        if(vertexAttributesArray != null){
            vertexAttributesArray = null;
        }
    }

    this.deallocate=function()
    {
        // code
        if(vao){
            gl.deleteVertexArray(vao);
            vao=null;
        }
        
        if(vbo_index){
            gl.deleteBuffer(vbo_index);
            vbo_index=null;
        }
        
        if(vbo_texture){
            gl.deleteBuffer(vbo_texture);
            vbo_texture=null;
        }
        
        if(vbo_normal){
            gl.deleteBuffer(vbo_normal);
            vbo_normal=null;
        }
        
        if(vbo_position){
            gl.deleteBuffer(vbo_position);
            vbo_position=null;
        }

        if(gIsInstanced == true)
        {
            if(vbo_modelMatrix){

                gl.deleteBuffer(vbo_modelMatrix);
                vbo_modelMatrix=null;
            }
        }
    }
}




function drawModel(model, tex, choice){


    var translateMatrix = mat4.create();
    var modelMatrix = mat4.create();


    mat4.identity(modelMatrix);
    mat4.identity(translateMatrix);


    gl.useProgram(gShaderProgramObject_Model_AAW);


        gl.uniform3fv(gLaUniform_AAW, gLightAmbient_AAW);
        gl.uniform3fv(gLdUniform_AAW, gLightDiffuse_AAW);
        gl.uniform3fv(gLsUniform_AAW, gLightSpecular_AAW);
        gl.uniform4fv(gLightPositionUniform_AAW, lightPosition);

        gl.uniform3fv(gKaUniform_AAW, gMaterialAmbient_AAW);
        gl.uniform3fv(gKdUniform_AAW, gMaterialDiffuse_AAW);
        gl.uniform3fv(gKsUniform_AAW, gMaterialSpecular_AAW);
        gl.uniform1f(gMaterialShininessUniform_AAW, gMaterialShininessUniform_AAW);
        
        gl.uniformMatrix4fv(gModelMatrixUniform_AAW, false, modelMatrix);
        gl.uniformMatrix4fv(gViewMatrixUniform_AAW, false, global_viewMatrix);
        gl.uniformMatrix4fv(gProjectionMatrixUniform_AAW, false, gPerspectiveProjectionMatrix);
        gl.uniform1i(gChoiceUniform_AAW, choice);

        if(model == gModel_Lamp)
            gl.uniform1i(gShadowChoiceUniform_AAW, 2);
        else
             gl.uniform1i(gShadowChoiceUniform_AAW, 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.uniform1i(gTextureSampler_AAW, 1);

        model.drawModel();


        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, null);

        
    gl.useProgram(null);

}


// function drawModelWithShadow(model, choice){

//     var translateMatrix = mat4.create();
//     var modelMatrix = mat4.create();

//     var sceneModelMatrix = mat4.create();
//     var sceneViewMatrix = mat4.create();
//     var sceneProjectionMatrix = mat4.create();
//     var shadowMatrix = mat4.create();

//     var scaleBiasMatrix;



//     mat4.identity(modelMatrix);
//     mat4.identity(translateMatrix);


//     var lightViewMatrix = mat4.create();
//     var lightProjectionMatrix = mat4.create();
//     var lightMVPMatrix = mat4.create();

//     // Pass 1
//     gl.bindFramebuffer(gl.FRAMEBUFFER, gFrameBuffer_DepthMap_AAW);
//     gl.viewport(0, 0, DEPTH_TEXTURE_SIZE_AAW, DEPTH_TEXTURE_SIZE_AAW);    


//         mat4.lookAt(lightViewMatrix, gLightPosition_AAW, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
//         mat4.perspective(lightProjectionMatrix, 45.0, parseFloat(canvas.width) / parseFloat(canvas.height), 0.1, 4000.0);

//         mat4.multiply(lightMVPMatrix, lightMVPMatrix, lightProjectionMatrix);
//         mat4.multiply(lightMVPMatrix, lightMVPMatrix, lightViewMatrix);


//         gl.clearDepth(1.0);
//         gl.clear(gl.DEPTH_BUFFER_BIT);
        
//         gl.enable(gl.POLYGON_OFFSET_FILL);
//         gl.polygonOffset(2.0, 4.0);

//         gl.useProgram(gShaderProgramObject_Model_AAW);
            
//             gl.uniformMatrix4fv(gModelMatrixUniform_AAW, false, modelMatrix);
//             gl.uniformMatrix4fv(gViewMatrixUniform_AAW, false, lightViewMatrix);
//             gl.uniformMatrix4fv(gProjectionMatrixUniform_AAW, false, lightProjectionMatrix);
//             gl.uniform1i(gChoiceUniform_AAW, choice);
//             gl.uniform1i(gShadowChoiceUniform_AAW, 0);

//             model.drawModel();

//         gl.useProgram(null);

//         gl.disable(gl.POLYGON_OFFSET_FILL);

//     gl.bindFramebuffer(gl.FRAMEBUFFER, null); 



//     // Pass 2
//     scaleBiasMatrix = mat4.fromValues(0.5, 0.0, 0.0, 0.0,
//                                       0.0, 0.5, 0.0, 0.0,
//                                       0.0, 0.0, 0.5, 0.0,
//                                       0.5, 0.5, 0.5, 1.0);

//     vec3.add(gCameraLookingAt, gCameraPosition, gCameraFront);
//     mat4.lookAt(global_viewMatrix, gCameraPosition, gCameraLookingAt, gCameraUp);

//     mat4.perspective(sceneProjectionMatrix, 45.0, parseFloat(canvas.width) / parseFloat(canvas.height), 0.1, 4000.0);

//     gl.viewport(0, 0, canvas.width, canvas.height);


//     gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    
//     gl.useProgram(gShaderProgramObject_Model_AAW);

//         mat4.multiply(shadowMatrix, shadowMatrix, scaleBiasMatrix);
//         mat4.multiply(shadowMatrix, shadowMatrix, lightProjectionMatrix);
//         mat4.multiply(shadowMatrix, shadowMatrix, lightViewMatrix);

//         gl.uniformMatrix4fv(gModelMatrixUniform_AAW, false, sceneModelMatrix);
//         gl.uniformMatrix4fv(gViewMatrixUniform_AAW, false, global_viewMatrix);
//         gl.uniformMatrix4fv(gProjectionMatrixUniform_AAW, false, sceneProjectionMatrix);
//         gl.uniformMatrix4fv(gShadowMatrixUnifom_AAW, false, shadowMatrix);

//         gl.uniform3fv(gLaUniform_AAW, gLightAmbient_AAW);
//         gl.uniform3fv(gLdUniform_AAW, gLightDiffuse_AAW);
//         gl.uniform3fv(gLsUniform_AAW, gLightSpecular_AAW);
//         gl.uniform4fv(gLightPositionUniform_AAW, gLightPosition_AAW);

//         gl.activeTexture(gl.TEXTURE0);
//         gl.bindTexture(gl.TEXTURE_2D, gShadowMapTexture_AAW);
//         gl.uniform1i(gTextureSamplerUniform_AAW, 0);

//         gl.uniform3fv(gKaUniform_AAW, [0.1, 0.1, 0.1]);
//         gl.uniform3fv(gKdUniform_AAW, [0.1, 0.5, 0.1]);
//         gl.uniform3fv(gKsUniform_AAW, [0.1, 0.1, 0.1]);
//         gl.uniform1f(gMaterialShininessUniform_AAW, 70.0);

//         gl.uniform1i(gChoiceUniform_AAW, choice);
//         gl.uniform1i(gShadowChoiceUniform_AAW, 1);

//         model.drawModel();


//         gl.activeTexture(gl.TEXTURE0);
//         gl.bindTexture(gl.TEXTURE_2D, null);

//     gl.useProgram(null);


// }