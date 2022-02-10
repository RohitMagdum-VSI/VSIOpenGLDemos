package com.rohit_r_jadhav.interleaved;

//For OpenGL
import android.opengl.GLSurfaceView;
import android.opengl.GLES32;

import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.EGLConfig;

import android.opengl.Matrix;

//For glBufferData
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

//For Event
import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnGestureListener;
import android.view.GestureDetector.OnDoubleTapListener;
import android.content.Context;

//For Bitmap
import android.graphics.BitmapFactory;
import android.graphics.Bitmap;
import android.opengl.GLUtils;


public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnGestureListener, OnDoubleTapListener{

	//For Gesture
	private GestureDetector gestureDetector_RRJ;
	private Context context_RRJ;

	//For Shader
	private int vertexShaderObject_RRJ;
	private int fragmentShaderObject_RRJ;
	private int shaderProgramObject_RRJ;

	//For Projection
	private float perspectiveProjectionMatrix_RRJ[] = new float[4 * 4];

	//For Uniform
        private int modelMatrix_Uniform_RRJ;
        private int viewMatrix_Uniform_RRJ;
        private int projectionMatrix_Uniform_RRJ;

	//For Light Uniform
        private int la_Uniform_RRJ;
        private int ld_Uniform_RRJ;
        private int ls_Uniform_RRJ;
        private int lightPosition_Uniform_RRJ;
        private int LKeyPress_Uniform_RRJ;

        private int ka_Uniform_RRJ;
        private int kd_Uniform_RRJ;
        private int ks_Uniform_RRJ;
        private int shininess_Uniform_RRJ;



	//For Lights
        private float lightAmbient_RRJ[] = new float[]{0.250f, 0.250f, 0.250f};
        private float lightDiffuse_RRJ[] = new float[]{1.0f, 1.0f, 1.0f};
        private float lightSpecular_RRJ[] = new float[]{1.0f, 1.0f, 1.0f};
        private float lightPosition_RRJ[] = new float[]{100.0f, 100.0f, 100.0f, 1.0f};
        private boolean bLights_RRJ = false;


	//For Material
        private float materialAmbient_RRJ[] =new float[] {0.250f, 0.250f, 0.250f};
        private float materialDiffuse_RRJ[] = new float[]{1.0f, 1.0f, 1.0f};
        private float materialSpecular_RRJ[] = new float[] {1.0f, 1.0f, 1.0f};
        private float materialShininess_RRJ = 128.0f;


	//For Texture
        private int samplerUniform_RRJ;
        private int textureMarble_RRJ[] = new int[1]; 



	//For Cube
	private int vao_Cube_RRJ[] = new int[1];
	private int vbo_Cube_Position_RRJ[] = new int[1];
	private float angle_Cube_RRJ = 360.0f;



	GLESView(Context drawingContext){
		super(drawingContext);

		context_RRJ = drawingContext;

		gestureDetector_RRJ = new GestureDetector(drawingContext, this, null, false);
		gestureDetector_RRJ.setOnDoubleTapListener(this);

		setEGLContextClientVersion(3);
		setRenderer(this);
		setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
	}

	@Override
	public boolean onTouchEvent(MotionEvent event){
		int eventaction = event.getAction();
		if(!gestureDetector_RRJ.onTouchEvent(event))
			super.onTouchEvent(event);
		return(true);
	}


	/********** Methods in OnDoubleTapListener Interface **********/
	@Override
	public boolean onDoubleTap(MotionEvent e){
		return(true);
	}

	@Override
	public boolean onDoubleTapEvent(MotionEvent e){
		return(true);
	}

	@Override
	public boolean onSingleTapConfirmed(MotionEvent e){

		if(bLights_RRJ == false)
			bLights_RRJ = true;
		else
			bLights_RRJ = false;

		return(true);
	}


	/********** Methods in OnGestureListener Interface **********/
	@Override
	public boolean onDown(MotionEvent e){
		return(true);
	}

	@Override
	public boolean onSingleTapUp(MotionEvent e){
		return(true);
	}

	@Override
	public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY){
		return(true);
	}

	@Override
	public void onLongPress(MotionEvent e){

	}

	@Override
	public void onShowPress(MotionEvent e){

	}

	@Override
	public boolean onScroll(MotionEvent e1, MotionEvent e2, float distanceX, float distanceY){
		uninitialize();
		System.exit(0);
		return(true);
	}



	/********** Methods in GLSurfaceView.Renderer Interface **********/
	@Override
	public void onSurfaceCreated(GL10 gl, EGLConfig config){
		String version_RRJ = gl.glGetString(GL10.GL_VERSION);
		String glsl_version_RRJ = gl.glGetString(GLES32.GL_SHADING_LANGUAGE_VERSION);
		String vendor_RRJ = gl.glGetString(GL10.GL_VENDOR);
		String renderer_RRJ = gl.glGetString(GL10.GL_RENDERER);

		System.out.println("RTR: OpenGL-ES Version: " + version_RRJ);
		System.out.println("RTR: OpenGLSL Version: " + glsl_version_RRJ);
		System.out.println("RTR: Vendor: " + vendor_RRJ);
		System.out.println("RTR: Renderer: " + renderer_RRJ);

		initialize();
	}

	@Override
	public void onSurfaceChanged(GL10 unused, int width, int height){
		resize(width, height);
	}

	@Override
	public void onDrawFrame(GL10 unused){
		update();
		display();
	}



	/********** NOW OpenGL **********/

	private void initialize(){

		System.out.println("RTR: In Initialize()");

		/********** Vertex Shader **********/
		vertexShaderObject_RRJ = GLES32.glCreateShader(GLES32.GL_VERTEX_SHADER);

		final String vertexShaderSourceCode_RRJ = String.format(
				"#version 320 es" +
				"\n" +
				"in vec4 vPosition;" +
				"in vec3 vNormal;" +
				"in vec4 vColor;" +
				"in vec2 vTex;" +

				"out vec4 outColor;" +
				"out vec2 outTex;" + 


				"uniform vec4 u_light_position;" +
		 
				"uniform mat4 u_model_matrix;" +
				"uniform mat4 u_view_matrix;" +
				"uniform mat4 u_projection_matrix;" +

				"out vec3 outViewer;" + 
				"out vec3 outLightDirection;" +
				"out vec3 outNormal;" + 		

				"void main() {" +

						


					"vec3 normalizeNormals;" + 
					"normalizeNormals = vNormal;" +
					"normalizeNormals = normalize(normalizeNormals);" +


					"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" +

					"outLightDirection = vec3(u_light_position - eyeCoordinate);" +

					"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" +
					"outNormal = vec3(normalMatrix * normalizeNormals);" +

					"outViewer = vec3(-eyeCoordinate.xyz);" +
						
					
					"outColor = vColor;" +
					"outTex = vTex;" +
					"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" +
					
				"}"
			);

		GLES32.glShaderSource(vertexShaderObject_RRJ, vertexShaderSourceCode_RRJ);

		GLES32.glCompileShader(vertexShaderObject_RRJ);

		int iCompileStatus_RRJ[] = new int[1];
		int iInfoLogLength_RRJ[] = new int[1];
		String szInfoLog_RRJ = null;

		GLES32.glGetShaderiv(vertexShaderObject_RRJ, GLES32.GL_COMPILE_STATUS, iCompileStatus_RRJ, 0);
		if(iCompileStatus_RRJ[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(vertexShaderObject_RRJ, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength_RRJ, 0);
			if(iInfoLogLength_RRJ[0] > 0){
				szInfoLog_RRJ = GLES32.glGetShaderInfoLog(vertexShaderObject_RRJ);
				System.out.println("RTR: Vertex Shader Compilation Error: " + szInfoLog_RRJ);
				szInfoLog_RRJ = null;
				uninitialize();
				System.exit(0);
			}
		}

		System.out.println("RTR: All VertexShader Operations Done!!");


		/********** Fragment Shader **********/
		fragmentShaderObject_RRJ = GLES32.glCreateShader(GLES32.GL_FRAGMENT_SHADER);

		final String fragmentShaderSourceCode_RRJ = String.format(
				"#version 320 es" +
				"\n" +
				"precision highp float;" +
				"precision highp sampler2D;" +

				"in vec3 outLightDirection;" +
				"in vec3 outNormal;" +
				"in vec3 outViewer;" +

				"in vec4 outColor;" +
				"in vec2 outTex;" +


				"uniform vec3 u_la;" +
				"uniform vec3 u_ld;" +
				"uniform vec3 u_ls;" +

				
				"uniform vec3 u_ka;" +
				"uniform vec3 u_kd;" +
				"uniform vec3 u_ks;" +
				"uniform float u_shininess;" +

				"uniform int u_LKey;" +

				"uniform sampler2D u_sampler;" +


				"out vec4 FragColor;" +
				"void main(){" +

					"vec3 PhongLight;" +

					"if(u_LKey == 1){" +

						"vec3 normalizeLightDirection = normalize(outLightDirection);" +
						"vec3 normalizeNormalVector = normalize(outNormal);" +
						"float S_Dot_N = max(dot(normalizeLightDirection, normalizeNormalVector), 0.0);" +

						"vec3 normalizeViewer = normalize(outViewer);" +
						"vec3 reflection = reflect(-normalizeLightDirection, normalizeNormalVector);" +
						"float R_Dot_V = max(dot(reflection, normalizeViewer), 0.0);" +

						"vec3 ambient = u_la * u_ka;" +
						"vec3 diffuse = u_ld * u_kd * S_Dot_N;" +
						"vec3 specular = u_ls * u_ks * pow(R_Dot_V, u_shininess);" +
						"PhongLight = ambient + diffuse + specular;" +

					"}" +
					"else {" + 
						"PhongLight = vec3(1.0f, 1.0f, 1.0);" +
					"}" +

					"vec4 tex = texture(u_sampler, outTex);" +
					"vec4 light = vec4(PhongLight, 1.0);" +
					
					"FragColor = tex * outColor * light;" +
				"}"
			);
		
		GLES32.glShaderSource(fragmentShaderObject_RRJ, fragmentShaderSourceCode_RRJ);

		GLES32.glCompileShader(fragmentShaderObject_RRJ);

		iCompileStatus_RRJ[0] = 0;
		iInfoLogLength_RRJ[0] = 0;
		szInfoLog_RRJ = null;

		GLES32.glGetShaderiv(fragmentShaderObject_RRJ, GLES32.GL_COMPILE_STATUS, iCompileStatus_RRJ, 0);
		if(iCompileStatus_RRJ[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(fragmentShaderObject_RRJ, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength_RRJ, 0);
			if(iInfoLogLength_RRJ[0] > 0){
				szInfoLog_RRJ = GLES32.glGetShaderInfoLog(fragmentShaderObject_RRJ);
				System.out.println("RTR: Fragment Shader Compilation Error: " + szInfoLog_RRJ);
				szInfoLog_RRJ = null;
				uninitialize();
				System.exit(0);
			}
		}

		System.out.println("RTR: All FragmentShader Operations Done!!");


		/********** Program Object **********/
		shaderProgramObject_RRJ = GLES32.glCreateProgram();

		GLES32.glAttachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
		GLES32.glAttachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);

		GLES32.glBindAttribLocation(shaderProgramObject_RRJ, GLESMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
		GLES32.glBindAttribLocation(shaderProgramObject_RRJ, GLESMacros.AMC_ATTRIBUTE_COLOR, "vColor");
		GLES32.glBindAttribLocation(shaderProgramObject_RRJ, GLESMacros.AMC_ATTRIBUTE_NORMAL, "vNormal");
		GLES32.glBindAttribLocation(shaderProgramObject_RRJ, GLESMacros.AMC_ATTRIBUTE_TEXCOORD0, "vTex");

		GLES32.glLinkProgram(shaderProgramObject_RRJ);

		int iProgramLinkStatus_RRJ[] = new int[1];
		iInfoLogLength_RRJ[0] = 0;
		szInfoLog_RRJ = null;

		GLES32.glGetProgramiv(shaderProgramObject_RRJ, GLES32.GL_LINK_STATUS, iProgramLinkStatus_RRJ, 0);
		if(iProgramLinkStatus_RRJ[0] == GLES32.GL_FALSE){
			GLES32.glGetProgramiv(shaderProgramObject_RRJ, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength_RRJ, 0);
			if(iInfoLogLength_RRJ[0] > 0){
				 szInfoLog_RRJ = GLES32.glGetProgramInfoLog(shaderProgramObject_RRJ);
				 System.out.println("RTR: Program Linking Error: "+ szInfoLog_RRJ);
				 szInfoLog_RRJ = null;
				 uninitialize();
				 System.exit(0);
			}
		}

		
		modelMatrix_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_model_matrix");
		viewMatrix_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_view_matrix");
		projectionMatrix_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_projection_matrix");

		la_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_la");
		ld_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_ld");
		ls_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_ls");
		lightPosition_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_light_position");
		

		ka_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_ka");
		kd_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_kd");
		ks_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_ks");
		shininess_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_shininess");

		LKeyPress_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_LKey");

		samplerUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_sampler");



		System.out.println("RTR: All ShaderProgram Operations Done!!");


		/********** Positions **********/
		
		float Cube_Position_RRJ[] = new float[]{


						//vPosition 		//vColor 			//vNormal 			//vTex
						1.0f, 1.0f, -1.0f,		1.0f, 0.0f, 0.0f,		0.0f, 1.0f, 0.0f,		1.0f, 1.0f,	   
						-1.0f, 1.0f, -1.0f,	1.0f, 0.0f, 0.0f,		0.0f, 1.0f, 0.0f,		0.0f, 1.0f, 	 
						-1.0f, 1.0f, 1.0f,		1.0f, 0.0f, 0.0f,		0.0f, 1.0f, 0.0f,		0.0f, 0.0f,	
						1.0f, 1.0f, 1.0f,		1.0f, 0.0f, 0.0f,		0.0f, 1.0f, 0.0f,		1.0f, 0.0f,
						//Bottom
						1.0f, -1.0f, -1.0f,	0.0f, 1.0f, 0.0f,		0.0f, -1.0f, 0.0f,		1.0f, 1.0f,
						-1.0f, -1.0f, -1.0f,	0.0f, 1.0f, 0.0f,		0.0f, -1.0f, 0.0f, 	0.0f, 1.0f,	
						-1.0f, -1.0f, 1.0f,	0.0f, 1.0f, 0.0f,		0.0f, -1.0f, 0.0f,		0.0f, 0.0f,
						1.0f, -1.0f, 1.0f,		0.0f, 1.0f, 0.0f,		0.0f, -1.0f, 0.0f,		1.0f, 0.0f,
						//Front
						1.0f, 1.0f, 1.0f,		0.0f, 0.0f, 1.0f,		0.0f, 0.0f, 1.0f,		1.0f, 1.0f,
						-1.0f, 1.0f, 1.0f,		0.0f, 0.0f, 1.0f,		0.0f, 0.0f, 1.0f,		0.0f, 1.0f,
						-1.0f, -1.0f, 1.0f,	0.0f, 0.0f, 1.0f,		0.0f, 0.0f, 1.0f,		0.0f, 0.0f,
						1.0f, -1.0f, 1.0f,		0.0f, 0.0f, 1.0f,		0.0f, 0.0f, 1.0f,		1.0f, 0.0f,
						//Back
						1.0f, 1.0f, -1.0f,		1.0f, 1.0f, 0.0f,		0.0f, 0.0f, -1.0f,		1.0f, 1.0f,
						-1.0f, 1.0f, -1.0f,	1.0f, 1.0f, 0.0f,		0.0f, 0.0f, -1.0f,		0.0f, 1.0f,
						-1.0f, -1.0f, -1.0f,	1.0f, 1.0f, 0.0f,		0.0f, 0.0f, -1.0f,		0.0f, 0.0f,
						1.0f, -1.0f, -1.0f,	1.0f, 1.0f, 0.0f,		0.0f, 0.0f, -1.0f,		1.0f, 0.0f,
						//Right
						1.0f, 1.0f, -1.0f,		0.0f, 1.0f, 1.0f,		1.0f, 0.0f, 0.0f,		1.0f, 1.0f,
						1.0f, 1.0f, 1.0f,		0.0f, 1.0f, 1.0f,		1.0f, 0.0f, 0.0f,		0.0f, 1.0f,
						1.0f, -1.0f, 1.0f,		0.0f, 1.0f, 1.0f,		1.0f, 0.0f, 0.0f,		0.0f, 0.0f,
						1.0f, -1.0f, -1.0f,	0.0f, 1.0f, 1.0f,		1.0f, 0.0f, 0.0f,		1.0f, 0.0f,
						//Left
						-1.0f, 1.0f, 1.0f,		1.0f, 0.0f, 1.0f,		-1.0f, 0.0f, 0.0f,		1.0f, 1.0f,
						-1.0f, 1.0f, -1.0f,	1.0f, 0.0f, 1.0f,		-1.0f, 0.0f, 0.0f,		0.0f, 1.0f,
						-1.0f, -1.0f, -1.0f,	1.0f, 0.0f, 1.0f,		-1.0f, 0.0f, 0.0f, 	0.0f, 0.0f,
						-1.0f, -1.0f, 1.0f,	1.0f, 0.0f, 1.0f,		-1.0f, 0.0f, 0.0f,		1.0f, 0.0f,
					
					};



	
		/********** Cube **********/
		GLES32.glGenVertexArrays(1, vao_Cube_RRJ, 0);
		GLES32.glBindVertexArray(vao_Cube_RRJ[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Cube_Position_RRJ, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Cube_Position_RRJ[0]);

				/********** For glBufferData **********/
				ByteBuffer Cube_positionByteBuffer_RRJ = ByteBuffer.allocateDirect(Cube_Position_RRJ.length * 4);
				Cube_positionByteBuffer_RRJ.order(ByteOrder.nativeOrder());
				FloatBuffer Cube_positionBuffer_RRJ = Cube_positionByteBuffer_RRJ.asFloatBuffer();
				Cube_positionBuffer_RRJ.put(Cube_Position_RRJ);
				Cube_positionBuffer_RRJ.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							Cube_Position_RRJ.length * 4,
							Cube_positionBuffer_RRJ,
							GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									11 * 4,  0 * 4);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_COLOR,
									3,
									GLES32.GL_FLOAT,
									false,
									11 * 4,  3 * 4);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_NORMAL,
									3,
									GLES32.GL_FLOAT,
									false,
									11 * 4,  6 * 4);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_TEXCOORD0,
									2,
									GLES32.GL_FLOAT,
									false,
									11 * 4,  9 * 4);

			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_COLOR);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_NORMAL);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_TEXCOORD0);


			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


			

		GLES32.glBindVertexArray(0);


		GLES32.glEnable(GLES32.GL_TEXTURE_2D);
		textureMarble_RRJ[0] = loadTexture(R.raw.marble);


		GLES32.glEnable(GLES32.GL_DEPTH_TEST);
		GLES32.glDepthFunc(GLES32.GL_LEQUAL);
		//GLES32.glClearDepth(1.0f);

		GLES32.glDisable(GLES32.GL_CULL_FACE);

		GLES32.glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	}



	private int loadTexture(int imageFileResourceID){
		int texture_RRJ[] = new int[1];

		BitmapFactory.Options options_RRJ = new BitmapFactory.Options();

		options_RRJ.inScaled = false;

		Bitmap bitmap_RRJ = BitmapFactory.decodeResource(context_RRJ.getResources(), imageFileResourceID, options_RRJ);

		GLES32.glPixelStorei(GLES32.GL_UNPACK_ALIGNMENT, 4);
		GLES32.glGenTextures(1, texture_RRJ, 0);
		GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, texture_RRJ[0]);

			GLES32.glTexParameteri(GLES32.GL_TEXTURE_2D, GLES32.GL_TEXTURE_MAG_FILTER, GLES32.GL_LINEAR);
			GLES32.glTexParameteri(GLES32.GL_TEXTURE_2D, GLES32.GL_TEXTURE_MIN_FILTER, GLES32.GL_LINEAR_MIPMAP_LINEAR);

			GLUtils.texImage2D(GLES32.GL_TEXTURE_2D,
							0,
							bitmap_RRJ,
							0);
			GLES32.glGenerateMipmap(GLES32.GL_TEXTURE_2D);

		GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, 0);
		return(texture_RRJ[0]);

	}



	private void uninitialize(){

		
		if(textureMarble_RRJ[0] != 0){
			GLES32.glDeleteTextures(1, textureMarble_RRJ, 0);
			textureMarble_RRJ[0] = 0;
		}

		if(vbo_Cube_Position_RRJ[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Cube_Position_RRJ, 0);
			vbo_Cube_Position_RRJ[0] = 0;
		}

		if(vao_Cube_RRJ[0] != 0){
			GLES32.glDeleteVertexArrays(1, vao_Cube_RRJ, 0);
			vao_Cube_RRJ[0] = 0;
		}

		

		if(shaderProgramObject_RRJ != 0){

			GLES32.glUseProgram(shaderProgramObject_RRJ);

				if(fragmentShaderObject_RRJ != 0){
					GLES32.glDetachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);
					GLES32.glDeleteShader(fragmentShaderObject_RRJ);
					fragmentShaderObject_RRJ = 0;
					System.out.println("RTR: Fragment Shader Detached and Deleted!!");
				}

				if(vertexShaderObject_RRJ != 0){
					GLES32.glDetachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
					GLES32.glDeleteShader(vertexShaderObject_RRJ);
					vertexShaderObject_RRJ = 0;
					System.out.println("RTR: Vertex Shader Detached and Deleted!!");
				}
				
			GLES32.glUseProgram(0);
			GLES32.glDeleteProgram(shaderProgramObject_RRJ);
			shaderProgramObject_RRJ = 0;
		}

		/*if(shaderProgramObject_RRJ != 0){

			int iShaderCount[] = new int[1];
			int iShaderNo;

			GLES32.glUseProgram(shaderProgramObject_RRJ);
				GLES32.glGetProgramiv(shaderProgramObject_RRJ, GLES32.GL_ATTACHED_SHADERS, iShaderCount, 0);
				System.out.println("RTR: ShaderCount: "+ iShaderCount[0]);
				int iShaders[] = new int[iShaderCount[0]];
				GLES32.glGetAttachedShaders(shaderProgramObject_RRJ, iShaderCount[0],
										iShaderCount, 0,
										iShaders, 0);

				for(iShaderNo = 0; iShaderNo < iShaderCount[0] ; iShaderNo++){
					GLES32.glDetachShader(shaderProgramObject_RRJ, iShaders[iShaderNo]);
					GLES32.glDeleteShader(iShaders[iShaderNo]);
					iShaders[iShaderNo] = 0;
					System.out.println("RTR: Shader Deleted!!");
				}
			GLES32.glUseProgram(0);
			GLES32.glDeleteProgram(shaderProgramObject_RRJ);
			shaderProgramObject_RRJ = 0;
		}*/
	}


	private void resize(int width, int height){
		if(height == 0)
			height = 1;

		GLES32.glViewport(0, 0, width, height);

		Matrix.setIdentityM(perspectiveProjectionMatrix_RRJ, 0);
		Matrix.perspectiveM(perspectiveProjectionMatrix_RRJ, 0,
						45.0f,
						(float)width / (float)height,
						0.1f,
						100.0f);
	}


	private void display(){

		float translateMatrix_RRJ[] = new float[4 * 4];
		float rotateMatrix_RRJ[] = new float[4 * 4];
		float scaleMatrix_RRJ[] = new float[4 * 4];
		float modelMatrix_RRJ[] = new float[4*4];
		float viewMatrix_RRJ[] = new float[4*4];

		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);

		GLES32.glUseProgram(shaderProgramObject_RRJ);

			


			/********** Cube **********/
			Matrix.setIdentityM(translateMatrix_RRJ, 0);
			Matrix.setIdentityM(rotateMatrix_RRJ, 0);
			Matrix.setIdentityM(scaleMatrix_RRJ, 0);
			Matrix.setIdentityM(modelMatrix_RRJ, 0);
			Matrix.setIdentityM(viewMatrix_RRJ, 0);


			Matrix.translateM(translateMatrix_RRJ, 0, 0.0f, 0.0f, -4.50f);
			Matrix.rotateM(rotateMatrix_RRJ, 0, angle_Cube_RRJ, 1.0f, 0.0f, 0.0f);
			Matrix.rotateM(rotateMatrix_RRJ, 0, angle_Cube_RRJ, 0.0f, 1.0f, 0.0f);
			Matrix.rotateM(rotateMatrix_RRJ, 0, angle_Cube_RRJ, 0.0f, 0.0f, 1.0f);
			Matrix.scaleM(scaleMatrix_RRJ, 0, 0.9f, 0.9f, 0.9f);
			Matrix.multiplyMM(modelMatrix_RRJ, 0, modelMatrix_RRJ, 0, translateMatrix_RRJ, 0);
			Matrix.multiplyMM(modelMatrix_RRJ, 0, modelMatrix_RRJ, 0, scaleMatrix_RRJ, 0);
			Matrix.multiplyMM(modelMatrix_RRJ, 0, modelMatrix_RRJ, 0, rotateMatrix_RRJ, 0);
			

			GLES32.glUniformMatrix4fv(modelMatrix_Uniform_RRJ, 1, false, modelMatrix_RRJ, 0);
			GLES32.glUniformMatrix4fv(viewMatrix_Uniform_RRJ, 1, false, viewMatrix_RRJ, 0);
			GLES32.glUniformMatrix4fv(projectionMatrix_Uniform_RRJ, 1, false, perspectiveProjectionMatrix_RRJ, 0);


			if(bLights_RRJ == true){
				GLES32.glUniform1i(LKeyPress_Uniform_RRJ, 1);

				GLES32.glUniform3fv(la_Uniform_RRJ, 1, lightAmbient_RRJ, 0);
				GLES32.glUniform3fv(ld_Uniform_RRJ, 1, lightDiffuse_RRJ, 0);
				GLES32.glUniform3fv(ls_Uniform_RRJ, 1, lightSpecular_RRJ, 0);
				GLES32.glUniform4fv(lightPosition_Uniform_RRJ, 1, lightPosition_RRJ, 0);

				GLES32.glUniform3fv(ka_Uniform_RRJ, 1, materialAmbient_RRJ, 0);
				GLES32.glUniform3fv(kd_Uniform_RRJ, 1, materialDiffuse_RRJ, 0);
				GLES32.glUniform3fv(ks_Uniform_RRJ, 1, materialSpecular_RRJ, 0);
				GLES32.glUniform1f(shininess_Uniform_RRJ, materialShininess_RRJ);	

			}
			else
				GLES32.glUniform1i(LKeyPress_Uniform_RRJ, 0);


			GLES32.glActiveTexture(GLES32.GL_TEXTURE0);
			GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, textureMarble_RRJ[0]);
			GLES32.glUniform1i(samplerUniform_RRJ, 0);

			
			GLES32.glBindVertexArray(vao_Cube_RRJ[0]);
				GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 0, 4);
				GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 4, 4);
				GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 8, 4);
				GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 12, 4);
				GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 16, 4);
				GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 20, 4);
			GLES32.glBindVertexArray(0);
			
		GLES32.glUseProgram(0);

		requestRender();
	}


	private void update(){
		angle_Cube_RRJ = angle_Cube_RRJ - 1.50f;
		if(angle_Cube_RRJ < 0.0f)
			angle_Cube_RRJ = 360.0f;
	}
};
