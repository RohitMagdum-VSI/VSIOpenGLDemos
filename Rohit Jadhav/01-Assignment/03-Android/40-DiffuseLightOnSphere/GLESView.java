package com.rohit_r_jadhav.diffuselight_on_sphere;

import android.opengl.GLSurfaceView;
import android.opengl.GLES32;
import android.opengl.Matrix;

import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.EGLConfig;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;

import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnDoubleTapListener;
import android.view.GestureDetector.OnGestureListener;
import android.content.Context;

public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnDoubleTapListener, OnGestureListener{

	GestureDetector gestureDetector;
	Context context;
	boolean bLights = false;

	GLESView(Context drawingContext){
		super(drawingContext);
		context = drawingContext;

		gestureDetector = new GestureDetector(drawingContext, this, null, false);
		gestureDetector.setOnDoubleTapListener(this);

		setEGLContextClientVersion(3);
		setRenderer(this);
		setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
	}

	@Override
	public boolean onTouchEvent(MotionEvent e){
		int action = e.getAction();
		if(!gestureDetector.onTouchEvent(e))
			super.onTouchEvent(e);
		return(true);
	}

	/********** Methods from OnDoubleTapListener ***********/
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
		if(bLights == false)
			bLights = true;
		else
			bLights = false;
		return(true);
	}


	/************ Methods from OnGestureListener **********/
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
		System.exit(1);
		return(true);
	}



	/********** Methods from GLSurfaceView.Renderer ***********/
	@Override
	public void onSurfaceCreated(GL10 gl, EGLConfig config){
		String version = gl.glGetString(GL10.GL_VERSION);
		String glsl_version = gl.glGetString(GLES32.GL_SHADING_LANGUAGE_VERSION);
		String renderer = gl.glGetString(GL10.GL_RENDERER);
		String vendor = gl.glGetString(GL10.GL_VENDOR);

		System.out.println("RTR: OpenGL Version: " + version);
		System.out.println("RTR: OpenGLSL Version: " + glsl_version);
		System.out.println("RTR: Renderer: " + renderer);
		System.out.println("RTR: Vendor: " + vendor);

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



	//For Sphere
	private int vao_Sphere[] = new int[1];
	private int vbo_Sphere_Position[] = new int[1];
	private int vbo_Sphere_Normal[] = new int[1];
	private int vbo_Sphere_Element[] = new int[1];
	private float angle_Sphere = 0.0f;
	private int numVertices;
	private int numElements;


	//For Shader
	private int vertexShaderObject;
	private int fragmentShaderObject;
	private int shaderProgramObject;

	//For Uniform
	private int mvUniform;
	private int projectionMatrixUniform;
	private int ld_Uniform;
	private int kd_Uniform;
	private int lightPositionUniform;
	private int LKeyPressUniform;

	//For Projection
	private float perspectiveProjectionMatrix[] = new float[4*4];


	private void initialize(){


		/********** Vertex Shader **********/
		vertexShaderObject = GLES32.glCreateShader(GLES32.GL_VERTEX_SHADER);

		final String vertexShaderSourceCode = String.format(
				"#version 320 es" +
				"\n" +
				"in vec4 vPosition;" +
				"in vec3 vNormal;" +
				"uniform mat4 u_mv_matrix;" +
				"uniform mat4 u_projection_matrix;" +
				"uniform vec3 u_ld;" +
				"uniform vec3 u_kd;" +
				"uniform vec4 u_light_position;" +
				"uniform int u_LKeyPress;" +
				"out vec3 diffuseColor;" +
				"void main() {" +
					"if(u_LKeyPress == 1){" +
						"vec4 eyeCoordinate = u_mv_matrix * vPosition;" +
						"vec3 source = normalize(vec3(u_light_position - eyeCoordinate));" +
						"mat3 normalMatrix = mat3(u_mv_matrix);" +
						"vec3 normal = normalize(vec3(normalMatrix * vNormal));" +
						"float S_Dot_N = max(dot(source, normal), 0.0);" +
						"diffuseColor = u_ld * u_kd * S_Dot_N;" +
					"}" +
					"gl_Position = u_projection_matrix * u_mv_matrix * vPosition;" +
				"}" 
			);

		GLES32.glShaderSource(vertexShaderObject, vertexShaderSourceCode);

		GLES32.glCompileShader(vertexShaderObject);

		int iCompileStatus[] = new int[1];
		int iInfoLogLength[] = new int[1];
		String szInfoLog = null;

		GLES32.glGetShaderiv(vertexShaderObject, GLES32.GL_COMPILE_STATUS, iCompileStatus, 0);
		if(iCompileStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(vertexShaderObject, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				szInfoLog = GLES32.glGetShaderInfoLog(vertexShaderObject);
				System.out.println("RTR: Vertex Shader Compilation Error: " + szInfoLog);
				szInfoLog = null;
				uninitialize();
				System.exit(0);
			}
		}



		/********** Fragment Shader **********/
		fragmentShaderObject = GLES32.glCreateShader(GLES32.GL_FRAGMENT_SHADER);

		final String fragmentShaderSourceCode = String.format(
				"#version 320 es" +
				"\n" +
				"precision highp float;" +
				"precision highp int;" +
				"uniform int u_LKeyPress;" +
				"in vec3 diffuseColor;" +
				"out vec4 FragColor;" +
				"void main(void)" +
				"{" +
					"if(u_LKeyPress == 1){" +
						"FragColor = vec4(diffuseColor, 1.0);" +
					"}" +
					"else{" +
						"FragColor = vec4(1.0, 1.0, 1.0, 1.0);" +
					"}" +
				"}"
			);
		
		GLES32.glShaderSource(fragmentShaderObject, fragmentShaderSourceCode);

		GLES32.glCompileShader(fragmentShaderObject);

		iCompileStatus[0] = 0;
		iInfoLogLength[0] = 0;
		szInfoLog = null;

		GLES32.glGetShaderiv(fragmentShaderObject, GLES32.GL_COMPILE_STATUS, iCompileStatus, 0);
		if(iCompileStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(fragmentShaderObject, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				szInfoLog = GLES32.glGetShaderInfoLog(fragmentShaderObject);
				System.out.println("RTR: Fragment Shader Compilation Error: " + szInfoLog);
				szInfoLog = null;
				uninitialize();
				System.exit(0);
			}
		}



		/********** Program Object **********/
		shaderProgramObject = GLES32.glCreateProgram();

		GLES32.glAttachShader(shaderProgramObject, vertexShaderObject);
		GLES32.glAttachShader(shaderProgramObject, fragmentShaderObject);

		GLES32.glBindAttribLocation(shaderProgramObject, GLESMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
		GLES32.glBindAttribLocation(shaderProgramObject, GLESMacros.AMC_ATTRIBUTE_NORMAL, "vNormal");

		GLES32.glLinkProgram(shaderProgramObject);

		int iProgramLinkStatus[] = new int[1];
		iInfoLogLength[0] = 0;
		szInfoLog = null;

		GLES32.glGetProgramiv(shaderProgramObject, GLES32.GL_LINK_STATUS, iProgramLinkStatus, 0);
		if(iProgramLinkStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetProgramiv(shaderProgramObject, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				 szInfoLog = GLES32.glGetProgramInfoLog(shaderProgramObject);
				 System.out.println("RTR: Program Linking Error: "+ szInfoLog);
				 szInfoLog = null;
				 uninitialize();
				 System.exit(0);
			}
		}

		mvUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_mv_matrix");
		projectionMatrixUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_projection_matrix");
		ld_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_ld");
		kd_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_kd");
		lightPositionUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_light_position");
		LKeyPressUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_LKeyPress");


		Sphere sphere = new Sphere();
		float sphere_Position[] = new float[1146];
		float sphere_Normal[] = new float[1146];
		float sphere_TexCoord[] = new float[764];
		short sphere_Element[] = new short[2280];
		


		sphere.getSphereVertexData(sphere_Position, sphere_Normal, sphere_TexCoord, sphere_Element);
		numVertices = sphere.getNumberOfSphereVertices();
		numElements = sphere.getNumberOfSphereElements();


		GLES32.glGenVertexArrays(1, vao_Sphere, 0);
		GLES32.glBindVertexArray(vao_Sphere[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Sphere_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Sphere_Position[0]);

				ByteBuffer spherePosition_ByteBuffer = ByteBuffer.allocateDirect(sphere_Position.length * 4);
				spherePosition_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer spherePosition_FloatBuffer = spherePosition_ByteBuffer.asFloatBuffer();
				spherePosition_FloatBuffer.put(sphere_Position);
				spherePosition_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							sphere_Position.length * 4,
							spherePosition_FloatBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);



			/********** Normal **********/ 
			GLES32.glGenBuffers(1, vbo_Sphere_Normal, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Sphere_Normal[0]);

				ByteBuffer sphereNormal_ByteBuffer = ByteBuffer.allocateDirect(sphere_Normal.length * 4);
				sphereNormal_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer sphereNormal_FloatBuffer = sphereNormal_ByteBuffer.asFloatBuffer();
				sphereNormal_FloatBuffer.put(sphere_Normal);
				sphereNormal_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							sphere_Normal.length * 4,
							sphereNormal_FloatBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_NORMAL,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_NORMAL);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


			/********** Elements **********/
			GLES32.glGenBuffers(1, vbo_Sphere_Element, 0);
			GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element[0]);

				ByteBuffer sphereElement_ByteBuffer = ByteBuffer.allocateDirect(sphere_Element.length * 4);
				sphereElement_ByteBuffer.order(ByteOrder.nativeOrder());
				ShortBuffer sphereElement_ShortBuffer = sphereElement_ByteBuffer.asShortBuffer();
				sphereElement_ShortBuffer.put(sphere_Element);
				sphereElement_ShortBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ELEMENT_ARRAY_BUFFER,
							sphere_Element.length * 4,
							sphereElement_ShortBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER, 0);


		GLES32.glBindVertexArray(0);

		GLES32.glEnable(GLES32.GL_DEPTH_TEST);
		GLES32.glDepthFunc(GLES32.GL_LEQUAL);

		GLES32.glDisable(GLES32.GL_CULL_FACE);

		GLES32.glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	}

	private void uninitialize(){

		if(vbo_Sphere_Element[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Sphere_Element, 0);
			vbo_Sphere_Element[0] = 0;
		}

		if(vbo_Sphere_Normal[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Sphere_Normal, 0);
			vbo_Sphere_Normal[0] = 0;
		}

		if(vbo_Sphere_Position[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Sphere_Position, 0);
			vbo_Sphere_Position[0] = 0;
		}

		if(vao_Sphere[0] != 0){
			GLES32.glDeleteVertexArrays(1, vao_Sphere, 0);
			vao_Sphere[0] = 0;
		}

		if(shaderProgramObject != 0){

			int iShaderCount[] = new int[1];
			int iShaderNo = 0;

			GLES32.glUseProgram(shaderProgramObject);

				/*GLES32.glGetProgramiv(shaderProgramObject, GLES32.GL_ATTACHED_SHADERS, iShaderCount, 0);
				System.out.println("RTR: Shader Count: " + iShaderCount[0]);

				int iShaders[] = new int[iShaderCount[0]];

				GLES32.glGetAttachedShaders(shaderProgramObject, iShaderCount[0], iShaderCount, 0,
										iShaders, 0);

				for(iShaderNo = 0; iShaderNo < iShaderCount[0]; iShaderNo++){
					GLES32.glDetachShader(shaderProgramObject, iShaders[iShaderNo]);
					GLES32.glDeleteShader(iShaders[iShaderNo]);
					iShaders[iShaderNo] = 0;
				}*/

				if(fragmentShaderObject != 0){
					GLES32.glDetachShader(shaderProgramObject, fragmentShaderObject);
					GLES32.glDeleteShader(fragmentShaderObject);
					fragmentShaderObject = 0;
				}

				if(vertexShaderObject != 0){
					GLES32.glDetachShader(shaderProgramObject, vertexShaderObject);
					GLES32.glDeleteShader(vertexShaderObject);
					vertexShaderObject = 0;
				}


			GLES32.glUseProgram(0);
			GLES32.glDeleteProgram(shaderProgramObject);
			shaderProgramObject = 0;
		}
	}

	private void resize(int width, int height){
		if(height == 0)
			height = 1;

		GLES32.glViewport(0, 0, width, height);

		Matrix.setIdentityM(perspectiveProjectionMatrix, 0);
		Matrix.perspectiveM(perspectiveProjectionMatrix, 0,
						45.0f,
						(float)width / (float)height,
						0.1f,
						100.0f);
	}


	private void display(){
		float translateMatrix[] = new float[4 * 4];
		float rotateMatrix[] = new float[4 * 4];
		float scaleMatrix[] = new float[4 * 4];
		float modelViewMatrix[] = new float[4 * 4];
		float modelViewProjectionMatrix[] = new float[4 * 4];

		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);

		GLES32.glUseProgram(shaderProgramObject);


			/********** Cube **********/
			Matrix.setIdentityM(translateMatrix, 0);
			Matrix.setIdentityM(rotateMatrix, 0);
			Matrix.setIdentityM(scaleMatrix, 0);
			Matrix.setIdentityM(modelViewMatrix, 0);
			Matrix.setIdentityM(modelViewProjectionMatrix, 0);

			Matrix.translateM(translateMatrix, 0, 0.0f, 0.0f, -1.50f);
			Matrix.rotateM(rotateMatrix, 0, angle_Sphere, 1.0f, 1.0f, 1.0f);
			Matrix.scaleM(scaleMatrix, 0, 0.9f, 0.9f, 0.9f);
			Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, translateMatrix, 0);
			Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, rotateMatrix, 0);
			Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, scaleMatrix, 0);

			GLES32.glUniformMatrix4fv(mvUniform, 1, false, modelViewMatrix, 0);
			GLES32.glUniformMatrix4fv(projectionMatrixUniform, 1, false, perspectiveProjectionMatrix, 0);

			
			if(bLights == true){
				GLES32.glUniform1i(LKeyPressUniform, 1);
				GLES32.glUniform3f(ld_Uniform, 1.0f, 1.0f, 1.0f);
				GLES32.glUniform3f(kd_Uniform, 0.50f, 0.50f, 0.50f);
				GLES32.glUniform4f(lightPositionUniform, 0.0f, 0.0f, 2.0f, 1.0f);
			}
			else
				GLES32.glUniform1i(LKeyPressUniform, 0);


		GLES32.glBindVertexArray(vao_Sphere[0]);

			GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element[0]);
			GLES32.glDrawElements(GLES32.GL_TRIANGLES, numElements, GLES32.GL_UNSIGNED_SHORT, 0);

		GLES32.glBindVertexArray(0);
			
		GLES32.glUseProgram(0);

		requestRender();
	}


	private void update(){
		angle_Sphere = angle_Sphere - 2.0f;

		if(angle_Sphere < 0.0f)
			angle_Sphere = 360.0f;
	}

}