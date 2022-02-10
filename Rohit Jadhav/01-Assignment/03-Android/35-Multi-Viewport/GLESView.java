package com.rohit_r_jadhav.multi_viewport;

//For OpenGL
import android.opengl.GLSurfaceView;
import android.opengl.GLES32;

import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.EGLConfig;

//Like Vmath
import android.opengl.Matrix;


//For glBufferData() and FloatBuffer
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

//For Event 
import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnGestureListener;
import android.view.GestureDetector.OnDoubleTapListener;
import android.content.Context;

public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnGestureListener, OnDoubleTapListener{

	private GestureDetector gestureDetector_RRJ;
	private Context context_RRJ;

	//For Shader
	private int vertexShaderObject_RRJ;
	private int fragmentShaderObject_RRJ;
	private int shaderProgramObject_RRJ;
	


	//For Triangle
	private int[] vao_Tri_RRJ = new int[1];
	private int[] vbo_Tri_Position_RRJ = new int[1];
	private int[] vbo_Tri_Color_RRJ = new int[1];

	//For Projection
	private float perspectiveProjectionMatrix_RRJ[] = new float[4 * 4];


	//For Uniform
	private int mvpUniform_RRJ;
	

	//For Viewport
	private int iViewportNo_RRJ;
	private int viewPortWidth_RRJ;
	private int viewPortHeight_RRJ;

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


	/********** Methods from OnDoubleTapListener Interface **********/
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

		iViewportNo_RRJ = iViewportNo_RRJ + 1;
		if(iViewportNo_RRJ > 9)
			iViewportNo_RRJ = 0;

		return(true);
	}

	/********** Methods from OnGestureListener Interface **********/
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



	/********** Methods From GLSurfaceView.Renderer Interface **********/
	@Override
	public void onSurfaceCreated(GL10 gl, EGLConfig config){
		String version_RRJ = gl.glGetString(GL10.GL_VERSION);
		String glsl_version_RRJ = gl.glGetString(GLES32.GL_SHADING_LANGUAGE_VERSION);
		String vendor_RRJ = gl.glGetString(GL10.GL_VENDOR);
		String renderer_RRJ = gl.glGetString(GL10.GL_RENDERER);

		System.out.println("RTR: OpenGL Version: " + version_RRJ);
		System.out.println("RTR: OpenGLSL Version: " + glsl_version_RRJ);
		System.out.println("RTR: Vendor: "+ vendor_RRJ);
		System.out.println("RTR: Renderer: "+ renderer_RRJ);

		initialize();
	}

	@Override 
	public void onSurfaceChanged(GL10 unused, int width, int height){
		resize(width, height);
	}

	@Override
	public void onDrawFrame(GL10 unused){
		display();
	}


	/********** Now OpenGL Starts **********/

	private void initialize(){

		/********** Vertex Shader **********/
		vertexShaderObject_RRJ = GLES32.glCreateShader(GLES32.GL_VERTEX_SHADER);

		final String vertexShaderSourceCode_RRJ = String.format(
				"#version 320 es" +
				"\n" +
				"in vec4 vPosition;" +
				"in vec3 vColor;" +
				"out vec3 outColor;" +

				"uniform mat4 u_mvp_matrix;" +
				"void main(void)" +
				"{" + 
					"outColor = vColor;" +
					"gl_Position = u_mvp_matrix * vPosition;"  +
				"}" 
			);

		GLES32.glShaderSource(vertexShaderObject_RRJ, vertexShaderSourceCode_RRJ);

		GLES32.glCompileShader(vertexShaderObject_RRJ);

		int iShaderCompileStatus_RRJ[] = new int[1];
		int iInfoLogLength_RRJ[] = new int[1];
		String szInfoLog_RRJ = null;

		GLES32.glGetShaderiv(vertexShaderObject_RRJ, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus_RRJ, 0);
		if(iShaderCompileStatus_RRJ[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(vertexShaderObject_RRJ, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength_RRJ, 0);
			if(iInfoLogLength_RRJ[0] > 0){

				szInfoLog_RRJ = GLES32.glGetShaderInfoLog(vertexShaderObject_RRJ);
				System.out.println("RTR: Vertex Shader Compilation Error: " + szInfoLog_RRJ);

				uninitialize();
				System.exit(0);
			
			}
		}


	


		/********** Fragment Shader **********/
		fragmentShaderObject_RRJ = GLES32.glCreateShader(GLES32.GL_FRAGMENT_SHADER);

		final String fragmentShaderSourceCode_RRJ = String.format(
				"#version 320 es" +
				"\n" +
				"precision highp float;" +
				"in vec3 outColor;" +
				"out vec4 FragColor;" +
				"void main(void)" +
				"{" +
					"FragColor = vec4(outColor, 1.0f);" +
				"}"
			);

		GLES32.glShaderSource(fragmentShaderObject_RRJ, fragmentShaderSourceCode_RRJ);

		GLES32.glCompileShader(fragmentShaderObject_RRJ);

		GLES32.glGetShaderiv(fragmentShaderObject_RRJ, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus_RRJ, 0);
		if(iShaderCompileStatus_RRJ[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(fragmentShaderObject_RRJ, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength_RRJ, 0);
			if(iInfoLogLength_RRJ[0] > 0){
				szInfoLog_RRJ = GLES32.glGetShaderInfoLog(fragmentShaderObject_RRJ);
				System.out.println("RTR: Fragment Shader Compilation Error: "+ szInfoLog_RRJ);
				uninitialize();
				System.exit(0);
			}
		}


		/********** Shader Program Object **********/
		shaderProgramObject_RRJ = GLES32.glCreateProgram();

		GLES32.glAttachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
		GLES32.glAttachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);

		GLES32.glBindAttribLocation(shaderProgramObject_RRJ, GLESMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
		GLES32.glBindAttribLocation(shaderProgramObject_RRJ, GLESMacros.AMC_ATTRIBUTE_COLOR, "vColor");

		GLES32.glLinkProgram(shaderProgramObject_RRJ);

		int iProgramLinkStatus_RRJ[] = new int[1];
		iInfoLogLength_RRJ[0] = 0;
		szInfoLog_RRJ = null;

		GLES32.glGetProgramiv(shaderProgramObject_RRJ, GLES32.GL_LINK_STATUS, iProgramLinkStatus_RRJ, 0);
		if(iProgramLinkStatus_RRJ[0] == GLES32.GL_FALSE){
			GLES32.glGetProgramiv(shaderProgramObject_RRJ, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength_RRJ, 0);
			if(iInfoLogLength_RRJ[0] > 0){
				szInfoLog_RRJ = GLES32.glGetProgramInfoLog(shaderProgramObject_RRJ);
				System.out.println("RTR: Shader Program Linking Error: "+ szInfoLog_RRJ);
				uninitialize();
				System.exit(0);
			}
		}



		mvpUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_mvp_matrix");



		/********** Position and Color **********/
		final float tri_Pos_RRJ[] = new float[]{
			0.0f, 1.0f, 0.0f,
			-1.0f, -1.0f, 0.0f,
			1.0f, -1.0f, 0.0f,
		};


		final float tri_Col_RRJ[] = new float[] {
			1.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 1.0f,
		};

			

		/********** Triangle **********/
		GLES32.glGenVertexArrays(1, vao_Tri_RRJ, 0);
		GLES32.glBindVertexArray(vao_Tri_RRJ[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Tri_Position_RRJ, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Tri_Position_RRJ[0]);

				/********** For glBufferData() ***********/
				//Allocate ByteBuffer
				ByteBuffer byteBuffer_Position_RRJ = ByteBuffer.allocateDirect(tri_Pos_RRJ.length * 4);
				//Set native ByteOrder
				byteBuffer_Position_RRJ.order(ByteOrder.nativeOrder());
				//Float Buffer
				FloatBuffer positionBuffer_RRJ = byteBuffer_Position_RRJ.asFloatBuffer();
				//Put Data into Float Buffer
				positionBuffer_RRJ.put(tri_Pos_RRJ);
				//Set Position to 0
				positionBuffer_RRJ.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
								tri_Pos_RRJ.length * 4,
								positionBuffer_RRJ,
								GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
										3,
										GLES32.GL_FLOAT,
										false,
										0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);





			/********** Color **********/
			GLES32.glGenBuffers(1, vbo_Tri_Color_RRJ, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Tri_Color_RRJ[0]);

				ByteBuffer byteBuffer_Color_RRJ = ByteBuffer.allocateDirect(tri_Col_RRJ.length * 4);
				byteBuffer_Color_RRJ.order(ByteOrder.nativeOrder());
				FloatBuffer floatBuffer_Color_RRJ = byteBuffer_Color_RRJ.asFloatBuffer();
				floatBuffer_Color_RRJ.put(tri_Col_RRJ);
				floatBuffer_Color_RRJ.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, tri_Col_RRJ.length * 4, floatBuffer_Color_RRJ, GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_COLOR, 3, GLES32.GL_FLOAT, false, 0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_COLOR);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

			
		GLES32.glBindVertexArray(0);

		iViewportNo_RRJ = 0;

		GLES32.glEnable(GLES32.GL_DEPTH_TEST);
		GLES32.glDepthFunc(GLES32.GL_LEQUAL);

		GLES32.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	}




	private void uninitialize(){

		if(vbo_Tri_Color_RRJ[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Tri_Color_RRJ, 0);
			vbo_Tri_Color_RRJ[0] = 0;
		}

		if(vbo_Tri_Position_RRJ[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Tri_Position_RRJ, 0);
			vbo_Tri_Position_RRJ[0] = 0;
		}

		if(vao_Tri_RRJ[0] != 0){
			GLES32.glDeleteVertexArrays(1, vao_Tri_RRJ, 0);
			vao_Tri_RRJ[0] = 0;
		}

		if(shaderProgramObject_RRJ != 0){

			int iShaderCount[] = new int[1];
			int iShaderNo;

			GLES32.glUseProgram(shaderProgramObject_RRJ);

					/*GLES32.glGetProgramiv(shaderProgramObject_RRJ, GLES32.GL_ATTACHED_SHADERS, iShaderCount, 0);
					int iShaders[] = new int[iShaderCount[0]];
					GLES32.glGetAttachedShaders(shaderProgramObject_RRJ, iShaderCount[0],
																iShaderCount, 0,
																iShaders, 0);
					
					for(iShaderNo = 0; iShaderNo < iShaderCount[0]; iShaderNo++){
							GLES32.glDetachShader(shaderProgramObject_RRJ, iShaders[iShaderNo]);
							GLES32.glDeleteShader(iShaders[iShaderNo]);
							iShaders[iShaderNo] = 0;
					}*/



					if(fragmentShaderObject_RRJ != 0){
						GLES32.glDetachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);
						GLES32.glDeleteShader(fragmentShaderObject_RRJ);
						fragmentShaderObject_RRJ = 0;
					}


					if(vertexShaderObject_RRJ != 0){
						GLES32.glDetachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
						GLES32.glDeleteShader(vertexShaderObject_RRJ);
						vertexShaderObject_RRJ = 0;
					}

			GLES32.glUseProgram(0);
			GLES32.glDeleteProgram(shaderProgramObject_RRJ);
			shaderProgramObject_RRJ = 0;
		}

	}



	private void resize(int width, int height){
		if(height == 0)
			height = 1;

		viewPortWidth_RRJ = width;
		viewPortHeight_RRJ = height;

		Matrix.setIdentityM(perspectiveProjectionMatrix_RRJ, 0);

		Matrix.perspectiveM(perspectiveProjectionMatrix_RRJ, 0,
							45.0f,
							(float)width / (float)height,
							0.1f,
							100.0f);
	}
	

	private void display(){

		float translateMatrix_RRJ[] = new float[4 * 4];
		float modelViewMatrix_RRJ[] = new float[4 * 4];
		float modelViewProjectionMatrix_RRJ[] = new float[4 * 4];

		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);

		

		GLES32.glUseProgram(shaderProgramObject_RRJ);

			/********** Lines **********/
			Matrix.setIdentityM(translateMatrix_RRJ, 0);
			Matrix.setIdentityM(modelViewMatrix_RRJ, 0);
			Matrix.setIdentityM(modelViewProjectionMatrix_RRJ, 0);

			Matrix.translateM(translateMatrix_RRJ, 0, 0.0f, 0.0f, -4.0f);
			Matrix.multiplyMM(modelViewMatrix_RRJ, 0, modelViewMatrix_RRJ, 0, translateMatrix_RRJ, 0);
			Matrix.multiplyMM(modelViewProjectionMatrix_RRJ, 0, perspectiveProjectionMatrix_RRJ, 0, modelViewMatrix_RRJ, 0);


			GLES32.glUniformMatrix4fv(mvpUniform_RRJ, 1, false, modelViewProjectionMatrix_RRJ, 0);


			if (iViewportNo_RRJ == 0)
				GLES32.glViewport(0, 0, viewPortWidth_RRJ, viewPortHeight_RRJ);
			else if (iViewportNo_RRJ == 1)
				GLES32.glViewport(0, 0, (viewPortWidth_RRJ) / 2, (viewPortHeight_RRJ) / 2);
			else if (iViewportNo_RRJ == 2)
				GLES32.glViewport(viewPortWidth_RRJ / 2, 0, viewPortWidth_RRJ / 2, viewPortHeight_RRJ / 2);
			else if (iViewportNo_RRJ == 3)
				GLES32.glViewport(viewPortWidth_RRJ / 2, viewPortHeight_RRJ / 2, viewPortWidth_RRJ / 2, viewPortHeight_RRJ / 2);
			else if (iViewportNo_RRJ == 4)
				GLES32.glViewport(0, viewPortHeight_RRJ / 2, viewPortWidth_RRJ / 2, viewPortHeight_RRJ / 2);
			else if (iViewportNo_RRJ == 5)
				GLES32.glViewport(0, 0, viewPortWidth_RRJ / 2, viewPortHeight_RRJ);
			else if (iViewportNo_RRJ == 6)
				GLES32.glViewport(viewPortWidth_RRJ / 2, 0, viewPortWidth_RRJ / 2, viewPortHeight_RRJ);
			else if (iViewportNo_RRJ == 7)
				GLES32.glViewport(0, viewPortHeight_RRJ / 2, viewPortWidth_RRJ, viewPortHeight_RRJ / 2);
			else if (iViewportNo_RRJ == 8)
				GLES32.glViewport(0, 0, viewPortWidth_RRJ, viewPortHeight_RRJ / 2);
			else if (iViewportNo_RRJ == 9)
				GLES32.glViewport(viewPortWidth_RRJ / 4, viewPortHeight_RRJ / 4, viewPortWidth_RRJ / 2, viewPortHeight_RRJ / 2);

			GLES32.glBindVertexArray(vao_Tri_RRJ[0]);
				GLES32.glDrawArrays(GLES32.GL_TRIANGLES, 0, 3);
			GLES32.glBindVertexArray(0);	

		GLES32.glUseProgram(0);

		requestRender();

	}


};

