package com.astromedicomp.gestures;

import android.content.Context;
import android.graphics.Color;
import android.widget.TextView;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnGestureListener;
import android.view.GestureDetector.OnDoubleTapListener;

public class MyView extends TextView implements OnGestureListener, OnDoubleTapListener
{
	private GestureDetector gestureDetector;

	MyView(Context context)
	{
		super(context);
		setText("Hello World For Gestures!!!");
		setTextSize(60);
		setTextColor(Color.rgb(128,128,128));
		setGravity(Gravity.CENTER);	

		//context : Global Hardware Context
		//this : who is going to use this class
		//null : If there are some events which are not 				//handled but we implement them, then the name of that 		//class is passed here.
		//false : Unused parameter(always false)
		gestureDetector = new GestureDetector(context, this, null, false);

		gestureDetector.setOnDoubleTapListener(this);

	}

		//Handling 'onTouchEvent' Is The Most IMPORTANT,
		//Because It Triggers All Gesture And Tap Events
		@Override
		public boolean onTouchEvent(MotionEvent event)
		{
			int eventaction = event.getAction();
			if(!gestureDetector.onTouchEvent(event))
				super.onTouchEvent(event);
			return(true);
		}

		//abstract method from OnDoubleTapListener so must be 			//implemented
		@Override
		public boolean onDoubleTap(MotionEvent e)
		{
			setText("Double Tap");
			return(true);
		}

		//abstract method from OnDoubleTapListener so must be 			//implemented
		@Override
		public boolean onDoubleTapEvent(MotionEvent e)
		{
			return(true);
		}


		//abstract method from OnDoubleTapListener so must be 			//implemented
		@Override
		public boolean onSingleTapConfirmed(MotionEvent e)
		{
			setText("Single Tap");
			return(true);
		}

		//abstract method from OnGestureListener so must be 			//implemented
		@Override
		public boolean onDown(MotionEvent e)
		{
			//Do not write any code here because code is 				//already written in onSingleTapConfirmed
			return(true);
		}

		//abstract method from OnGestureListener so must be 			//implemented
		@Override
		public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY)
		{
			setText("Fling");
			return(true);
		}

		//abstract method from OnGestureListener so must be 			//implemented
		@Override
		public void onLongPress(MotionEvent e)
		{
			setText("Long Press");
		}

		//abstract method from OnGestureListener so must be 			//implemented
		@Override 
		public boolean onScroll(MotionEvent e1, MotionEvent e2, float distanceX, float distanceY)
		{
			setText("Scroll");
			return(true);
		}

		//abstract method from OnGestureListener so must be 			//implemented
		@Override
		public void onShowPress(MotionEvent e)
		{
			
		}	

		//abstract method from OnGestureListener so must be 			//implemented
		@Override
		public boolean onSingleTapUp(MotionEvent e)
		{
			return(true);
		}	
	
}