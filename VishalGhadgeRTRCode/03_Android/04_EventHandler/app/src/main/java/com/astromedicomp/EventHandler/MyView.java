package com.astromedicomp.EventHandler;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;
import android.graphics.Color;
import android.view.Gravity;
import android.content.Context;
import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnGestureListener;
import android.view.GestureDetector.OnDoubleTapListener;

public class MyView extends TextView implements OnGestureListener,OnDoubleTapListener{

	private GestureDetector gestureDetector;
	
    MyView(Context context)
	{
		super(context);
		setText("Event handler");
		setTextColor(Color.rgb(0,100,0));
		setTextSize(60);
		setGravity(Gravity.CENTER);
		
		gestureDetector = new GestureDetector(context,this, null, false);
		gestureDetector.setOnDoubleTapListener(this);	//	this means handler i.e) who is going to handle.
	}
	
	//
	//	Handling 'onTouchEvent' is the most important.
	//	Because it triggers all gesture and tap events.
	//
	@Override
	public boolean onTouchEvent(MotionEvent event)
	{
		int eventAction = event.getAction();
		if (!gestureDetector.onTouchEvent(event))
		{
			super.onTouchEvent(event);
		}
		return(true);
	}
	
	//
	//	abstract method from OnDoubleTapListener so must be implemented
	//
	@Override
	public boolean onDoubleTap(MotionEvent e)
	{
		setText("Double Tap");
		return(true);
	}
	
	//
	//	abstract method from OnDoubleTapListener so must be implemented
	//
	@Override
	public boolean onDoubleTapEvent(MotionEvent e)
	{
		//	Do not write any code here bacause already written 'onDoubleTap'
		return true;
	}
	
	//
	//	abstract method from OnDoubleTapListener so must be implemented
	//
	@Override
	public boolean onSingleTapConfirmed(MotionEvent e)
	{
		setText("Single Tap");
		return true;
	}
	
	//
	//	abstract method from OnDoubleTapListener so must be implemented
	//
	@Override
	public boolean onDown(MotionEvent e)
	{
		//	Do not write any code here bacause already written 'onSingleTapConfirmed'
		return true;
	}
	
	//
	//	abstract method from OnGestureListener so must be implemented
	//
	@Override
	public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY)
	{
		setText("Fling");
		return true;
	}
	
	//
	//	abstract method from OnGestureListener so must be implemented
	//
	@Override
	public void onLongPress(MotionEvent e)
	{
		setText("Long press");
	}
	
	//
	//	abstract method from OnGestureListener so must be implemented
	//
	@Override
	public boolean onScroll(MotionEvent e1,MotionEvent e2, float distanceX, float distanceY)
	{
		setText("Scroll");
		return true;
	}
	
	//
	//	abstract method from OnGestureListener so must be implemented
	//
	@Override
	public void onShowPress(MotionEvent e)
	{
		
	}
	
	//
	//	abstract method from OnGestureListener so must be implemented
	//
	@Override
	public boolean onSingleTapUp(MotionEvent e)
	{
		return true;
	}
}
