package com.astromedicomp.hello2;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;
import android.graphics.Color;
import android.view.Gravity;
import android.content.Context;

public class MyView extends TextView {

    MyView(Context context)
	{
		super(context);
		setText("Hello World 2");
		setTextColor(Color.rgb(0,100,0));
		setTextSize(60);
		setGravity(Gravity.CENTER);
	}
}
