//
//  MyView.m
//  Window
//
//  Created by Vishal on 6/10/18.
//

#import "MyView.h"

@implementation MyView
{
	NSString *centralText;
}


- (id)initWithFrame:(CGRect)frameRect
{
	self=[super initWithFrame:frameRect];
	if (self)
	{
		//	Initialization
		
		//	Set scenes background color
		[self setBackgroundColor:[UIColor whiteColor]];
		
		centralText=@"Hello World !!!";
		
		//	Gesture Recognition
		
		//	Tap gesture code.
		UITapGestureRecognizer *singleTapGestureRecognizer = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(onSingleTap:)];
		[singleTapGestureRecognizer setNumberOfTapsRequired:1];
		[singleTapGestureRecognizer setNumberOfTouchesRequired:1];
		[singleTapGestureRecognizer setDelegate:self];
		[self addGestureRecognizer:singleTapGestureRecognizer];
		
		UITapGestureRecognizer *doubleTapGestureRecognizer = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(onDoubleTap:)];
		[doubleTapGestureRecognizer setNumberOfTapsRequired:2];
		[doubleTapGestureRecognizer setNumberOfTouchesRequired:1];	//	Touch of 1 finger.
		[doubleTapGestureRecognizer setDelegate:self];
		[self addGestureRecognizer:doubleTapGestureRecognizer];
		
		//	This will allow to diffrentiate between single tap and double tap.
		[singleTapGestureRecognizer requireGestureRecognizerToFail:doubleTapGestureRecognizer];
		
		//	Swipe gesture
		UISwipeGestureRecognizer *swipeGestureRecognizer = [[UISwipeGestureRecognizer alloc] initWithTarget:self action:@selector(onSwipe:)];
		[self addGestureRecognizer:swipeGestureRecognizer];
	}
    
    return self;
}

/*
// Only override drawRect: if you perform custom drawing.
// An empty implementation adversely affects performance during animation.
*/
- (void)drawRect:(CGRect)rect
{
	// Black background
	UIColor *fillColor = [UIColor blackColor];
	[fillColor set];
	UIRectFill(rect);
	
	//	dictionary with kvc
	NSDictionary *dictionaryForTextAttributes = [NSDictionary dictionaryWithObjectsAndKeys:[UIFont fontWithName:@"Helvetica" size:24],NSFontAttributeName,[UIColor greenColor], NSForegroundColorAttributeName, nil];
												
	CGSize textSize = [centralText sizeWithAttributes:dictionaryForTextAttributes];
	
	CGPoint point;
	point.x = (rect.size.width / 2) - (textSize.width/2);
	point.y = (rect.size.height / 2) - (textSize.height/2) + 12;
	
	[centralText drawAtPoint:point withAttributes:dictionaryForTextAttributes];
}

//	To become first responder
- (BOOL)acceptsFirstResponder
{
	return YES;
}

- (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event
{
	centralText = @"'Touches began' event occured.";
	[self setNeedsDisplay];
}

- (void)onSingleTap:(UITapGestureRecognizer *)gr
{
	centralText = @"'On Single TAP' event occured";
	[self setNeedsDisplay];
}

- (void)onDoubleTap:(UITapGestureRecognizer *)gr
{
	centralText = @"'On Double TAP' event occured";
	[self setNeedsDisplay];
}

- (void)onLongPress:(UILongPressGestureRecognizer *)gr
{
	centralText = @"'On Long Press' event occured";
	[self setNeedsDisplay];
}

- (void)onSwipe:(UISwipeGestureRecognizer *)gr
{
	[self release];
	exit(0);
}

- (void)dealloc
{
	[super dealloc];
}

@end
