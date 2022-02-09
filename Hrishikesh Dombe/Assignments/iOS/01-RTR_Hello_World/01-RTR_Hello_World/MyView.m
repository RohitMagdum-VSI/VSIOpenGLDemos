#import "MyView.h"

@implementation MyView
{
    NSString *centralText;
}

- (id)initWithFrame:(CGRect)frameRect
{
    self=[super initWithFrame:frameRect];

    if(self)
    {
        [self setBackgroundColor:[UIColor whiteColor]];

        centralText=@"Hello World !!!";

        //Tap Gesture Code
        UITapGestureRecognizer *singleTapGestureRecognizer = [[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(onSingleTap:)];
        [singleTapGestureRecognizer setNumberOfTapsRequired:1];
        [singleTapGestureRecognizer setNumberOfTouchesRequired:1];
        [singleTapGestureRecognizer setDelegate:self];
        [self addGestureRecognizer:singleTapGestureRecognizer];

        UITapGestureRecognizer *doubleTapGestureRecognizer=[[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(onDoubleTap:)];
        [doubleTapGestureRecognizer setNumberOfTapsRequired:2];
        [doubleTapGestureRecognizer setNumberOfTouchesRequired:1];
        [doubleTapGestureRecognizer setDelegate:self];
        [self addGestureRecognizer:doubleTapGestureRecognizer];

        //This will allow to differentiate between single tap and double tap
        [singleTapGestureRecognizer requireGestureRecognizerToFail:doubleTapGestureRecognizer];

        //Swipe
        UISwipeGestureRecognizer *swipeGestureRecognizer = [[UISwipeGestureRecognizer alloc]initWithTarget:self action:@selector(onSwipe:)];
        [self addGestureRecognizer:swipeGestureRecognizer];

        //Long Press
        UILongPressGestureRecognizer *longPressGestureRecognizer = [[UILongPressGestureRecognizer alloc]initWithTarget:self action:@selector(onLongPress:)];
        [self addGestureRecognizer:longPressGestureRecognizer];
    }
    return(self);
}

- (void)drawRect:(CGRect)rect
{
    UIColor *fillColor=[UIColor grayColor];
    [fillColor set];
    UIRectFill(rect);

    NSDictionary *dictionaryForTextAttributes=[NSDictionary dictionaryWithObjectsAndKeys:
                                                [UIFont fontWithName:@"Helvetica" size:24], NSFontAttributeName,
                                                [UIColor blackColor], NSForegroundColorAttributeName,
                                                nil];

    CGSize textSize = [centralText sizeWithAttributes:dictionaryForTextAttributes];

    CGPoint point;
    point.x=(rect.size.width/2)-(textSize.width/2);
    point.y=(rect.size.height/2)-(textSize.height/2)+12;

    [centralText drawAtPoint:point withAttributes:dictionaryForTextAttributes];
}

-(BOOL)acceptsFirstResponder
{
    return(YES);
}

- (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *)UIEvent
{

}

-(void)onSingleTap:(UITapGestureRecognizer *)gesture
{
    centralText = @"'onSingleTap' Event Occured";
    [self setNeedsDisplay];
}

-(void)onDoubleTap:(UITapGestureRecognizer *)gesture
{
    centralText = @"'onDoubleTap' Event Occured";
    [self setNeedsDisplay];
}

-(void)onSwipe:(UISwipeGestureRecognizer *)gesture
{
    [self release];
    exit(0);
}

-(void)onLongPress:(UILongPressGestureRecognizer *)gesture
{
    centralText = @"'onLongPress' Event Occured";
    [self setNeedsDisplay];
}

-(void)dealloc
{
    [super dealloc];
}

@end
