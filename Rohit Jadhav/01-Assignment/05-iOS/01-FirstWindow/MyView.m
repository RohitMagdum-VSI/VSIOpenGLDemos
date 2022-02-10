//
//  MyView.m
//  01-FirstWindow
//
//  Created by user160249 on 3/22/20.
//

#import "MyView.h"

@implementation MyView{
    NSString *centralText_RRJ;
}

-(id)initWithFrame:(CGRect)frameRect{

    self = [super initWithFrame:frameRect];

    if(self){

        [self setBackgroundColor:[UIColor whiteColor]];

        centralText_RRJ = @"Hello iOS World !!";

        //Single Tap
        UITapGestureRecognizer *singleTapGestureRecognizer_RRJ = [[UITapGestureRecognizer alloc] initWithTarget: self action:@selector(onSingleTap:)];
        [singleTapGestureRecognizer_RRJ setNumberOfTapsRequired:1];
        [singleTapGestureRecognizer_RRJ setNumberOfTouchesRequired:1];
        [singleTapGestureRecognizer_RRJ setDelegate:self];
        [self addGestureRecognizer:singleTapGestureRecognizer_RRJ];

        //Double Tap
        UITapGestureRecognizer *doubleTapGestureRecognizer_RRJ = [[UITapGestureRecognizer alloc] initWithTarget: self action:@selector(onDoubleTap:)];
        [doubleTapGestureRecognizer_RRJ setNumberOfTapsRequired:2];
        [doubleTapGestureRecognizer_RRJ setNumberOfTouchesRequired:1];
        [doubleTapGestureRecognizer_RRJ setDelegate:self];
        [self addGestureRecognizer: doubleTapGestureRecognizer_RRJ];


        //To Know the difference between doubleTap and 2 singleTap
        [singleTapGestureRecognizer_RRJ requireGestureRecognizerToFail:doubleTapGestureRecognizer_RRJ];


        //Swipe
        UISwipeGestureRecognizer *swipeGestureRecognizer_RRJ = [[UISwipeGestureRecognizer alloc] initWithTarget:self action:@selector(onSwipe:)];
        [self addGestureRecognizer:swipeGestureRecognizer_RRJ];


        //Long Press
        UILongPressGestureRecognizer *longPressGestureRecognizer_RRJ = [[UILongPressGestureRecognizer alloc] initWithTarget:self action:@selector(onLongPress:)];
        [self addGestureRecognizer:longPressGestureRecognizer_RRJ];

    }
    return(self);
}


-(void)drawRect:(CGRect)rect{

    UIColor *fillColor_RRJ = [UIColor blackColor];
    [fillColor_RRJ set];
    UIRectFill(rect);


    //Dictionary With KVC
    NSDictionary *dictionaryForTextAttributes_RRJ = [NSDictionary dictionaryWithObjectsAndKeys:
        [UIFont fontWithName:@"Helvetica" size:24], NSFontAttributeName,
        [UIColor greenColor], NSForegroundColorAttributeName, nil];

    CGSize textSize_RRJ = [centralText_RRJ sizeWithAttributes:dictionaryForTextAttributes_RRJ];

    CGPoint point_RRJ;
    point_RRJ.x = (rect.size.width / 2) - (textSize_RRJ.width / 2);
    point_RRJ.y = (rect.size.height / 2) - (textSize_RRJ.height / 2) + 12;

    [centralText_RRJ drawAtPoint:point_RRJ withAttributes:dictionaryForTextAttributes_RRJ];
}

-(BOOL)acceptsFirstResponder{
    return(YES);
}

-(void)my_add{
    
}

-(void)touchesBegan:(NSSet*)touches withEvent:(UIEvent*)event{

    /*

    centralText_RRJ = @"tochesBegan: Event Occured";

    [self setNeedsDisplay];

    */
}

-(void)onSingleTap:(UITapGestureRecognizer*)gr{

    centralText_RRJ = @"'onSingleTap' Event Occured";
    [self setNeedsDisplay];
}

-(void)onDoubleTap:(UITapGestureRecognizer*)gr{

    centralText_RRJ = @"'onDoubleTap' Event Occured";
    [self setNeedsDisplay];
}

-(void)onSwipe:(UISwipeGestureRecognizer*)gr{
    [self release];
    exit(0);
}

-(void)onLongPress:(UILongPressGestureRecognizer*)gr{

    centralText_RRJ = @"'onLongPress' Event Occured";
    [self setNeedsDisplay];
}

-(void)dealloc{

    [super dealloc];
}

@end
