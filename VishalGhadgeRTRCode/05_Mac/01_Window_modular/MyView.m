//Headers
#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>
#include"MyView.h"

@implementation MyView
{
	NSString *centralText;
}

-(id) initWithFrame:(NSRect)frame;
{
	self = [super initWithFrame:frame];
	
	if (self)
	{
		[[self window]setContentView:self];
		centralText=@"Hellow World !!!";
	}
	return(self);
}

-(void)drawRect:(NSRect)dirtyRect
{
	NSColor *fillColor=[NSColor blackColor];
	[fillColor set];
	NSRectFill(dirtyRect);
	
	//	dictionary with key
	NSDictionary *dictionaryForTextAttributes = [NSDictionary dictionaryWithObjectsAndKeys:
	[NSFont fontWithName:@"Helvetica" size:32], NSFontAttributeName, [NSColor greenColor],
	NSForegroundColorAttributeName, nil];
	
	NSSize textSize = [centralText sizeWithAttributes:dictionaryForTextAttributes];
	
	NSPoint point;
	point.x = (dirtyRect.size.width/2) - (textSize.width/2);
	point.y = (dirtyRect.size.height/2) - (textSize.height/2) + 12;
	
	[centralText drawAtPoint:point withAttributes:dictionaryForTextAttributes];
}

-(BOOL)acceptsFirstResponder
{
	//	Code
	[[self window]makeFirstResponder:self];
	return(YES);
}

-(void)keyDown:(NSEvent *)theEvent
{
	int key = (int)[[theEvent characters]characterAtIndex:0];
    centralText=@"key is pressed";
    [[self window]toggleFullScreen:self];
    
    switch(key)
	{
		case 27:	//	Esc key
				[self release];
				[NSApp terminate:self];
				break;
		case 'F':
		case 'f':
				centralText=@"'F' key is pressed";
				[[self window]toggleFullScreen:self];
				break;
		default:
				break;
	}
}

-(void)mouseDown:(NSEvent *)theEvent
{
	centralText = @"Left mouse button is clicked.";
	[self setNeedsDisplay:YES];	//	RePainting
}

-(void)mouseDragged:(NSEvent *)theEvent
{
	//	Code
}

-(void)rightMouseDown:(NSEvent *)theEvent
{
	centralText = @"Right mouse button is clicked.";
	[self setNeedsDisplay:YES];	//	RePainting
}

-(void) dealloc
{
	[super dealloc];
}
@end
