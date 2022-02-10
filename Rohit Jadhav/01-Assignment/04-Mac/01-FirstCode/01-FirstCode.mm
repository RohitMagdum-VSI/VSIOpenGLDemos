#import<Foundation/Foundation.h>
#import<Cocoa/Cocoa.h>


@interface AppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@end

@interface MyView : NSView
@end

int main(int argc, const char *argv[]){

	NSAutoreleasePool *pPool = [ [NSAutoreleasePool alloc] init ];

	NSApp = [NSApplication sharedApplication];

	[NSApp setDelegate : [[AppDelegate alloc] init]];

	[NSApp run];

	[pPool release];

	return(0);
}




/******************** AppDelegate ********************/
@implementation AppDelegate{
	//Class Variable
	@private
		NSWindow *window;
		MyView *view;
}


//Like WM_CREATE
- (void) applicationDidFinishLaunching:(NSNotification*)aNotification{

	//Part 1: Window
	NSRect win_rect;
	win_rect = NSMakeRect(0.0,0.0, 800.0, 600.0);

	window = [ [NSWindow alloc] initWithContentRect:win_rect 
										styleMask:NSWindowStyleMaskTitled | 
										NSWindowStyleMaskClosable | 
										NSWindowStyleMaskResizable |
										NSWindowStyleMaskMiniaturizable 
										backing:NSBackingStoreBuffered 
										defer:NO];

	[window setTitle:@"macOS-RohitRJadhav-01-FirstWindow"];
	[window center];

	//Part 2: View
	view = [ [MyView alloc] initWithFrame:win_rect];

	[window setContentView:view];
	[window setDelegate:self];
	[window makeKeyAndOrderFront:self];
}


//Like WM_DESTROY
- (void)applicationWillTerminate:(NSNotification*)notification{

}


//Like WM_CLOSE
- (void)windowWillClose:(NSNotification*)notification{
	[NSApp terminate:self];
}

- (void)dealloc{

	[view release];

	[window release];

	[super dealloc];
}

@end


/******************** MyView ********************/
@implementation MyView{
	NSString *centerText;
}

-(id)initWithFrame:(NSRect)frame{

	self = [super initWithFrame:frame];

	if(self){

		[[self window]setContentView:self];

		centerText = @"Hello World!!";
	}

	return(self);
}


//Like WM_PAINT
-(void)drawRect:(NSRect)dirtyRect{

	NSColor *fillColor = [NSColor blackColor];
	[fillColor set];
	NSRectFill(dirtyRect);


	NSDictionary *dictionaryForTextAttribute = [NSDictionary dictionaryWithObjectsAndKeys: [NSFont fontWithName:@"Helvetica" size:32],
													NSFontAttributeName, [NSColor greenColor],
													NSForegroundColorAttributeName, nil];

	NSSize textSize = [centerText sizeWithAttributes:dictionaryForTextAttribute];

	NSPoint point;
	point.x = (dirtyRect.size.width/2) - (textSize.width/2);
	point.y = (dirtyRect.size.height/2) - (textSize.height/2) + 12;

	[centerText drawAtPoint:point withAttributes:dictionaryForTextAttribute];
} 


-(BOOL)acceptsFirstResponder{
	[[self window]makeFirstResponder:self];
	return(YES);
}



-(void)keyDown:(NSEvent*)theEvent{
	int key = (int)[[theEvent characters]characterAtIndex:0];
	switch(key){
		case 27:
			[self release];
			[NSApp terminate:self];
			break;

		case 'F':
		case 'f':
		centerText = @"F Key is Press!!";
		[[self window]toggleFullScreen:self];
		break;

	default:
		break;
	}
}

-(void)mouseDown:(NSEvent*)event{
	centerText = @"Mouse is Click!!";
	[self setNeedsDisplay:YES];
}

-(void)mouseDragged:(NSEvent*)event{

}

-(void)rightMouseDown:(NSEvent*)event{
	centerText = @"Right Mouse Button is Click!!";
	[self setNeedsDisplay:YES];
}

-(void)dealloc{
	[super dealloc];
}



@end
