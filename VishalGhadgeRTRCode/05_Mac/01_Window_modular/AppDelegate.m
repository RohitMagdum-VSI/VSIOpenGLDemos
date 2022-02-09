//Headers
#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>
#include "AppDelegate.h"
#include"MyView.h"

//	interface implementation
@implementation AppDelegate
{
	@private
			NSWindow *window;
			MyView *view;
}

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification
{
	NSRect win_rect;
	
	win_rect = NSMakeRect(0.0,0.0,800.0,600.0);
	
	//	Create simple window
	window = [[NSWindow alloc] initWithContentRect:win_rect
	styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
	| NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
	backing:NSBackingStoreBuffered defer:NO];
	
	[window setTitle:@"mac OS Window"];
	[window center];
	
	view = [[MyView alloc]initWithFrame:win_rect];
	
	[window setContentView:view];
	[window setDelegate:self];
	[window makeKeyAndOrderFront:self];
}

- (void)applicationWillTerminate:(NSNotification *)notification	//	Same as WmDestroy/WmClose
{
	//	Code
}

- (void)windowWillClose:(NSNotification*)notification
{
	//	Code
	[NSApp terminate:self];
}

- (void) dealloc
{
	//	Code
	[view release];
	
	[window release];
	
	[super dealloc];
}
@end	//	implementation of AppDelegate
