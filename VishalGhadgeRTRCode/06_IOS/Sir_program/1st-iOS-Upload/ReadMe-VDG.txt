01) Start Xcode.
    Xcode > Top Main Menu > File > New > Project > iOS > Application > Single View Application > Next

02) From ‘Choose a template for your new project’ Dialog, Choose Following Options :

    Product Name : Window
    Team : Skip
    Organisation Name : Keep Empty
    Organisation Identifier : astromedicomp.org
    Language : Objective-C
    Use Core Data : Uncheck
    Include Unit Tests : Uncheck
    Include UI Tests : Uncheck

03) Next

04) Browse And Select Location
    Uncheck Create Git repository on Box
    Create

    Xcode IDE Creates 5 Code Files For Us : AppDelegate.m/.h, ViewController.m/.h And main.m
    We Can See Them In The Left Pane.
    [ There Are Some Additional Files Too, But They Are Not Code Files ].

05) Left Panel > Click On <Project Name> Blue Icon (.xcodeproj) > In Middle Pane,
    From Top Tab Bar, Choose ‘General’ Tab. In It :

    A) Signing : Uncheck ‘Automatically manage signing’
    B) Keep ’Signing (Debug)’ And ’Signing (Release)’ Untouched.
    C) Deployment Info :
       a) Deployment Target : 11.3 <Latest>
       b) Device : iPhone
       c) Main Interface : < Keep Empty >
       d) Device Orientation : Check Landscape Right
       e) Check Hide Status Bar Checkbox
       f) Check Requires full screen
    D) App Icons and Launch Images : 
       a) If You Have Icons, Then Click The Arrow Icon Ahead Of ‘App Icons Source’ Field
          And Then Drag And Drop Specific Sized png Images Onto The Image Placeholders.
       b) Launch Screen File : < Keep Empty >

06) Xcode, In The Top Navigation Bar, May Show Yellow Warning Icon.
    In Left Pane > Click On ‘Issue Navigator Icon’ And Then Click On The Actual Issue Text.
    From Displayed Dialog, Choose ‘Add’ To Add The Required Default Launch Image For Retina.
    The Image Will Appear In The Left Pane.

07) In Left Pane > Right Click On <Project Name> Yellow Folder Icon >
    From The Context Menu, Choose ‘New Group’.
    Rename The ‘New Group’ To ‘Resources’ And Drag It Down As Last In ‘<Project Name> Group.
    Drag The png Image From previous Step Into This ‘Resources’ Group.
    Here Onwards We Will Use This Folder To Store Images Like Resources.

08) At The Top Of Xcode UI, There Is <Active scheme> Dropdown.
    Choose ‘iPhone 8 Plus Emulator’.

09) From Xcode Menubar, Choose : Product > Build Or Press Command+B

10) From Xcode Menubar, Choose : Product > Run Or Press Command+R
    A Message Like : ‘[CATransaction synchronize] called within transaction’
    In ‘Debug Area’ At The Bottom Of Xcode IDE.
    Ignore It. It Is Neither Warning Nor Error Message. It Is An Informative Message
    Regarding ‘Core Animation’.

11) When Needed, From Xcode Menubar, Choose : Product > Clean Or Press Shift+Command+K

12) Emulator Will Appear With Our Application Running And Showing ‘Black’ Colored Fullscreen Window.
    Press ‘Home’ Button To Let Window Disappear.
    From Xcode’s Top Button Bar, Choose ’Stop the running scheme or application’ Button.

===================================================================================================================
13) Similar To Step 7, Again Go To Left Pane > Right Click On <Project Name> Yellow Folder Icon >
    From The Context Menu, Choose ‘New File’.
    From iOS > Source > Choose ‘Cocoa Touch Class’ And Press ‘Next’.

    ‘Choose a template for your new file’ Dialog Will Appear.
    In ‘Class:”’ Field, Type ‘MyView’
    In Subclass of’ Field Type Or Choose ‘UIView’
    In ‘Language’ Field, Choose ‘Objective-C’
    Press ’Next’.
    Nothing Special To Do Here. We Will Be Brought To Our Project’s Finder Location.
    Just See That The Top Of Dialog Is Showing Our Project’s Name i.e. ‘Window’
    And Below, ’Targets’ Checkbox With Our Project Name, Is Checked.
    Press ‘Create’

    Two Files, MyView.m And MyView.h Files Will Be Created And Will Be Visible 
    Somewhere In ‘Left Pane’.
    Just For The Sake Of Sanity, Drag Them Under <Project Name> Yellow Folder,
    Between ViewController.m And Main.storyboard.

    This Arrangement, As I Said, Is Just For Sanity Indicating The Work Flow.
    i.e. AppDelegate Creates ViewController And ViewController Contains MyView.

14) ARC ( Automatic Reference Counting ) : Make It ‘Off’
    - In Left Pane, Click Top Level Project Name File ( Blue Icon )
    - In Middle Pane, Go To “Build Settings”, In It Go To “Apple LLVM - Language - Objective C”
      Category, Inside It Look For “Objective-C Automatic Reference Counting”, And Select “No”.

15) Write Empty dealloc Methods ( With Only Call To super ) In AppDelegate.m, ViewController.m
    And MyView.m

16) Repeat Clean, Build And Run Steps.
    Press ‘Home’ Button To Let Window Disappear.
    From Xcode’s Top Button Bar, Choose ’Stop the running scheme or application’ Button.

17) In Left Pane, Click On ‘Info.plist’ And Then In Middle Pane,
    a) Bring Mouse Nearer To ’Supported interface orientation’ Option
       And When + and - Signs Appear, Click On ‘-‘ Sign To Remove This Option.
    b) Repeat Above Step To Remove ’Supported interface orientation (iPad)’ Option.
    c) Repeat Clean, Build And Run Steps.
       Press ‘Home’ Button To Let Window Disappear.
       From Xcode’s Top Button Bar, Choose ’Stop the running scheme or application’ Button.

18) Remove Interface Block From Both AppDelegate.m And ViewController.m
    And Write Given Code In AppDelegate.m, ViewController.m And MyView.m 
    Along With Already Existing Code.
    And Repeat Clean, Build And Run Steps.
    Press ‘Home’ Button To Let Window Disappear.
    From Xcode’s Top Button Bar, Choose ’Stop the running scheme or application’ Button.

===================================================================================================================
