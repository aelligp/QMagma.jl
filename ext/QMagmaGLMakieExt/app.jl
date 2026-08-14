# Entry point of the interactive app: window sizing, wiring and display.

function set_qmagma_window_icon!(screen)
    path = normpath(@__DIR__, "..", "..", "docs", "src", "assets", "favicon.ico")
    isfile(path) || error("QMagma window icon not found at $path")
    if Sys.isapple()
        # GLFW ignores SetWindowIcon on macOS, so set the Cocoa application icon directly.
        app_class = ccall(:objc_getClass, Ptr{Cvoid}, (Cstring,), "NSApplication")
        shared = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "sharedApplication")
        app = ccall(:objc_msgSend, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), app_class, shared)
        string_class = ccall(:objc_getClass, Ptr{Cvoid}, (Cstring,), "NSString")
        from_utf8 = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "stringWithUTF8String:")
        ns_path = ccall(
            :objc_msgSend, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Cstring),
            string_class, from_utf8, path
        )
        image_class = ccall(:objc_getClass, Ptr{Cvoid}, (Cstring,), "NSImage")
        alloc = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "alloc")
        image = ccall(:objc_msgSend, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), image_class, alloc)
        init = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "initWithContentsOfFile:")
        image = ccall(
            :objc_msgSend, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
            image, init, ns_path
        )
        image == C_NULL && error("macOS could not load the QMagma window icon at $path")
        set_icon = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "setApplicationIconImage:")
        ccall(
            :objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
            app, set_icon, image
        )
    else
        image = GLMakie.Makie.FileIO.load(path)
        slices = ndims(image) == 2 ? (image,) : eachslice(image; dims = 3)
        icons = [reinterpret(NTuple{4, UInt8}, Matrix(slice)) for slice in slices]
        GLMakie.GLFW.SetWindowIcon(screen.glscreen, icons)
        GLMakie.GLFW.PollEvents()
    end
    return nothing
end

function sill_intrusion_1D(; size = nothing)
    GLMakie.activate!()
    GLMakie.closeall() # close any open screen

    if size === nothing
        win_w = 1500
        win_h = 900
        try
            monitor = GLMakie.GLFW.GetPrimaryMonitor()
            vidmode = GLMakie.GLFW.GetVideoMode(monitor)
            # GetVideoMode reports the full screen height; subtract a fixed margin for
            # the menu bar/dock/title bar (not otherwise queryable here) rather than a
            # percentage, since the menu bar's height doesn't scale with screen size
            win_h = min(win_h, vidmode.height - 130)
        catch e
            @warn "Could not query monitor size; using default window height" exception = e
        end
        size = (win_w, win_h)
    end

    ui = build_layout(size)
    wire_buttons!(ui)
    wire_simulation!(ui)

    screen = display(ui.fig; title = "QMagma")
    set_qmagma_window_icon!(screen)

    # center the window on the primary monitor
    try
        monitor = GLMakie.GLFW.GetPrimaryMonitor()
        vidmode = GLMakie.GLFW.GetVideoMode(monitor)
        win_w, win_h = size
        x = max(0, div(vidmode.width - win_w, 2))
        y = max(0, div(vidmode.height - win_h, 2))
        GLMakie.GLFW.SetWindowPos(screen.glscreen, x, y)
    catch e
        @warn "Could not center the window automatically" exception = e
    end

    # Force GLMakie's resize-driven relayout once: the bottom chamber panel (a nested
    # GridLayout in the figure's 2nd row, added after the 1st row) is otherwise not measured
    # until the user manually resizes, so it renders off-window on first display. Nudging the
    # window size by 1px and back fires the resize callback that recomputes the whole layout.
    try
        win_w, win_h = size
        GLMakie.GLFW.SetWindowSize(screen.glscreen, win_w, win_h - 1)
        GLMakie.GLFW.SetWindowSize(screen.glscreen, win_w, win_h)
    catch e
        @warn "Could not nudge the window size to force the initial relayout" exception = e
    end

    return nothing
end
