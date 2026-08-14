# Makie widget constructors and the enable/disable plumbing the control panel uses.

# Few helpers (height=18 keeps each widget within its Fixed(18) grid row, otherwise the
# default widget heights are taller than the row and stick out past the colored Box behind
# it; Textbox additionally needs fontsize/textpadding reduced from their defaults, since the
# default 8px top+bottom textpadding alone exceeds the 18px row, clipping the displayed text):
add_textbox(fig, label, value) = [Label(fig, label), Textbox(fig, stored_string = string(value), validator = typeof(value), height = 18, fontsize = 11, textpadding = (4, 4, 2, 2))]
add_togglebox(fig, label, active) = [Label(fig, label), Toggle(fig, active = active, height = 18)]
get_valuebox(box::Vector) = parse(box[2].validator.val, box[2].stored_string.val)

function set_textbox_enabled!(box, enabled)
    label, textbox = box
    label.color = enabled ? :black : (:gray, 0.55)
    textbox.textcolor = enabled ? :black : (:gray, 0.65)
    textbox.boxcolor = enabled ? :transparent : (:gray, 0.15)
    textbox.boxcolor_hover = enabled ? :transparent : (:gray, 0.15)
    textbox.boxcolor_focused = enabled ? :transparent : (:gray, 0.15)
    textbox.bordercolor = enabled ? (:gray, 0.8) : (:gray, 0.45)
    textbox.bordercolor_hover = enabled ? (:gray, 0.55) : (:gray, 0.45)
    enabled || GLMakie.Makie.defocus!(textbox)
    return nothing
end

function bind_textbox_enabled!(box, enabled)
    textbox = box[2]
    on(enabled) do active
        set_textbox_enabled!(box, active)
    end
    on(textbox.focused) do focused
        !enabled[] && focused && GLMakie.Makie.defocus!(textbox)
    end
    set_textbox_enabled!(box, enabled[])
    return nothing
end

function set_menu_enabled!(menu, enabled)
    menu.textcolor = enabled ? :black : (:gray, 0.55)
    menu.dropdown_arrow_color = enabled ? (:black, 0.2) : (:gray, 0.45)
    enabled || (menu.is_open[] = false)
    return nothing
end

function bind_menu_enabled!(menu, enabled)
    on(enabled) do active
        set_menu_enabled!(menu, active)
    end
    on(menu.is_open) do open
        !enabled[] && open && (menu.is_open[] = false)
    end
    set_menu_enabled!(menu, enabled[])
    return nothing
end
