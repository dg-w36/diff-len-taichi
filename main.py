import taichi as ti
from src.surface import *
from src.gui_helper import *

# init taichi and prepare GGUI
ti.init(ti.cuda)
window = ti.ui.Window('lens', res = (1200, 800), pos = (150, 150) )
canvas = window.get_canvas()
gui = window.get_gui() 

# prepare surface
surface_num = 4
surf3d_list = [None for i in range(surface_num)]
init_h = -10
for i in range(surface_num):
    if(i%2 == 0):
        surf3d_list[i] = aspherical_3d(init_h, 0.01, 1, 1.5)
    else:
        surf3d_list[i] = aspherical_3d(init_h, -0.01, 1.5, 1)
    init_h += 3

# helper for draw surface
point_num = 200
draw_helper = draw_curve(point_num, [-20,20,-20,20],[0,0.67,0,1])
tmp_curve = ti.Vector.field(2, ti.f32, shape=point_num)

# params of rays
rays_num = 10
fov_num = 3
ray_helper = draw_rays(rays_num, fov_num, [-20,20,-20,20],[0,0.67,0,1])
gui_rays = Rays_3d(rays_num, fov_num, 9, 0, for_show=True)
opt_rays = Rays_3d(rays_num*5, fov_num, 9, 0)
spot_rays = Rays_3d(rays_num, fov_num, 9, 0)
ray_bundle_width = 3

# helper for draw spot diagram
spot_helper = draw_spot_diagram(opt_rays.ray_nums, fov_num, [-4,4,-2,2],[0.667,1,0,1])

# loss function and var for optimization
loss_mean = ti.Vector.field(2, dtype=ti.f32, shape=(fov_num), needs_grad=True)
loss = ti.field(float, shape=(), needs_grad=True)
min_loss = 1e10
down_count = 0
@ti.kernel
def mse_loss():
    for i,j in opt_rays.ray_field:
        loss_mean[j] += opt_rays.ray_field[i,j].re.xy / opt_rays.ray_nums
    for i,j in opt_rays.ray_field:
        error_v = (opt_rays.ray_field[i,j].re.xy-loss_mean[j])
        loss[None] += (error_v.x**2 +error_v.y**2) / opt_rays.ray_nums

# forward once to warm up device
opt_rays.build_rays_random(ray_bundle_width, 1)
with ti.ad.Tape(loss):
    for surf in surf3d_list:
        intersection(opt_rays, surf)
        refract(opt_rays, surf)
    opt_rays.intersect_with_plane(15)
    mse_loss()

# var for slider
tmp_height = 0
tmp_radius = 0.5
surf_index = 0
mode_setting = True
lr_slider = 25
opt_count = 0
fov_slider = 1
max_count = 100
click = False

# main loop
while window.running:
    if(mode_setting) : # setting mode
        with gui.sub_window("surface params", 0, 0, 0.667, 0.15) as w:
            surf_index = w.slider_int("surface index", surf_index, 0, surface_num-1)
            tmp_height = w.slider_float("surface height", surf3d_list[surf_index].height[None], -20, 10)
            tmp_radius = w.slider_float("surface curvature", surf3d_list[surf_index].curvature[None], -0.1, 0.1)
            fov_slider = w.slider_float("number of FOV", fov_slider, 0, 50)
        with gui.sub_window("opt params", 0, 0.85, 0.667, 0.15) as w:
            lr_slider = w.slider_float("lr", lr_slider, 1, 100)
            max_count = w.slider_float("opt cound", max_count, 1, 100)
            click = w.button("start opt")
        
        # update params from slider
        surf3d_list[surf_index].set_height(tmp_height)
        surf3d_list[surf_index].set_curvature(tmp_radius)
        
        lr_real = lr_slider/1e6
        if(click):
            mode_setting = False
            opt_count = 0
        
    else: # optmization mode
        if(opt_count >= max_count):
            mode_setting = True
            continue
        
        # forward  
        loss_mean.fill(0.0)
        opt_rays.build_rays_random(ray_bundle_width,fov_slider)
        with ti.ad.Tape(loss):
            for surf in surf3d_list:
                intersection_no_grad(opt_rays, surf)
                refract(opt_rays, surf)
            opt_rays.intersect_with_plane(15)
            mse_loss()

        if(min_loss > loss[None]):
            min_loss = loss[None]

        # update loss massage
        with gui.sub_window("optimization", 0, 0, 0.667, 0.1) as w:
            w.text("loss is %f"%(loss[None]))
        
        # gradient descent update
        for surf in surf3d_list:
            surf.height[None] -= lr_real*1e4 * surf.height.grad[None]
            surf.curvature[None] -= lr_real * surf.curvature.grad[None]
            
            # update high order params
            surf.params[1] -= (lr_real/1e5) * surf.params.grad[1]

            # print(surf.height[None], surf.height.grad[None], end="/")
            # print(surf.curvature[None], surf.curvature.grad[None])
            # print(surf.params[0], surf.params[1], surf.params[2])

        print("loss is: ", loss[None])
        opt_count += 1

    spot_rays.build_rays_random(ray_bundle_width, fov_slider)
    for surf in surf3d_list:
        intersection_no_grad(spot_rays, surf)
        refract(spot_rays, surf)
    spot_rays.intersect_with_plane(15)
    spot_helper.show_spot(canvas, spot_rays)

    # draw curve and ray
    gui_rays.build_rays_uniform(ray_bundle_width,fov_slider)
    for i in range(surface_num):
        intersection(gui_rays, surf3d_list[i])
        ray_helper.show_curve_3d(canvas, gui_rays)
        
        surf3d_list[i].get_curve(tmp_curve, 10, point_num)
        draw_helper.show_curve(canvas, tmp_curve, i==surf_index)
        
        refract(gui_rays, surf3d_list[i])
    
    gui_rays.intersect_with_plane(15)
    ray_helper.show_curve_3d(canvas, gui_rays)

    window.show()