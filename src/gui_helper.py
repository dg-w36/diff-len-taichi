import taichi as ti
import taichi.math as tm
import matplotlib as mpl

# helper function for gui
@ti.func
def real_to_relative(points: ti.template(), real_region: ti.template(), relative_region: ti.template()):
    # [x_min,x_max,y_min,y_max] = real_region e.g. [-20,20,-20,20]
    # [rx_min,rx_max,ry_min,ry_max] = relative_region e.g. [0,0.5,0,1]
    for i in points:
        tmp_y = (points[i].x -  real_region[0]) / ( real_region[1] -  real_region[0])
        tmp_x = (points[i].y -  real_region[2]) / ( real_region[3] -  real_region[2])
        points[i].x = ( relative_region[1] -  relative_region[0]) * tmp_x +  relative_region[0]
        points[i].y = ( relative_region[3] -  relative_region[2]) * tmp_y +  relative_region[2]

@ti.data_oriented
class draw_curve() :
    def __init__(self, point_num , real_region_curve, relative_region_curve) -> None:
        self.curve_points = ti.Vector.field(2, ti.f32, shape=(2*point_num-2))
        self.point_num = point_num
        self.real_region_curve = tm.vec4(real_region_curve)
        self.relative_region_curve = tm.vec4(relative_region_curve)

    @ti.kernel
    def prepare_curve(self, points: ti.template()):
        self.curve_points[0] = points[0]
        self.curve_points[2*self.point_num-3] = points[self.point_num-1]
        for i in range(1, self.point_num-1):
            self.curve_points[2*i-1] = points[i]
            self.curve_points[2*i] = points[i]
        real_to_relative(self.curve_points, self.real_region_curve, self.relative_region_curve)

    def show_curve(self, canvas, curve_points, choose=False):
        self.prepare_curve(curve_points)
        color_tmp = (0.68, 0.68, 0.19) if choose else (0.68, 0.26, 0.19)
        canvas.lines(self.curve_points, width=0.005, color = color_tmp)

@ti.data_oriented
class draw_rays() :
    def __init__(self, rays_num, fov_num, real_region_rays, relative_region_rays) -> None:
        self.ray_points = ti.Vector.field(2, ti.f32, shape=(2*rays_num))
        self.rays_num = rays_num
        self.fov_num = fov_num

        self.real_region_rays = tm.vec4(real_region_rays)
        self.relative_region_rays = tm.vec4(relative_region_rays)
    
    @ti.kernel
    def prepare_ray(self, rays: ti.template(), index: int):
        for i in range(rays.ray_nums):
            self.ray_points[2*i] = rays.ray_field[i, index].ro
            self.ray_points[2*i+1] = rays.ray_field[i, index].re

        real_to_relative(self.ray_points, self.real_region_rays, self.relative_region_rays)
    
    def show_curve(self, canvas, rays):
        cmap = mpl.colormaps['Accent']
        for i in range(self.fov_num):
            self.prepare_ray(rays, i)
            canvas.lines(self.ray_points, width=0.0015, color = cmap(i)[:3])

    @ti.kernel
    def prepare_ray_3d(self, rays: ti.template(), index: int):
        for i in range(rays.ray_nums):
            self.ray_points[2*i] = [rays.ray_field[i, index].ro.x, rays.ray_field[i, index].ro.z]
            self.ray_points[2*i+1] = [rays.ray_field[i, index].re.x, rays.ray_field[i, index].re.z]

        real_to_relative(self.ray_points, self.real_region_rays, self.relative_region_rays)

    def show_curve_3d(self, canvas, rays):
        cmap = mpl.colormaps['Accent']
        for i in range(self.fov_num):
            self.prepare_ray_3d(rays, i)
            canvas.lines(self.ray_points, width=0.0015, color = cmap(i)[:3])

@ti.data_oriented
class draw_spot_diagram():
    def __init__(self, spot_num, fov_num, real_region, relative_region) -> None:
        self.spot_points = ti.Vector.field(2, ti.f32, shape=(spot_num))
        self.block_points = ti.Vector.field(2, ti.f32, shape=(8))
        self.spot_num = spot_num
        self.fov_num = fov_num

        self.real_region = real_region
        self.relative_region = relative_region
        self.block_points[0] = [self.relative_region[0], self.relative_region[2]]
        self.block_points[1] = [self.relative_region[1], self.relative_region[2]]
        self.block_points[2] = [self.relative_region[0], self.relative_region[2]]
        self.block_points[3] = [self.relative_region[0], self.relative_region[3]]
        self.block_points[4] = [self.relative_region[1], self.relative_region[3]]
        self.block_points[5] = [self.relative_region[1], self.relative_region[2]]
        self.block_points[6] = [self.relative_region[1], self.relative_region[3]]
        self.block_points[7] = [self.relative_region[0], self.relative_region[3]]


    @ti.kernel
    def prepare_spot(self, rays: ti.template(), index: int):
        for i in range(rays.ray_nums):
            self.spot_points[i] = rays.ray_field.re[i, index].xy
        real_to_relative(self.spot_points, self.real_region, self.relative_region)

    def show_spot(self, canvas, rays):
        cmap = mpl.colormaps['Accent']
        for i in range(self.fov_num):
            self.prepare_spot(rays, i)
            canvas.circles(self.spot_points, radius=0.0015, color = cmap(i)[:3])
        canvas.lines(self.block_points, width=0.0015, color = (1.0,0.0,0.0))

@ti.kernel
def get_psf(rays: ti.template(), psf_img: ti.types.ndarray(), region: ti.template(), shape: ti.template()) :    
    #[xmin, xmax, ymin, ymax] = region
    # shape is psf shape
    delta_x = (region[1] - region[0])/shape.x
    delta_y = (region[3] - region[2])/shape.x
    for i in ti.grouped(rays.ray_field):
        index_x = int((rays.ray_field[i].re.x - region[0]) // delta_x)
        index_y = int((rays.ray_field[i].re.y - region[2]) // delta_y)
        psf_img[index_x, index_y, 0] += (1.0/rays.ray_nums)
        psf_img[index_x, index_y, 1] += (1.0/rays.ray_nums)
        psf_img[index_x, index_y, 2] += (1.0/rays.ray_nums)
