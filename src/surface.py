import taichi as ti
import taichi.math as tm
from collections import OrderedDict

@ti.dataclass
class Ray3d:
    ro: tm.vec3
    rd: tm.vec3
    re: tm.vec3
    t: float

    @ti.func
    def ray_propergate(self, t):
        self.t = t
        self.re = self.ro + self.t * self.rd

    @ti.func
    def ray_sec_plane(self, height):
        self.t = (height - self.ro.z) / self.rd.z
        self.re = self.ro + self.t * self.rd

    @ti.func
    def ray_sec_surface(self, surf3d):
        # init_height
        self.ray_sec_plane(surf3d.height[None])
        
        ti.loop_config(serialize=True)
        for i in range(10): 
            delta = surf3d.curve_func(self.re.x, self.re.y) - self.re.z
            self.t -= delta / (tm.dot(self.rd, surf3d.curve_normal_func(self.re.x, self.re.y)))
            self.re = self.ro + self.t * self.rd
    
    @ti.func
    def ray_sec_sphere(self, surf3d: ti.template()):
        if ti.abs(surf3d.curvature[None]) < 1e-6 :
            self.ray_sec_plane(surf3d.height[None])
        else:
            r = 1/surf3d.curvature[None]
            origin = tm.vec3([0,0,r])
            d = origin-self.ro
            project = tm.dot(d, self.rd)
            delta = ti.sqrt(r**2 - tm.dot(d,d) + project**2)
            sign_r = tm.dot(self.rd, origin)

            if(sign_r > 0):
                self.t = project - delta
            else:
                self.t = project + delta
            self.re = self.ro + self.t * self.rd
        
        self.surf_to_global(surf3d.pos[None])

    @ti.func
    def ray_reflct_surface(self, surface):
        self.ro = self.re
        
        self.rd = tm.normalize(tm.refract(self.rd, surface.curve_normal_func(self.re.x, self.re.y), surface.n_in/surface.n_out))
        
        # In_v = surface.n_in * self.rd
        # N_v = surface.curve_normal_func(self.re.x, self.re.y)
        # In_on_N = tm.dot(In_v, -N_v)
        # Out_on_N = ti.sqrt(surface.n_out**2 - surface.n_in**2 + In_on_N**2)
        # delta = In_on_N - Out_on_N
        # self.rd = tm.normalize(self.rd + delta * N_v)

@ti.data_oriented
class Rays_3d() :
    def __init__(self, ray_nums, fov_nums, width, fov, for_show=False):
        self.fov_nums = fov_nums
        if(for_show):
            self.ray_field = Ray3d.field(shape=(ray_nums, fov_nums), needs_grad=False)
            self.ray_nums = ray_nums
            self.build_rays_uniform(width,fov)
        else:
            self.ray_field = Ray3d.field(shape=(ray_nums*ray_nums, fov_nums), needs_grad=True)
            self.ray_nums = ray_nums*ray_nums
            self.build_rays_random(width,fov)

    # uniform ray bundle for show
    @ti.kernel
    def build_rays_uniform(self, width:ti.f32, fov:ti.f32):
        delta_pos = 2*width / (self.ray_nums-1)
        delta_angle = fov / (self.fov_nums-1)
        for i in ti.grouped(self.ray_field):
            self.ray_field[i].ro.x = delta_pos * i[0] - width
            self.ray_field[i].ro.y = 0
            self.ray_field[i].ro.z = -30
            self.ray_field[i].rd = [tm.sin((delta_angle * i[1])/180.0 * tm.pi), 0,
                                    tm.cos((delta_angle * i[1])/180.0 * tm.pi)]

    # random ray bundle for optimization
    @ti.kernel
    def build_rays_random(self, width:ti.f32, fov:ti.f32):
        delta_angle = fov / (self.fov_nums-1)
        for i in ti.grouped(self.ray_field):
            self.ray_field[i].ro.x = 2*(ti.random() - 0.5)*width
            self.ray_field[i].ro.y = 2*(ti.random() - 0.5)*width
            while(self.ray_field[i].ro.x**2 + self.ray_field[i].ro.y**2 > width **2):
                self.ray_field[i].ro.x = 2*(ti.random() - 0.5)*width
                self.ray_field[i].ro.y = 2*(ti.random() - 0.5)*width
            self.ray_field[i].ro.z = -30
            self.ray_field[i].rd = [tm.sin((delta_angle * i[1])/180.0 * tm.pi), 0,
                                    tm.cos((delta_angle * i[1])/180.0 * tm.pi)]

    @ti.kernel
    def intersect_with_plane(self, z:ti.f32):
        for i in ti.grouped(self.ray_field):
            self.ray_field[i].ray_sec_plane(z)
    
    @ti.kernel
    def propergate(self, t:ti.f32):
        for i in ti.grouped(self.ray_field):
            self.ray_field[i].ray_propergate(t)

@ti.data_oriented
class aspherical_3d():

    def __init__(self, height, curvature, n_in, n_out, max_order=10) -> None:
        self.height = ti.field(ti.f32, shape=(), needs_grad=True)
        self.curvature = ti.field(ti.f32, shape=(), needs_grad=True)

        self.n_in = n_in
        self.n_out = n_out

        self.height[None] = height
        self.curvature[None] = curvature

        self.usage_order = max_order
        self.max_order = max_order

    @ti.func
    def curve_func(self, x: ti.f32, y:ti.f32) -> ti.f32:
        k = (self.curvature[None])**2*(x**2+y**2)
        tmp_a = ti.sqrt(1-k)
        sum = (self.curvature[None])*(x**2+y**2) / (1+tmp_a) + self.height[None]
        return sum

    @ti.func
    def curve_tangent_vec(self, x:ti.f32, y:ti.f32) -> tm.vec2:
        r2 = x**2+y**2
        k = self.curvature[None]**2*r2
        tmp_a = ti.sqrt(1-k)
        der_x = 2*x*self.curvature[None] * (1 + tmp_a - k/2) / (tmp_a*(1+tmp_a)**2)
        der_y = 2*y*self.curvature[None] * (1 + tmp_a - k/2) / (tmp_a*(1+tmp_a)**2)
        return ti.Vector([der_x, der_y])
    
    @ti.func
    def curve_normal_func(self, x:ti.f32, y:ti.f32) -> ti.math.vec3:
        der = self.curve_tangent_vec(x, y)
        N = tm.vec3([der.x, der.y, -1])
        return N / ti.sqrt(N.x**2+N.y**2+N.z**2)
    
    # @ti.func
    # def curve_normal_func(self, x:ti.f32, y:ti.f32) -> ti.math.vec3:
    #     N = tm.vec3([self.curve_tangent_vec(x), self.curve_tangent_vec(y), -1])
    #     return N / ti.sqrt(N.x**2+N.y**2+N.z**2)

    @ti.kernel
    def get_curve(self, out_points: ti.template(), width: ti.f32, point_num:ti.i32) :
        delta = (width*2) / (point_num-1)    
        for i in range(point_num):
            out_points[i] = [(delta*i-width), self.curve_func((delta*i-width), 0)]
    
    def set_height(self, height):
        self.height[None] = height
    
    def set_curvature(self, curvature):
        self.curvature[None] = curvature

@ti.kernel
def intersection(ray_in: ti.template(), surf: ti.template()):
    for i in ti.grouped(ray_in.ray_field):
        ray_in.ray_field[i].ray_sec_surface(surf)

@ti.ad.no_grad
def intersection_no_grad(ray_in: ti.template(), surf: ti.template()):
    intersection(ray_in, surf)

@ti.kernel
def refract(ray_in: ti.template(), surf: ti.template()):
    for i in ti.grouped(ray_in.ray_field):
        ray_in.ray_field[i].ray_reflct_surface(surf)
