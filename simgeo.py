
import math


class Rect(object):

    def __init__(self,tl_x,tl_y,width,height):
        self.x = int(tl_x)
        self.y = int(tl_y)
        self.w = int(width)
        self.h = int(height)

    def __str__(self):
        return 'x:%d y:%d w:%d h:%d'%(self.x,self.y,self.w,self.h)

    def get_int_rect(self):

        dr_x = int(self.x+self.w-1+0.5)
        dr_y = int(self.y+self.h-1+0.5)
        x = int(self.x+0.5)
        y = int(self.y+0.5)
        return Rect.from_points(x,y,dr_x,dr_y)

    @staticmethod
    def from_points(tl_x,tl_y,dr_x,dr_y):
        return Rect(tl_x,tl_y,dr_x-tl_x+1,dr_y-tl_y+1)

    def get_copy(self):
        return Rect(self.x,self.y,self.w,self.h)

    def get_center(self):
        return self.x + (self.w-1)/2.0, self.y + (self.h-1)/2.0

    def get_tl(self):
        return self.x,self.y

    def get_dr(self):
        return self.x + self.w -1, self.y+self.h-1

    def get_area(self):
        area = self.w*self.h
        if area <= 0:
            area = 0
        return area

    def get_top(self):
        return self.y

    def get_left(self):
        return self.x

    def get_right(self):
        return self.x + self.w -1

    def get_bottom(self):
        return self.y + self.h -1

    def get_intersect_rect(self,r):
        tl_x = max(self.x,r.x)
        tl_y = max(self.y,r.y)
        dr_x = min(self.x+self.w,r.x+r.w) -1
        dr_y = min(self.y+self.h,r.y+r.h) -1
        return Rect.from_points(tl_x,tl_y,dr_x,dr_y)

    def get_intersect_ratio(self,r):
        ir_area = float(self.get_intersect_rect(r).get_area())
        self_area = float(self.get_area())
        r_area = float(r.get_area())

        u_area = self_area+r_area-ir_area
        if abs(u_area) < 1e-6:
            return 0
        return ir_area/u_area

    def scale_from_center(self,x_scale,y_scale): # scale with center unchanged
        assert x_scale > 0 and y_scale > 0

        sw = int(self.w * x_scale + 0.5)
        sh = int(self.h * y_scale + 0.5)
        cx, cy = self.get_center()

        tl_x = round(cx - (sw-1)/2.0)
        tl_y = round(cy - (sh-1)/2.0)

        return Rect(tl_x,tl_y,sw,sh)

    def is_in_rect(self,rect):
        assert self.w > 0 and self.h > 0
        return self.x >= rect.x and self.y >= rect.y and self.w + self.x <= rect.w and self.h + self.y <= rect.h



