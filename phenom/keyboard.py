from OpenGL.GL import *
from OpenGL.GLUT import *
from phenom.animator import *

from noumena.complex import *

class KeyboardHandler:

    def keyboard(self, key, x, y):

        if(key == "\033"):
            self.engine.cleanup()
            self.cleanup()
            exit()

        elif(key == "`"):
            if(self.state.manual_iter):
                self.engine.next_frame = True
            self.state.manual_iter = not self.state.manual_iter

        elif(key == "\040"):
            self.engine.next_frame = True            

        elif(key in ["a", "s", "d", "f", "g"]):
            i = ["a", "s", "d", "f", "g"].index(key)
            z0 = r_to_p(self.state.zn[i])
            z1 = list(z0)
            z1[0] += 0.1
            self.animator.animate_var("self.state.zn[" + str(i) + "]", "radial_2d", 200, {"s":z0, "e":z1})

        elif(key in ["z", "x", "c", "v", "b"]):
            i = ["z", "x", "c", "v", "b"].index(key)
            z0 = r_to_p(self.state.zn[i])
            z1 = list(z0)
            z1[0] -= 0.1
            self.animator.animate_var("self.state.zn[" + str(i) + "]", "radial_2d", 200, {"s":z0, "e":z1})

        elif(key in ["A", "S", "D", "F", "G"]):
            i = ["A", "S", "D", "F", "G"].index(key)
            z0 = r_to_p(self.state.zn[i])
            z1 = list(z0)
            z1[1] += 2 * pi / 16
            self.animator.animate_var("self.state.zn[" + str(i) + "]", "radial_2d", 200, {"s":z0, "e":z1})

        elif(key in ["Z", "X", "C", "V", "B"]):
            i = ["Z", "X", "C", "V", "B"].index(key)
            z0 = r_to_p(self.state.zn[i])
            z1 = list(z0)
            z1[1] -= 2 * pi / 16
            self.animator.animate_var("self.state.zn[" + str(i) + "]", "radial_2d", 200, {"s":z0, "e":z1})

