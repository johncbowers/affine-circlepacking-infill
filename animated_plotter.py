import vedo

class AnimatedPlotter:

    def __init__(self, update_fn, draw_fn, dt=16, cam=dict(pos=(0, 0, 5), focalPoint=(0, 0, 0), viewup=(0, 1, 0), distance=5)):
        self.update_fn = update_fn
        self.draw_fn = draw_fn
        self.dt = dt
        self.cam = cam
        self.plotter = vedo.Plotter(size=(1600, 1600))
    
    def show(self):
        self.plotter.addCallback("timer", self._updateAndDraw)
        self.timerId = self.plotter.timerCallback("create", dt=self.dt)
        return self.plotter.show(camera=self.cam).close()

    def _updateAndDraw(self, evt):
        self.update_fn()
        self.draw_fn(self.plotter)

# viewer = Viewer(axes=1, dt=20).initialize()

# viewer.plotter += vedo.Cube()
# viewer.plotter += vedo.Sphere(r=0.1).x(1.5)
# viewer.plotter += "Sphere color is"

# viewer.plotter.show()