from neuronumba.basic.attr import HasAttr, Attr


class Integrator(HasAttr):
    dt = Attr(default=None, required=True)



