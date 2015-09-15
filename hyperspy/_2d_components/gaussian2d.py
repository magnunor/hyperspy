# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import math

import numpy as np

from hyperspy.component import Component

pi2 = 2 * math.pi


class Gaussian2D(Component):

    """Normalized 2D eliptical gaussian function component

    .. math::

        f(x,y) = \\frac{A}{2\pi s_x s_y}}}e^{-\\frac{\left(x-x0\\right)^{2}}{2s_{x}^{2}}                     \\frac{\left(y-y0\\right)^{2}}{2s_{y}^{2}}}

    +------------+-----------+
    | Parameter  | Attribute |
    +------------+-----------+
    +------------+-----------+
    |      a     | amplitude |
    +------------+-----------+
    |    x0,y0   |  centre   |
    +------------+-----------+
    |   s_x,s_y  |   sigma   |
    +------------+-----------+
    |    theta   | rotation  |
    +------------+-----------+

    """

    def __init__(self,
                 A=1.,
                 sigma_x=1.,
                 sigma_y=1.,
                 centre_x=0.,
                 centre_y=0.,
                 rotation=0.,
                ):
        Component.__init__(self, ['A',
                                  'sigma_x',
                                  'sigma_y',
                                  'centre_x',
                                  'centre_y',
                                  'rotation',
                                 ])
        self.A.value = A
        self.sigma_x.value = sigma_x
        self.sigma_y.value = sigma_y
        self.centre_x.value = centre_x
        self.centre_y.value = centre_y
        self.rotation.value = rotation
        self.rotation.free = False

    #TODO: add in boundaries and gradients for enhancement

    def function(self, x, y):
        A = self.A.value
        sx = self.sigma_x.value
        sy = self.sigma_y.value
        x0 = self.centre_x.value
        y0 = self.centre_y.value
        theta = self.rotation.value

        temp_2sx2 = 2*sx**2
        temp_2sy2 = 2*sy**2
        temp_cos2_theta = math.cos(theta)**2
        temp_sin2_theta = math.sin(theta)**2
        temp_sin_theta2 = math.sin(2*theta)

        a = temp_cos2_theta/temp_2sx2 + temp_sin2_theta/temp_2sy2
        b = -temp_sin_theta2/(2*temp_2sx2) + temp_sin_theta2/(2*temp_2sy2)
        c = temp_sin2_theta/temp_2sx2 + temp_cos2_theta/temp_2sy2

        return A * (1 / (sx * sy * pi2)) * np.exp(
            -(a*(x - x0) ** 2 +
              2*b*(x - x0) * (y - y0) + 
              c*(y - y0) ** 2))

   #TODO: add further useful properties of 2D gaussian e.g. ellipticity (sigma ratio), fwhm in each direction...
