from .. import _c_ops
from .integration import sTOp, rTReflectedOp
from .rotation import dotRxyOp, dotRxyTOp, dotRzOp, rotateOp
from .filter import FOp
from .misc import spotYlmOp, pTOp
from .utils import *
import theano
import theano.tensor as tt
import theano.sparse as ts
from theano.ifelse import ifelse
import numpy as np
import logging


__all__ = ["Ops", "OpsReflected", "OpsRV"]


class Ops(object):
    """
    Everything in radians here.
    Everything is a Theano operation.

    """

    def __init__(self, ydeg, udeg, fdeg, nw, lazy, quiet=False):
        """

        """
        # Logging
        if quiet:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)

        # Instantiate the C++ Ops
        self.ydeg = ydeg
        self.udeg = udeg
        self.fdeg = fdeg
        self.deg = (ydeg + udeg + fdeg)
        self.filter = (fdeg > 0) or (udeg > 0)
        self._c_ops = _c_ops.Ops(ydeg, udeg, fdeg)
        self.nw = nw
        self.lazy = lazy
        if self.lazy:
            self.cast = to_tensor
        else:
            self.cast = to_array

        # Solution vectors
        self.sT = sTOp(self._c_ops.sT, self._c_ops.N)
        self.rT = tt.shape_padleft(tt.as_tensor_variable(self._c_ops.rT))
        self.rTA1 = tt.shape_padleft(tt.as_tensor_variable(self._c_ops.rTA1))

        # Change of basis matrices
        self.A = ts.as_sparse_variable(self._c_ops.A)
        self.A1 = ts.as_sparse_variable(self._c_ops.A1)
        self.A1Inv = ts.as_sparse_variable(self._c_ops.A1Inv)

        # Rotation operations
        self.R = self._c_ops.R
        self.apply_rotation = rotateOp(self._c_ops.rotate, self.nw)
        self.dotRz = dotRzOp(self._c_ops.dotRz)
        self.dotRxy = dotRxyOp(self._c_ops.dotRxy)
        self.dotRxyT = dotRxyTOp(self._c_ops.dotRxyT)

        # Filter
        self.F = FOp(self._c_ops.F, self._c_ops.N, self._c_ops.Ny)

        # Misc
        self.spotYlm = spotYlmOp(self._c_ops.spotYlm, self.ydeg, self.nw)
        self.pT = pTOp(self._c_ops.pT, self.deg)

        # Map rendering
        self.rect_res = 0
        self.ortho_res = 0
    
    def compute_ortho_grid(self, res):
        """
        Compute the polynomial basis on the plane of the sky.

        """
        dx = 2.0 / res
        y, x = tt.mgrid[-1:1:dx, -1:1:dx]
        x = tt.reshape(x, [1, -1])
        y = tt.reshape(y, [1, -1])
        z = tt.sqrt(1 - x ** 2 - y ** 2)
        return tt.concatenate((x, y, z))

    def compute_rect_grid(self, res):
        """
        Compute the polynomial basis on a rectangular lat/lon grid.

        """
        dx = np.pi / res
        lat, lon = tt.mgrid[-np.pi/2:np.pi/2:dx, -3*np.pi/2:np.pi/2:2*dx]
        x = tt.reshape(tt.cos(lat) * tt.cos(lon), [1, -1])
        y = tt.reshape(tt.cos(lat) * tt.sin(lon), [1, -1])
        z = tt.reshape(tt.sin(lat), [1, -1])
        R = RAxisAngle([1, 0, 0], -np.pi / 2)
        return tt.dot(R, tt.concatenate((x, y, z)))

    def dotR(self, M, inc, obl, theta):
        """

        """

        res = self.dotRxyT(M, inc, obl)
        res = self.dotRz(res, theta)
        res = self.dotRxy(res, inc, obl)
        return res

    @autocompile(
        "X", tt.dvector(), tt.dvector(), tt.dvector(), tt.dvector(), 
            tt.dscalar(), tt.dscalar(), tt.dscalar(), tt.dvector(), 
            tt.dvector()
    )
    def X(self, theta, xo, yo, zo, ro, inc, obl, u, f):
        """

        """
        # Compute the occultation mask
        b = tt.sqrt(xo ** 2 + yo ** 2)
        b_rot = (tt.ge(b, 1.0 + ro) | tt.le(zo, 0.0) | tt.eq(ro, 0.0))
        b_occ = tt.invert(b_rot)
        i_rot = tt.arange(b.size)[b_rot]
        i_occ = tt.arange(b.size)[b_occ]

        # Determine shapes
        rows = theta.shape[0]
        cols = self.rTA1.shape[1]

        # Compute filter operator
        if self.filter:
            F = self.F(u, f)

        # Rotation operator
        if self.filter:
            rTA1 = ts.dot(tt.dot(self.rT, F), self.A1)
        else:
            rTA1 = self.rTA1
        X_rot = tt.set_subtensor(
            tt.zeros((rows, cols))[i_rot], 
            self.dotR(rTA1, inc, obl, theta[i_rot])
        )

        # Occultation + rotation operator
        sT = self.sT(b[i_occ], ro)
        sTA = ts.dot(sT, self.A)
        theta_z = tt.arctan2(xo[i_occ], yo[i_occ])
        sTAR = self.dotRz(sTA, theta_z)
        if self.filter:
            A1InvFA1 = ts.dot(ts.dot(self.A1Inv, F), self.A1)
            sTAR = tt.dot(sTAR, A1InvFA1) 
        X_occ = tt.set_subtensor(
            tt.zeros((rows, cols))[i_occ], 
            self.dotR(sTAR, inc, obl, theta[i_occ])
        )

        return X_rot + X_occ

    @autocompile(
        "flux", tt.dvector(), tt.dvector(), tt.dvector(), tt.dvector(), 
                tt.dscalar(), tt.dscalar(), tt.dscalar(), MapVector(), 
                tt.dvector(), tt.dvector()
    )
    def flux(self, theta, xo, yo, zo, ro, inc, obl, y, u, f):
        """

        """
        return tt.dot(
            self.X(theta, xo, yo, zo, ro, inc, obl, u, f, no_compile=True), y
        )

    @autocompile(
        "intensity", tt.dvector(), tt.dvector(), tt.dvector(), MapVector(), 
                    tt.dvector(), tt.dvector()
    )
    def intensity(self, xpt, ypt, zpt, y, u, f):
        """

        """
        # Compute the polynomial basis at the point
        pT = self.pT(xpt, ypt, zpt)

        # Transform the map to the polynomial basis
        A1y = ts.dot(self.A1, y)

        # Apply the filter
        if self.filter:
            A1y = tt.dot(self.F(u, f), A1y)

        # Dot the polynomial into the basis
        return tt.dot(pT, A1y)
    
    @autocompile(
        "render", tt.iscalar(), tt.iscalar(), tt.dvector(), tt.dscalar(), 
                tt.dscalar(), MapVector(), tt.dvector(), tt.dvector()
    )
    def render(self, res, projection, theta, inc, obl, y, u, f):
        """

        """
        # Compute the Cartesian grid
        xyz = ifelse(
            tt.eq(projection, STARRY_RECTANGULAR_PROJECTION),
            self.compute_rect_grid(res),
            self.compute_ortho_grid(res)
        )

        # Compute the polynomial basis
        pT = self.pT(xyz[0], xyz[1], xyz[2])

        # If lat/lon, rotate the map so that north points up
        y = ifelse(
            tt.eq(projection, STARRY_RECTANGULAR_PROJECTION),
            self.align(y, self.get_axis(inc, obl, no_compile=True), 
                    to_tensor([0, 1, 0]), no_compile=True),
            y
        )

        # Rotate the map and transform into the polynomial basis
        if self.nw is None:
            yT = tt.tile(y, [theta.shape[0], 1])
            Ry = tt.transpose(self.dotR(yT, inc, obl, -theta))
        else:
            Ry = tt.transpose(self.dotR(tt.transpose(y), inc, obl, 
                            -tt.tile(theta[0], self.nw)))
        A1Ry = ts.dot(self.A1, Ry)

        # Apply the filter *only if orthographic*
        if self.filter:
            f0 = tt.zeros_like(f)
            f0 = tt.set_subtensor(f0[0], np.pi)
            A1Ry = ifelse(
                tt.eq(projection, STARRY_ORTHOGRAPHIC_PROJECTION),
                tt.dot(self.F(u, f), A1Ry),
                tt.dot(self.F(u, f0), A1Ry),
            )

        # Dot the polynomial into the basis
        res = tt.reshape(tt.dot(pT, A1Ry), [res, res, -1])

        # We need the shape to be (nframes, npix, npix)
        return res.dimshuffle(2, 0, 1)
    
    @autocompile(
        "get_inc_obl", tt.dvector()
    )
    def get_inc_obl(self, axis):
        """

        """
        axis /= axis.norm(2)
        inc_obl = tt.zeros(2)
        obl = tt.arctan2(axis[0], axis[1])
        sino = tt.sin(obl)
        coso = tt.cos(obl)
        inc = tt.switch(
            tt.lt(tt.abs_(sino), 1e-10),
            tt.arctan2(axis[1] / coso, axis[2]),
            tt.arctan2(axis[0] / sino, axis[2])
        )
        inc_obl = tt.set_subtensor(inc_obl[0], inc)
        inc_obl = tt.set_subtensor(inc_obl[1], obl)
        return inc_obl
    
    @autocompile(
        "get_axis", tt.dscalar(), tt.dscalar()
    )
    def get_axis(self, inc, obl):
        """

        """
        axis = tt.zeros(3)
        sino = tt.sin(obl)
        coso = tt.cos(obl)
        sini = tt.sin(inc)
        cosi = tt.cos(inc)
        axis = tt.set_subtensor(axis[0], sino * sini)
        axis = tt.set_subtensor(axis[1], coso * sini)
        axis = tt.set_subtensor(axis[2], cosi)
        return axis
    
    def set_map_vector(self, vector, inds, vals):
        """

        """
        res = tt.set_subtensor(vector[inds], vals * tt.ones_like(vector[inds]))
        return res

    @autocompile(
        "latlon_to_xyz", tt.dvector(), tt.dvector(), tt.dvector()
    )
    def latlon_to_xyz(self, axis, lat, lon):
        """

        """
        # Get the `lat = 0, lon = 0` point
        u = [axis[1], -axis[0], 0]
        theta = tt.arccos(axis[2])
        R0 = RAxisAngle(u, theta)
        origin = tt.dot(R0, [0.0, 0.0, 1.0])

        # Now rotate it to `lat, lon`
        R1 = VectorRAxisAngle([1.0, 0.0, 0.0], -lat)
        R2 = VectorRAxisAngle([0.0, 1.0, 0.0], lon)
        R = tt.batched_dot(R2, R1)
        xyz = tt.transpose(tt.dot(R, origin))
        return xyz
    
    @autocompile(
        "rotate", tt.dvector(), tt.dscalar(), MapVector()
    )
    def rotate(self, u, theta, y):
        """

        """
        u /= u.norm(2)
        return self.apply_rotation(u, theta, y)
    
    @autocompile(
        "align", MapVector(), tt.dvector(), tt.dvector()
    )
    def align(self, y, source, dest):
        """

        """
        source /= source.norm(2)
        dest /= dest.norm(2)
        axis = cross(source, dest)
        theta = tt.arccos(tt.dot(source, dest))
        return self.rotate(axis, theta, y, no_compile=True)

    @autocompile(
        "add_spot", MapVector(), tt.dvector(), tt.dscalar(), 
                    tt.dscalar(), tt.dscalar(), tt.dscalar(),
                    tt.dscalar()
    )
    def add_spot(self, y, amp, sigma, lat, lon, inc, obl):
        """

        """
        y_new = y + self.spotYlm(amp, sigma, lat, lon, inc, obl)
        y_new /= y_new[0]
        return y_new


class OpsRV(Ops):
    """

    """

    @autocompile(
        "compute_rv_filter", tt.dscalar(), tt.dscalar(), 
                                  tt.dscalar(), tt.dscalar()
    )
    def compute_rv_filter(self, inc, obl, veq, alpha):
        """

        """
        # Define some angular quantities
        cosi = tt.cos(inc)
        sini = tt.sin(inc)
        cosl = tt.cos(obl)
        sinl = tt.sin(obl)
        A = sini * cosl
        B = -sini * sinl
        C = cosi

        # Compute the Ylm expansion of the RV field
        return tt.reshape([
             0,
             veq * np.sqrt(3) * B * 
                (-A ** 2 * alpha - B ** 2 * alpha - 
                 C ** 2 * alpha + 5) / 15,
             0,
             veq * np.sqrt(3) * A * 
                (-A ** 2 * alpha - B ** 2 * alpha - 
                 C ** 2 * alpha + 5) / 15,
             0,
             0,
             0,
             0,
             0,
             veq * alpha * np.sqrt(70) * B * 
                (3 * A ** 2 - B ** 2) / 70,
             veq * alpha * 2 * np.sqrt(105) * C * 
                (-A ** 2 + B ** 2) / 105,
             veq * alpha * np.sqrt(42) * B * 
                (A ** 2 + B ** 2 - 4 * C ** 2) / 210,
             0,
             veq * alpha * np.sqrt(42) * A * 
                (A ** 2 + B ** 2 - 4 * C ** 2) / 210,
             veq * alpha * 4 * np.sqrt(105) * A * B * C / 105,
             veq * alpha * np.sqrt(70) * A * 
                (A ** 2 - 3 * B ** 2) / 70], [-1]
        ) * np.pi
    

    @autocompile(
        "rv", tt.dvector(), tt.dvector(), tt.dvector(), tt.dvector(), 
              tt.dscalar(), tt.dscalar(), tt.dscalar(), MapVector(), 
              tt.dvector(), tt.dscalar(), tt.dscalar()
    )
    def rv(self, theta, xo, yo, zo, ro, inc, obl, y, u, veq, alpha):
        """

        """
        # Compute the velocity-weighted intensity
        f = self.compute_rv_filter(inc, obl, veq, alpha, no_compile=True)
        Iv = self.flux(theta, xo, yo, zo, ro, inc, obl, y, u, f, no_compile=True)

        # Compute the inverse of the intensity
        f0 = tt.zeros_like(f)
        f0 = tt.set_subtensor(f0[0], np.pi)
        I = self.flux(theta, xo, yo, zo, ro, inc, obl, y, u, f0, no_compile=True)
        invI = tt.ones((1,)) / I
        invI = tt.where(tt.isinf(invI), 0.0, invI)

        # The RV signal is just the product        
        return Iv * invI


class OpsReflected(Ops):
    """

    """

    def __init__(self, *args, **kwargs):
        """

        """
        super(OpsReflected, self).__init__(*args, **kwargs)
        self.rT = rTReflectedOp(self._c_ops.rTReflected, self._c_ops.N)

    def compute_illumination(self, xyz, source):
        """

        """
        b = -source[:, 2]
        invsr = 1.0 / tt.sqrt(source[:, 0] ** 2 + source[:, 1] ** 2)
        cosw = source[:, 1] * invsr
        sinw = -source[:, 0] * invsr
        xrot = tt.shape_padright(xyz[0]) * cosw + \
               tt.shape_padright(xyz[1]) * sinw
        yrot = -tt.shape_padright(xyz[0]) * sinw + \
                tt.shape_padright(xyz[1]) * cosw
        I = tt.sqrt(1.0 - b ** 2) * yrot - b * tt.shape_padright(xyz[2])
        I = tt.switch(
                tt.eq(tt.abs_(b), 1.0),
                tt.switch(
                    tt.eq(b, 1.0),
                    tt.zeros_like(I),           # midnight
                    tt.shape_padright(xyz[2])   # noon
                ),
                I
            )
        I = tt.switch(tt.gt(I, 0.0), I, tt.zeros_like(I))
        return I

    @autocompile(
        "intensity", tt.dvector(), tt.dvector(), tt.dvector(), MapVector(), 
                     tt.dvector(), tt.dvector(), tt.dmatrix()
    )
    def intensity(self, xpt, ypt, zpt, y, u, f, source):
        """

        """
        # Compute the polynomial basis at the point
        pT = self.pT(xpt, ypt, zpt)

        # Transform the map to the polynomial basis
        A1y = ts.dot(self.A1, y)

        # Apply the filter
        if self.filter:
            A1y = tt.dot(self.F(u, f), A1y)

        # Dot the polynomial into the basis
        intensity = tt.dot(pT, A1y)

        # Weight the intensity by the illumination
        xyz = tt.concatenate((
            tt.reshape(xpt, [1, -1]),
            tt.reshape(ypt, [1, -1]),
            tt.reshape(xpt, [1, -1])
        ))
        I = self.compute_illumination(xyz, source)
        intensity = tt.switch(
            tt.isnan(intensity),
            intensity,
            intensity * I
        )
        return intensity

    @autocompile(
        "X", tt.dvector(), tt.dvector(), tt.dvector(), tt.dvector(), 
             tt.dscalar(), tt.dscalar(), tt.dscalar(), tt.dvector(), 
             tt.dvector(), tt.dmatrix()
    )
    def X(self, theta, xo, yo, zo, ro, inc, obl, u, f, source):
        """

        """
        # Figure out if there's an occultation
        b = tt.sqrt(xo ** 2 + yo ** 2)
        occultation = (tt.lt(b, 1.0 + ro) & tt.gt(zo, 0.0) & tt.gt(ro, 0.0)).any()
                
        # Determine shapes
        rows = theta.shape[0]
        cols = self.rTA1.shape[1]

        # Compute the semi-minor axis of the terminator
        # and the reflectance integrals
        source /= tt.reshape(source.norm(2, axis=1), [-1, 1])
        bterm = -source[:, 2]
        rT = self.rT(bterm)

        # Transform to Ylms and rotate on the sky plane
        rTA1 = ts.dot(rT, self.A1)        
        norm = 1.0 / tt.sqrt(source[:, 0] ** 2 + source[:, 1] ** 2)
        cosw = tt.switch(
            tt.eq(tt.abs_(bterm), 1.0),
            tt.ones_like(norm),
            source[:, 1] * norm
        )
        sinw = tt.switch(
            tt.eq(tt.abs_(bterm), 1.0),
            tt.zeros_like(norm),
            source[:, 0] * norm
        )
        theta_z = tt.arctan2(source[:, 0], source[:, 1])
        rTA1Rz = self.dotRz(rTA1, theta_z)

        # Apply limb darkening?
        if self.filter:
            F = self.F(u, f)
            A1InvFA1 = ts.dot(ts.dot(self.A1Inv, F), self.A1)
            rTA1Rz = tt.dot(rTA1Rz, A1InvFA1)
        
        # Rotate to the correct phase
        X_rot = self.dotR(rTA1Rz, inc, obl, theta)

        # TODO: Implement occultations in reflected light
        # Throw error if there's an occultation
        X_occ = RaiseValuerErrorIfOp(
            "Occultations in reflected light not yet implemented."
            )(occultation)

        # We're done
        return X_rot + X_occ

    @autocompile(
        "flux", tt.dvector(), tt.dvector(), tt.dvector(), tt.dvector(), 
                tt.dscalar(), tt.dscalar(), tt.dscalar(), MapVector(), 
                tt.dvector(), tt.dvector(), tt.dmatrix()
    )
    def flux(self, theta, xo, yo, zo, ro, inc, obl, y, u, f, source):
        """

        """
        return tt.dot(
            self.X(theta, xo, yo, zo, ro, inc, obl, 
                   u, f, source, no_compile=True), 
            y
        )

    @autocompile(
        "render", tt.iscalar(), tt.iscalar(), tt.dvector(), 
                  tt.dscalar(), tt.dscalar(), MapVector(), 
                  tt.dvector(), tt.dvector(), tt.dmatrix()
    )
    def render(self, res, projection, theta, inc, obl, y, u, f, source):
        """

        """
        # Compute the Cartesian grid
        xyz = ifelse(
            tt.eq(projection, STARRY_RECTANGULAR_PROJECTION),
            self.compute_rect_grid(res),
            self.compute_ortho_grid(res)
        )

        # Compute the polynomial basis
        pT = self.pT(xyz[0], xyz[1], xyz[2])

        # If lat/lon, rotate the map so that north points up
        if self.nw is None:
            y = ifelse(
                tt.eq(projection, STARRY_RECTANGULAR_PROJECTION),
                tt.reshape(
                    self.dotRxy(
                        self.dotRz(
                            tt.reshape(y, [1, -1]), tt.reshape(-obl, [1])
                        ), np.pi / 2 - inc, 0
                    ), [-1]
                ),
                y
            )
        else:
            y = ifelse(
                tt.eq(projection, STARRY_RECTANGULAR_PROJECTION),
                tt.transpose(
                    self.dotRxy(
                        self.dotRz(
                            tt.transpose(y), tt.tile(-obl, [self.nw])
                        ), np.pi / 2 - inc, 0
                    ),
                ),
                y
            )

        # Rotate the source vector as well
        source /= tt.reshape(source.norm(2, axis=1), [-1, 1])
        map_axis = self.get_axis(inc, obl, no_compile=True)
        axis = cross(map_axis, [0, 1, 0])
        angle = tt.arccos(map_axis[1])
        R = RAxisAngle(axis, -angle)
        source = ifelse(
            tt.eq(projection, STARRY_RECTANGULAR_PROJECTION),
            tt.dot(source, R),
            source
        )

        # Rotate the map and transform into the polynomial basis
        if self.nw is None:
            yT = tt.tile(y, [theta.shape[0], 1])
            Ry = tt.transpose(self.dotR(yT, inc, obl, -theta))
        else:
            Ry = tt.transpose(self.dotR(tt.transpose(y), inc, obl, 
                              -tt.tile(theta[0], self.nw)))
        A1Ry = ts.dot(self.A1, Ry)

        # Apply the filter *only if orthographic*
        if self.filter:
            f0 = tt.zeros_like(f)
            f0 = tt.set_subtensor(f0[0], np.pi)
            A1Ry = ifelse(
                tt.eq(projection, STARRY_ORTHOGRAPHIC_PROJECTION),
                tt.dot(self.F(u, f), A1Ry),
                tt.dot(self.F(u, f0), A1Ry),
            )

        # Dot the polynomial into the basis
        image = tt.dot(pT, A1Ry)

        # Compute the illumination profile 
        I = self.compute_illumination(xyz, source)

        # Weight the image by the illumination
        image = tt.switch(
            tt.isnan(image),
            image,
            image * I
        )

        # Reshape and return
        return tt.reshape(image, [res, res, -1])