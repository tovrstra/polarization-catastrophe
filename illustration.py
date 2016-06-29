#!/usr/bin/env python
"""Illustrates how the polarization catastrophe can be avoided systematically."""

# The usual stuff, you know where to get it.
import numpy as np
import matplotlib.pyplot as pt
from scipy.special import erf
import nose  # nosetests

# romin can be found here: https://github.com/QuantumElephant/romin
# It is only used for its derivative checker.
from romin import deriv_check


class PointInteractions(object):
    """Interactions between point charges and point dipoles.

    Notes
    -----

    The parameter delta is always to be computed as follows:
    coordinates[iatom0] - coordinates[iatom1]

    For point interactions, the radius1 and radius2 arguments are ignored.
    """

    def pot0(self, d, radius1, radius2):
        """Interaction between two point charges as function of distance."""
        return 1/d

    def pot1(self, d, radius1, radius2):
        """Derivative of pot0 towards d."""
        return -1/d**2

    def pot2(self, d, radius1, radius2):
        """Derivative of pot1 towards d."""
        return 2/d**3

    def energy_ss(self, delta, radius1, radius2):
        """Interaction between two charges as function of relative vector."""
        d = np.linalg.norm(delta)
        return self.pot0(d, radius1, radius2)

    def energy_ps(self, delta, radius1, radius2):
        """Interaction between dipole and charge as function of relative vector."""
        d = np.linalg.norm(delta)
        return -self.pot1(d, radius1, radius2)*delta/d

    def energy_sp(self, delta, radius1, radius2):
        """Interaction between charge and dipole as function of relative vector."""
        return -self.energy_ps(delta, radius1, radius2)

    def energy_pp(self, delta, radius1, radius2):
        """Interaction between two dipoles as function of relative vector."""
        d = np.linalg.norm(delta)
        p1 = self.pot1(d, radius1, radius2)
        p2 = self.pot2(d, radius1, radius2)
        return (-p1/d)*np.identity(3) + (-p2/d**2 + p1/d**3)*np.outer(delta, delta)


class GaussInteractions(PointInteractions):
    def pot0(self, d, radius1, radius2):
        """Interaction between two gaussian charges as function of distance."""
        beta = 1/np.sqrt(2*radius1**2 + 2*radius2**2)
        return erf(beta*d)/d

    def pot1(self, d, radius1, radius2):
        """Derivative of pot0 towards d."""
        p0 = self.pot0(d, radius1, radius2)
        beta = 1/np.sqrt(2*radius1**2 + 2*radius2**2)
        return (-p0 + 2*beta/np.sqrt(np.pi)*np.exp(-(beta*d)**2))/d

    def pot2(self, d, radius1, radius2):
        """Derivative of pot1 towards d."""
        beta = 1/np.sqrt(2*radius1**2 + 2*radius2**2)
        p1 = self.pot1(d, radius1, radius2)
        return -2*p1/d - 4*beta**3/np.sqrt(np.pi)*np.exp(-(beta*d)**2)

    def energy_self2_s(self, radius1):
        """Twice the self-interaction energy of an s-type function."""
        return 1.0/(np.sqrt(np.pi)*radius1)

    def energy_self2_p(self, radius1):
        """Twice the self-interaction energy of a p-type function."""
        return 1.0/(6*np.sqrt(np.pi)*radius1**3)

    def alpha_to_radius(self, alpha1):
        """Convert an atomic polarizability to a radius for a p-type function (lower bound)."""
        return (alpha1/(6*np.sqrt(np.pi)))**(1.0/3.0)


def check_interactions(i):
    """Auxiliary function to test the analytic derivatives of any Interactions object.

    Parameters
    ----------
    i : PointInteractions
        An instance of PointInteractions or a subclass thereof.

    Notes
    -----
    This is a very unforgiving test that relies on deriv_check, which uses Gauss-Legendre
    integration to validate the implementation of partial derivatives.
    """
    from functools import partial
    for irep in xrange(10):
        radius1 = np.random.uniform(1, 2)
        radius2 = np.random.uniform(1, 2)
        ds = list(np.random.uniform(1, 3, 50))
        p0 = partial(i.pot0, radius1=radius1, radius2=radius2)
        p1 = partial(i.pot1, radius1=radius1, radius2=radius2)
        p2 = partial(i.pot2, radius1=radius1, radius2=radius2)
        deriv_check(p0, p1, ds)
        deriv_check(p1, p2, ds)

        deltas = list(np.random.uniform(-2, 2, (50, 3)))
        ss = partial(i.energy_ss, radius1=radius1, radius2=radius2)
        ps = partial(i.energy_ps, radius1=radius1, radius2=radius2)
        sp = partial(i.energy_sp, radius1=radius1, radius2=radius2)
        pp = partial(i.energy_pp, radius1=radius1, radius2=radius2)
        deriv_check(ss, sp, deltas)
        deriv_check(ps, pp, deltas)


def test_point():
    check_interactions(PointInteractions())


def test_gauss():
    check_interactions(GaussInteractions())


def test_gauss_self():
    i = GaussInteractions()
    for radius1 in 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0:
        for irep in xrange(10):
            delta = radius1*1e-3*np.random.normal(0, 1, 3)
            np.testing.assert_almost_equal(
                i.energy_self2_s(radius1),
                i.energy_ss(delta, radius1, radius1),
                decimal=3,
            )
            tens_pp = i.energy_pp(delta, radius1, radius1)
            np.testing.assert_almost_equal(i.energy_self2_p(radius1), tens_pp[0, 0], decimal=3)
            np.testing.assert_almost_equal(i.energy_self2_p(radius1), tens_pp[1, 1], decimal=3)
            np.testing.assert_almost_equal(i.energy_self2_p(radius1), tens_pp[2, 2], decimal=3)
            np.testing.assert_almost_equal(0.0, tens_pp[0, 1], decimal=3)
            np.testing.assert_almost_equal(0.0, tens_pp[0, 2], decimal=3)
            np.testing.assert_almost_equal(0.0, tens_pp[1, 2], decimal=3)
            np.testing.assert_almost_equal(radius1, i.alpha_to_radius(1/i.energy_self2_p(radius1)))


def plot_gauss_ss():
    """Plot the interaction between two identical s-type Gaussians as function of distance.

    The s-type functions have a unit charge.
    """
    i = GaussInteractions()
    ds = np.linspace(0.001, 5.0, 250)
    radius1 = 1.2
    e1s = np.array([i.pot0(d, radius1, radius1) for d in ds])
    e2s = np.array([i.energy_ss(np.array([d, 0.0, 0.0]), radius1, radius1) for d in ds])
    pt.clf()
    pt.title('Interaction between two Gaussian s-type functions')
    pt.plot(ds, e1s, label='Scalar arg', color='r', lw=6)
    pt.plot(ds, e2s, label='Vector arg', color='b', lw=2)
    pt.axhline(i.energy_self2_s(radius1), color='k', lw=2, label='2 x E_self')
    pt.xlabel('Distance [a.u.]')
    pt.ylabel('Energy [a.u.]')
    pt.legend(loc=0)
    pt.savefig('plot_gauss_ss.png')


def plot_gauss_pp():
    """Plot the interaction between two identical p-type Gaussians as function of distance.

    The p-type functions are aligned with the x-axis and approach each other along the
    x-axis. The p-type functions have a unit dipole moment.
    """
    i = GaussInteractions()
    ds = np.linspace(0.001, 5.0, 250)
    radius1 = 0.8
    e1s = np.array([-i.pot2(d, radius1, radius1) for d in ds])
    e2s = np.array([i.energy_pp(np.array([d, 0.0, 0.0]), radius1, radius1)[0, 0] for d in ds])
    pt.clf()
    pt.title('Interaction between two Gaussian p-type functions')
    pt.plot(ds, e1s, label='Scalar arg', color='r', lw=6)
    pt.plot(ds, e2s, label='Vector arg', color='b', lw=2)
    pt.axhline(i.energy_self2_p(radius1), color='k', lw=2, label='2 x E_self')
    pt.xlabel('Distance [a.u.]')
    pt.ylabel('Energy [a.u.]')
    pt.legend(loc=0)
    pt.savefig('plot_gauss_pp.png')


def plot_eigenvalues_water_static():
    """Make two plots with the eigenvalues of the static PFF response of water.

    The water geometry is compressed to arbitrarily size to check that the polarization
    catastrophe is not happening.
    """
    i = GaussInteractions()

    # Definition of the water parameters:
    # - Geometry taken from S66.
    # - Polarizabilities computed by rescaling isolated-atom values with MBIS method.
    #   (Partitioning of HF/def2qzvppd density.)
    # - Radii computed by matching atomic polarizability with self-interaction.
    # Atomic units are used.
    natom = 3
    centers = np.array([  # every row is one atom
        [-1.32695823E+00, -1.05938531E-01,  1.87881523E-02],  # oxygen
        [-1.93166525E+00,  1.60017432E+00, -2.17105231E-02],  # hydrogen
        [ 4.86644281E-01,  7.95980917E-02,  9.86247880E-03],  # hydrogen
    ])
    alphas = np.array([7.59806807999, 0.832314698674, 0.832314698674])
    radii = np.array([i.alpha_to_radius(alpha) for alpha in alphas])

    make_eigenvalue_plot(i, natom, centers, alphas, radii, 'plot_eigenvaules_water_static.png')
    radii *= 1.1
    make_eigenvalue_plot(i, natom, centers, alphas, radii, 'plot_eigenvaules_water_static_safe.png')


def make_eigenvalue_plot(i, natom, centers, alphas, radii, fn_png):
    """Make plot of eigenvalues of static PFF response of a molecule with given parameters."""
    scales = 10**np.linspace(-2, 0, 50)
    evals = []
    for scale in scales:
        # Build PFF matrix
        pff_matrix33 = np.zeros((natom, 3, natom, 3), float)
        for iatom0 in xrange(natom):
            pff_matrix33[iatom0, :, iatom0, :] = np.identity(3)/alphas[iatom0]
            for iatom1 in xrange(0, iatom0):
                delta = scale*(centers[iatom0] - centers[iatom1])
                tens = i.energy_pp(delta, radii[iatom0], radii[iatom1])
                pff_matrix33[iatom0, :, iatom1, :] = tens
                pff_matrix33[iatom1, :, iatom0, :] = tens
        pff_matrix = pff_matrix33.reshape(3*natom, 3*natom)
        evals.append(np.linalg.eigvalsh(pff_matrix))
    evals = np.array(evals)

    pt.clf()
    for col in evals.T:
        pt.loglog(scales, col)
    pt.xlabel('Scaling factor geometry [1]')
    pt.ylabel('Eigenvalues PFF matrix [a.u.]')
    pt.savefig(fn_png)


def main():
    """Main program, includes unit testing."""
    print 'Running unit tests. Please wait.'
    nose.core.TestProgram(defaultTest='__main__', exit=False)
    print 'Making plots.'
    plot_gauss_ss()
    plot_gauss_pp()
    plot_eigenvalues_water_static()


if __name__ == '__main__':
    main()
