# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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

from unittest import mock

import numpy as np
import pytest
import dask.array as da
from scipy.ndimage import gaussian_filter, gaussian_filter1d, rotate

import hyperspy.api as hs
from hyperspy.decorators import lazifyTestClass


@lazifyTestClass(ragged=False)
class TestSignal2D:

    def setup_method(self, method):
        self.im = hs.signals.Signal2D(np.arange(0., 18).reshape((2, 3, 3)))
        self.ragged = None

    @pytest.mark.parametrize('parallel', [True, False])
    def test_constant_sigma(self, parallel):
        s = self.im
        s.map(gaussian_filter, sigma=1, parallel=parallel, ragged=self.ragged)
        np.testing.assert_allclose(s.data, np.array(
            [[[1.68829507, 2.2662213, 2.84414753],
              [3.42207377, 4., 4.57792623],
              [5.15585247, 5.7337787, 6.31170493]],

             [[10.68829507, 11.2662213, 11.84414753],
              [12.42207377, 13., 13.57792623],
              [14.15585247, 14.7337787, 15.31170493]]]))

    @pytest.mark.parametrize('parallel', [True, False])
    def test_constant_sigma_navdim0(self, parallel):
        s = self.im.inav[0]
        s.map(gaussian_filter, sigma=1, parallel=parallel, ragged=self.ragged, inplace= not self.ragged)
        np.testing.assert_allclose(s.data, np.array(
            [[1.68829507, 2.2662213, 2.84414753],
             [3.42207377, 4., 4.57792623],
             [5.15585247, 5.7337787, 6.31170493]]))

    @pytest.mark.parametrize('parallel', [True, False])
    def test_variable_sigma(self, parallel):
        s = self.im

        sigmas = np.array([0, 1])

        s.map(gaussian_filter,
              sigma=sigmas, parallel=parallel, ragged=self.ragged)
        np.testing.assert_allclose(s.data, np.array(
            [[[0.42207377, 1., 1.57792623],
              [3.42207377, 4., 4.57792623],
              [6.42207377, 7., 7.57792623]],

             [[9.42207377, 10., 10.57792623],
              [12.42207377, 13., 13.57792623],
              [15.42207377, 16., 16.57792623]]]))

    @pytest.mark.parametrize('parallel', [True, False])
    def test_variable_sigma_navdim0(self, parallel):
        s = self.im

        sigma = 1
        s.map(gaussian_filter, sigma=sigma, parallel=parallel,
              ragged=self.ragged)
        np.testing.assert_allclose(s.data, np.array(
            [[[1.68829507, 2.2662213, 2.84414753],
              [3.42207377, 4., 4.57792623],
              [5.15585247, 5.7337787, 6.31170493]],

             [[10.68829507, 11.2662213, 11.84414753],
              [12.42207377, 13., 13.57792623],
              [14.15585247, 14.7337787, 15.31170493]]]))

    @pytest.mark.parametrize('parallel', [True, False])
    def test_axes_argument(self, parallel):
        s = self.im
        s.map(rotate, angle=45, reshape=False, parallel=parallel,
              ragged=self.ragged)
        np.testing.assert_allclose(s.data, np.array(
            [[[0., 2.23223305, 0.],
              [0.46446609, 4., 7.53553391],
              [0., 5.76776695, 0.]],

             [[0., 11.23223305, 0.],
              [9.46446609, 13., 16.53553391],
              [0., 14.76776695, 0.]]]))

    @pytest.mark.parametrize('parallel', [True, False])
    def test_different_shapes(self, parallel):
        s = self.im
        angles = hs.signals.BaseSignal([0, 45])
        if s._lazy:
            s = s.map(rotate, angle=angles.T, reshape=True, inplace=False,
                  ragged=True)
        else:
            s.map(rotate, angle=angles.T, reshape=True, show_progressbar=None,
                  parallel=parallel, ragged=True)
        # the dtype
        assert s.data.dtype is np.dtype('O')
        # the special slicing
        if not s._lazy:
            assert s.inav[0].data.base is s.data[0]
        # actual values
        np.testing.assert_allclose(s.data[0],
                                   np.arange(9.).reshape((3, 3)),
                                   atol=1e-7)
        np.testing.assert_allclose(s.data[1],
                                   np.array([[0., 0., 0., 0.],
                                             [0., 10.34834957,
                                                 13.88388348, 0.],
                                             [0., 12.11611652,
                                                 15.65165043, 0.],
                                             [0., 0., 0., 0.]]))

    @pytest.mark.parametrize('ragged', [True, False])
    def test_ragged(self, ragged):
        s = self.im
        out = s.map(lambda x: x, inplace=False, ragged=ragged)
        assert out.axes_manager.navigation_shape == s.axes_manager.navigation_shape
        if ragged:
            if s._lazy:
                s.map(lambda x: x, inplace=True, ragged=ragged)
            for i in range(s.axes_manager.navigation_size):
                np.testing.assert_allclose(s.data[i], out.data[i])
        else:
            np.testing.assert_allclose(s.data, out.data)

    @pytest.mark.parametrize('ragged', [True, False])
    def test_ragged_navigation_shape(self, ragged):
        s = hs.stack([self.im]*3)
        out = s.map(lambda x: x, inplace=False, ragged=ragged)
        assert out.axes_manager.navigation_shape == s.axes_manager.navigation_shape
        assert out.data.shape[:2] == s.axes_manager.navigation_shape[::-1]


@lazifyTestClass(ragged=False)
class TestSignal1D:

    def setup_method(self, method):
        self.s = hs.signals.Signal1D(np.arange(0., 6).reshape((2, 3)))
        self.ragged = None

    @pytest.mark.parametrize('parallel', [True, False])
    def test_constant_sigma(self, parallel):
        s = self.s
        m = mock.Mock()
        s.events.data_changed.connect(m.data_changed)
        s.map(gaussian_filter1d, sigma=1, parallel=parallel,
              ragged=self.ragged)
        np.testing.assert_allclose(s.data, np.array(
            ([[0.42207377, 1., 1.57792623],
              [3.42207377, 4., 4.57792623]])))
        assert m.data_changed.called

    @pytest.mark.parametrize('parallel', [True, False])
    def test_variable_signal_parameter(self, parallel):
        s = self.s
        m = mock.Mock()
        s.events.data_changed.connect(m.data_changed)
        s.map(lambda A, B: A - B, B=s, parallel=parallel, ragged=self.ragged)
        np.testing.assert_allclose(s.data, np.zeros_like(s.data))
        assert m.data_changed.called

    @pytest.mark.parametrize('parallel', [True, False])
    def test_constant_signal_parameter(self, parallel):
        s = self.s
        m = mock.Mock()
        s.events.data_changed.connect(m.data_changed)
        s.map(lambda A, B: A - B, B=s.inav[0], parallel=parallel,
              ragged=self.ragged)
        np.testing.assert_allclose(s.data, np.array(
            ([[0., 0., 0.],
              [3., 3., 3.]])))
        assert m.data_changed.called

    @pytest.mark.parametrize('parallel', [True, False])
    def test_dtype(self, parallel):
        s = self.s
        s.map(lambda data: np.sqrt(np.complex128(data)),
              parallel=parallel, ragged=self.ragged)
        assert s.data.dtype is np.dtype('complex128')

    @pytest.mark.parametrize('ragged', [True, False])
    def test_ragged(self, ragged):
        s = self.s
        out = s.map(lambda x: x, inplace=False, ragged=ragged)
        if ragged:
            for i in range(s.axes_manager.navigation_size):
                np.testing.assert_allclose(s.data[i], out.data[i])
        else:
            np.testing.assert_allclose(s.data, out.data)


@lazifyTestClass(ragged=False)
class TestSignal0D:

    def setup_method(self, method):
        self.s = hs.signals.BaseSignal(np.arange(0., 6).reshape((2, 3)))
        self.s.axes_manager.set_signal_dimension(0)
        self.ragged = None

    @pytest.mark.parametrize('parallel', [True, False])
    def test(self, parallel):
        s = self.s
        m = mock.Mock()
        s.events.data_changed.connect(m.data_changed)
        s.map(lambda x, e: x ** e, e=2, parallel=parallel, ragged=self.ragged)
        np.testing.assert_allclose(
            s.data, (np.arange(0., 6) ** 2).reshape((2, 3,)))
        assert m.data_changed.called

    @pytest.mark.parametrize('parallel', [True, False])
    def test_nav_dim_1(self, parallel):
        s = self.s.inav[1, 1]
        m = mock.Mock()
        s.events.data_changed.connect(m.data_changed)
        s.map(lambda x, e: x ** e, e=2, parallel=parallel, ragged=self.ragged)
        np.testing.assert_allclose(s.data, self.s.inav[1, 1].data ** 2)
        #assert m.data_changed.called


_alphabet = 'abcdefghijklmnopqrstuvwxyz'


@lazifyTestClass(ragged=False)
class TestChangingAxes:

    def setup_method(self, method):
        self.base = hs.signals.BaseSignal(np.empty((2, 3, 4, 5, 6, 7)))
        self.ragged = None
        for ax, name in zip(self.base.axes_manager._axes, _alphabet):
            ax.name = name

    @pytest.mark.parametrize('parallel', [True, False])
    def test_one_nav_reducing(self, parallel):
        s = self.base.transpose(signal_axes=4).inav[0, 0]
        s.map(np.mean, axis=1, parallel=parallel, ragged=self.ragged)
        assert list('def') == [ax.name for ax in
                               s.axes_manager._axes]
        assert 0 == len(s.axes_manager.navigation_axes)
        s.map(np.mean, axis=(1, 2), parallel=parallel, ragged=self.ragged)
        assert ['f'] == [ax.name for ax in s.axes_manager._axes]
        assert 0 == len(s.axes_manager.navigation_axes)

    @pytest.mark.parametrize('parallel', [True, False])
    def test_one_nav_increasing(self, parallel):
        s = self.base.transpose(signal_axes=4).inav[0, 0]
        s.map(np.tile, reps=(2, 1, 1, 1, 1),
              parallel=parallel, ragged=self.ragged)
        assert len(s.axes_manager.signal_axes) == 5
        assert set('cdef') <= {ax.name for ax in
                               s.axes_manager._axes}
        assert 0 == len(s.axes_manager.navigation_axes)
        assert s.data.shape == (2, 4, 5, 6, 7)

    @pytest.mark.parametrize('parallel', [True, False])
    def test_reducing(self, parallel):
        s = self.base.transpose(signal_axes=4)
        s.map(np.mean, axis=1, parallel=parallel, ragged=self.ragged)
        assert list('abdef') == [ax.name for ax in
                                 s.axes_manager._axes]
        assert 2 == len(s.axes_manager.navigation_axes)
        s.map(np.mean, axis=(1, 2), parallel=parallel, ragged=self.ragged)
        assert ['f'] == [ax.name for ax in
                         s.axes_manager.signal_axes]
        assert list('ba') == [ax.name for ax in
                              s.axes_manager.navigation_axes]
        assert 2 == len(s.axes_manager.navigation_axes)

    @pytest.mark.parametrize('parallel', [True, False])
    def test_increasing(self, parallel):
        s = self.base.transpose(signal_axes=4)
        s.map(np.tile, reps=(2, 1, 1, 1, 1),
              parallel=parallel, ragged=self.ragged)
        assert len(s.axes_manager.signal_axes) == 5
        assert set('cdef') <= {ax.name for ax in
                               s.axes_manager.signal_axes}
        assert list('ba') == [ax.name for ax in
                              s.axes_manager.navigation_axes]
        assert 2 == len(s.axes_manager.navigation_axes)
        assert s.data.shape == (2, 3, 2, 4, 5, 6, 7)


@pytest.mark.parametrize('parallel', [True, False])
def test_new_axes(parallel):
    s = hs.signals.Signal1D(np.empty((10, 10)))
    s.axes_manager.navigation_axes[0].name = 'a'
    s.axes_manager.signal_axes[0].name = 'b'

    def test_func(d, i):
        i = int(i)
        _slice = () + (None,) * i + (slice(None),)
        return d[_slice]
    res = s.map(test_func, inplace=False,
                i=hs.signals.BaseSignal(np.arange(10)).T,
                parallel=parallel, ragged=True)
    assert res is not None
    sl = res.inav[:2]
    assert sl.axes_manager._axes[-1].name == 'a'
    sl = res.inav[-1]
    assert isinstance(sl, hs.signals.BaseSignal)
    ax_names = {ax.name for ax in sl.axes_manager._axes}
    assert len(ax_names) == 1
    assert not 'a' in ax_names
    assert not 'b' in ax_names
    assert 0 == sl.axes_manager.navigation_dimension


class TestLazyMap:
    def setup_method(self, method):
        dask_array = da.zeros((10, 11, 12, 13), chunks=(3, 3, 3, 3))
        self.s = hs.signals.Signal2D(dask_array).as_lazy()

    @pytest.mark.parametrize('chunks', [(3, 2), (3, 3)])
    def test_map_iter(self,chunks):
        iter_array, _ = da.meshgrid(range(11), range(10))
        iter_array = iter_array.rechunk(chunks)
        s_iter = hs.signals.BaseSignal(iter_array).T
        s_iter = s_iter.as_lazy()
        f = lambda a, b: a + b
        s_out = self.s.map(function=f, b=s_iter, inplace=False)
        np.testing.assert_array_equal(s_out.mean(axis=(2, 3)).data, iter_array)

    def test_map_nav_size_error(self):
        iter_array, _ = da.meshgrid(range(12), range(10))
        s_iter = hs.signals.BaseSignal(iter_array).T
        f = lambda a, b: a + b
        with pytest.raises(ValueError):
            self.s.map(function=f, b=s_iter, inplace=False)

    def test_map_iterate_array(self):
        s = self.s
        iter_array, _ = np.meshgrid(range(11), range(10))
        f = lambda a, b: a + b
        iterating_kwargs = {'b':iter_array.T}
        s_out = s._map_iterate(function=f, iterating_kwargs=iterating_kwargs,
                               inplace=False)
        np.testing.assert_array_equal(s_out.mean(axis=(2, 3)).data, iter_array)

    def test_keep_navigation_chunks(self):
        s = self.s
        s_out = s.map(lambda x: x, inplace=False, lazy_result=True)
        assert s._get_navigation_chunk_size() == s_out._get_navigation_chunk_size()

    def test_keep_navigation_chunks_cropping(self):
        s = self.s
        s1 = s.inav[1:-2, 2:-1]
        s_out = s1.map(lambda x: x, inplace=False, lazy_result=True)
        assert s1._get_navigation_chunk_size() == s_out._get_navigation_chunk_size()


def a_function(image, add=4):
    return image + add


class TestLazyResultInplace:
    def setup_method(self):
        data = np.zeros((32, 40, 64, 64), dtype=np.uint16)
        data[:, :, 32 - 10 : 32 + 10, 32 - 10 : 32 + 10] = 100
        s = hs.signals.Signal2D(data)
        dask_array = da.from_array(data, chunks=(32, 32, 32, 32))
        s_lazy = hs.signals.Signal2D(dask_array).as_lazy()
        self.s_signal_image = data[0, 0].copy()
        self.s = s
        self.s_lazy = s_lazy

    def test_lazy_input_not_lazy_result_not_inplace(self):
        s = self.s_lazy
        add = 1
        s_out = s.map(a_function, add=add, inplace=False, lazy_result=False)
        assert not s_out._lazy
        s.compute()
        for ix, iy in np.ndindex(s_out.axes_manager.navigation_shape):
            np.testing.assert_allclose(s_out.data[iy, ix], self.s_signal_image + add)
            np.testing.assert_allclose(s.data[iy, ix], self.s_signal_image)

    def test_not_lazy_input_not_lazy_result_not_inplace(self):
        s = self.s
        add = 1
        s_out = s.map(a_function, add=add, inplace=False, lazy_result=False)
        assert not s_out._lazy
        for ix, iy in np.ndindex(s_out.axes_manager.navigation_shape):
            np.testing.assert_allclose(s_out.data[iy, ix], self.s_signal_image + add)
            np.testing.assert_allclose(s.data[iy, ix], self.s_signal_image)

    def test_lazy_input_lazy_result_not_inplace(self):
        s = self.s_lazy
        add = 1
        s_out = s.map(a_function, add=add, inplace=False, lazy_result=True)
        assert s_out._lazy
        s_out.compute()
        s.compute()
        for ix, iy in np.ndindex(s_out.axes_manager.navigation_shape):
            np.testing.assert_allclose(s_out.data[iy, ix], self.s_signal_image + add)
            np.testing.assert_allclose(s.data[iy, ix], self.s_signal_image)

    def test_not_lazy_input_lazy_result_not_inplace(self):
        s = self.s
        add = 1
        s_out = s.map(a_function, add=add, inplace=False, lazy_result=True)
        assert s_out._lazy
        s_out.compute()
        for ix, iy in np.ndindex(s_out.axes_manager.navigation_shape):
            np.testing.assert_allclose(s_out.data[iy, ix], self.s_signal_image + add)
            np.testing.assert_allclose(s.data[iy, ix], self.s_signal_image)

    def test_lazy_input_not_lazy_result_inplace(self):
        s = self.s_lazy
        add = 1
        s.map(a_function, add=add, inplace=True, lazy_result=False)
        assert not s._lazy
        for ix, iy in np.ndindex(s.axes_manager.navigation_shape):
            np.testing.assert_allclose(s.data[iy, ix], self.s_signal_image + add)

    def test_not_lazy_input_not_lazy_result_inplace(self):
        s = self.s
        add = 1
        s.map(a_function, add=add, inplace=True, lazy_result=False)
        assert not s._lazy
        for ix, iy in np.ndindex(s.axes_manager.navigation_shape):
            np.testing.assert_allclose(s.data[iy, ix], self.s_signal_image + add)

    def test_lazy_input_lazy_result_inplace(self):
        s = self.s_lazy
        add = 1
        s.map(a_function, add=add, inplace=True, lazy_result=True)
        assert s._lazy
        s.compute()
        for ix, iy in np.ndindex(s.axes_manager.navigation_shape):
            np.testing.assert_allclose(s.data[iy, ix], self.s_signal_image + add)

    def test_not_lazy_input_lazy_result_inplace(self):
        s = self.s
        add = 1
        s.map(a_function, add=add, inplace=True, lazy_result=True)
        assert s._lazy
        s.compute()
        for ix, iy in np.ndindex(s.axes_manager.navigation_shape):
            np.testing.assert_allclose(s.data[iy, ix], self.s_signal_image + add)


class TestOutputDtype:
    @pytest.mark.parametrize('dtype', [np.uint16, np.uint32, np.uint64, np.int32, np.float32])
    def test_output_dtype_specified_not_inplace(self, dtype):
        def a_function_dtype(data):
            return data.astype("float32")
        s = hs.signals.Signal1D(np.zeros((10, 100)), dtype=np.int16)
        s_out = s.map(a_function_dtype, inplace=False, output_dtype=dtype, lazy_result=True)
        assert s_out.data.dtype == dtype
        s_out.compute()
        assert s_out.data.dtype == dtype

    @pytest.mark.parametrize('dtype', [np.uint16, np.uint32, np.uint64, np.int32, np.float32])
    def test_output_dtype_specified_inplace(self, dtype):
        def a_function_dtype(data):
            return data.astype("float32")
        s = hs.signals.Signal1D(np.zeros((10, 100)), dtype=np.int16)
        s.map(a_function_dtype, inplace=True, output_dtype=dtype, lazy_result=True)
        assert s.data.dtype == dtype
        s.compute()
        assert s.data.dtype == dtype

    @pytest.mark.parametrize('dtype', [np.uint16, np.uint32, np.uint64, np.int32, np.float32])
    def test_output_dtype_auto(self, dtype):
        def a_function_dtype(data, dtype_to_function):
            return data.astype(dtype_to_function)
        s = hs.signals.Signal1D(np.zeros((10, 100)), dtype=np.int16)
        s_out = s.map(a_function_dtype, inplace=False, dtype_to_function=dtype, lazy_result=True)
        assert s_out.data.dtype == dtype
        s_out.compute()
        assert s_out.data.dtype == dtype


class TestOutputDtype:
    @pytest.mark.parametrize('output_signal_size', [(10,), (10, 20), (10, 20, 30)])
    def test_output_signal_size(self, output_signal_size):
        def a_function_signal_size(data, output_signal_size_for_function):
            return np.zeros(output_signal_size_for_function)
        s = hs.signals.Signal1D(np.zeros((10, 100)), dtype=np.int16)
        s_out = s.map(
            a_function_signal_size,
            inplace=False,
            output_signal_size=output_signal_size,
            lazy_result=True,
            output_signal_size_for_function=output_signal_size,
        )
        assert s_out.data[0].shape == output_signal_size
        s_out.compute()
        assert s_out.data[0].shape == output_signal_size

    def test_output_signal_size_wrong_size(self):
        def a_function(data):
            return np.zeros(10)
        s = hs.signals.Signal1D(np.zeros((10, 100)), dtype=np.int16)
        s_out = s.map(
            a_function,
            inplace=False,
            output_signal_size=(11, ),
            lazy_result=True
        )
        with pytest.raises(ValueError):
            s_out.compute()


class TestOutputSignalSizeScalarWithNavigationDimensions:
    @pytest.mark.parametrize('nav_shape', ((9, ), (8, 7), (6, 5, 4)))
    def test_not_lazy_result(self, nav_shape):
        def a_function(image):
            return 10
        data_shape = nav_shape + (20, 30)
        data = np.zeros(data_shape)
        s = hs.signals.Signal2D(data)
        s_out = s.map(a_function, inplace=False, lazy_result=False)
        assert s_out.data.shape == nav_shape
        assert s_out.axes_manager.navigation_shape == nav_shape[::-1]
        assert (s_out.data == np.ones(nav_shape, dtype=np.float) * 10).all()
        assert s.data.shape == data_shape
        assert s.axes_manager.shape == nav_shape[::-1] + (30, 20)

        s.map(a_function, inplace=True, lazy_result=False)
        assert s.data.shape == nav_shape
        assert s.axes_manager.navigation_shape == nav_shape[::-1]

    @pytest.mark.parametrize('nav_shape', ((9, ), (8, 7), (6, 5, 4)))
    def test_lazy_result(self, nav_shape):
        def a_function(image):
            return 10
        data_shape = nav_shape + (20, 30)
        data = np.zeros(data_shape)
        s = hs.signals.Signal2D(data)
        s_out = s.map(a_function, inplace=False, lazy_result=True)
        assert s_out.data.shape == nav_shape
        assert s_out.axes_manager.navigation_shape == nav_shape[::-1]
        assert s.data.shape == data_shape
        assert s.axes_manager.shape == nav_shape[::-1] + (30, 20)

        s.map(a_function, inplace=True, lazy_result=True)
        assert s.data.shape == nav_shape
        assert s.axes_manager.navigation_shape == nav_shape[::-1]


class TestFunctionChangingIteratingKwargs:
    def test_not_inplace_not_lazy_result(self):
        def a_function(image, value):
            value[:] = 8
            return image + value[0] / value[1]
        s = hs.signals.Signal2D(np.zeros((2, 3, 10, 10)))
        data_iter = np.ones((2, 3, 2), dtype=np.uint16)

        s_iter = hs.signals.Signal1D(data_iter.copy())
        s_out = s.map(a_function, value=s_iter, inplace=False, lazy_result=False)
        assert (s_iter.data.dtype == data_iter.dtype)
        assert (s_iter.data == data_iter).all()
        assert (s_out.data == 1.).all()

    def test_inplace_not_lazy_result(self):
        def a_function(image, value):
            value[:] = 8
            return image + value[0] / value[1]
        s = hs.signals.Signal2D(np.zeros((2, 3, 10, 10)))
        data_iter = np.ones((2, 3, 2), dtype=np.uint16)

        s_iter = hs.signals.Signal1D(data_iter.copy())
        s.map(a_function, value=s_iter, inplace=True, lazy_result=False)
        assert (s_iter.data.dtype == data_iter.dtype)
        assert (s_iter.data == data_iter).all()
        assert (s.data == 1.).all()

    def test_inplace_lazy_result(self):
        def a_function(image, value):
            value[:] = 8
            return image + value[0] / value[1]
        s = hs.signals.Signal2D(np.zeros((2, 3, 10, 10)))
        data_iter = np.ones((2, 3, 2), dtype=np.uint16)

        s_iter = hs.signals.Signal1D(data_iter.copy())
        s.map(a_function, value=s_iter, inplace=True, lazy_result=True)
        s.compute()
        assert (s_iter.data.dtype == data_iter.dtype)
        assert (s_iter.data == data_iter).all()
        assert (s.data == 1.).all()

    def test_not_inplace_lazy_result(self):
        def a_function(image, value):
            value[:] = 8
            return image + value[0] / value[1]
        s = hs.signals.Signal2D(np.zeros((2, 3, 10, 10)))
        data_iter = np.ones((2, 3, 2), dtype=np.uint16)

        s_iter = hs.signals.Signal1D(data_iter.copy())
        s_out = s.map(a_function, value=s_iter, inplace=False, lazy_result=True)
        s_out.compute()
        assert (s_iter.data.dtype == data_iter.dtype)
        assert (s_iter.data == data_iter).all()
        assert (s_out.data == 1.).all()


def test_dask_array_store():
    def a_function(image):
        image = image * 101
        return image
    s = hs.signals.Signal2D(np.ones((10, 12, 20, 24)), dtype=np.int16)
    s.map(a_function, inplace=True, lazy_result=False)
    assert (s.data == 101).all()


class TestFunctionChangingArgs:
    def test_not_inplace_not_lazy_result(self):
        def a_function(image, animage):
            animage *= 1.5
            return image + animage
        s = hs.signals.Signal2D(np.zeros((2, 3, 10, 10)))
        animage_orig = np.arange(0, 100, dtype=np.float32).reshape(10, 10)
        animage = animage_orig.copy()
        s_out = s.map(a_function, inplace=False, animage=animage, lazy_result=False)
        assert (animage == animage_orig).all()


@pytest.mark.parametrize('ragged', [True, False, None])
def test_singleton(ragged):
    sig = hs.signals.Signal2D(np.empty((3, 2)))
    sig.axes_manager[0].name = 'x'
    sig.axes_manager[1].name = 'y'
    sig1 = sig.map(lambda x: 3, inplace=False, ragged=ragged)
    sig2 = sig.map(np.sum, inplace=False, ragged=ragged)
    sig.map(np.sum, inplace=True, ragged=ragged)
    sig_list = (sig, sig1, sig2)
    for _s in sig_list:
        assert len(_s.axes_manager._axes) == 1
        assert _s.axes_manager[0].name == 'Scalar'
        assert isinstance(_s, hs.signals.BaseSignal)
        assert not isinstance(_s, hs.signals.Signal1D)


def test_lazy_singleton():
    sig = hs.signals.Signal2D(np.empty((3, 2)))
    sig = sig.as_lazy()
    sig.axes_manager[0].name = 'x'
    sig.axes_manager[1].name = 'y'
    # One without arguments
    sig1 = sig.map(lambda x: 3, inplace=False, ragged=False)
    sig2 = sig.map(np.sum, inplace=False, ragged=False)
    # in place not supported for lazy signal and ragged
    sig.map(np.sum, ragged=False, inplace=True)
    sig_list = [sig1, sig2, sig]
    for _s in sig_list:
        assert len(_s.axes_manager._axes) == 1
        assert _s.axes_manager[0].name == 'Scalar'
        assert isinstance(_s, hs.signals.BaseSignal)
        assert not isinstance(_s, hs.signals.Signal1D)
        #assert isinstance(_s, LazySignal)


def test_lazy_singleton_ragged():
    sig = hs.signals.Signal2D(np.empty((3, 2)))
    sig = sig.as_lazy()
    sig.axes_manager[0].name = 'x'
    sig.axes_manager[1].name = 'y'
    # One without arguments
    sig1 = sig.map(lambda x: 3, inplace=False, ragged=True)
    sig2 = sig.map(np.sum, inplace=False, ragged=True)
    # in place not supported for lazy signal and ragged
    sig_list = (sig1, sig2)
    for _s in sig_list:
        assert isinstance(_s, hs.signals.BaseSignal)
        assert not isinstance(_s, hs.signals.Signal1D)
        #assert isinstance(_s, LazySignal)


def test_map_ufunc(caplog):
    data = np.arange(100, 200).reshape(10, 10)
    s = hs.signals.Signal1D(data)
    # check that it works and it raises a warning
    caplog.clear()
    # s.map(np.log)
    assert np.log(s) == s.map(np.log)
    np.testing.assert_allclose(s.data, np.log(data))
    assert "can direcly operate on hyperspy signals" in caplog.records[0].message
