# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
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

# INFORMATION ABOUT THE SEMPER FORMAT LABEL:
# Picture labels consist of a sequence of bytes
# ---------------------------------------------
# v61:256B | v7:at least 256B, rounded up to multiple of block size 64,
#               with max set by LNLAB in params.f
# The versions have the same contents for the first 256B, as set out below,
# and referred to as the label 'base'; beyond this, in the label 'extension',
# the structure is deliberately undefined, allowing you to make your own use
# of the additional storage.
#
# From Aug14:
#  B1-6    S e m p e r (ic chars)
#   7,8    ncol msb,lsb (BIG-ended)
#   9,10   nrow msb,lsb
#  11,12   nlay msb,lsb
#  13,14   ccol msb,lsb
#  15,16   crow msb,lsb
#  17,18   clay msb,lsb
#    19    class: 1-20
#            for image,macro,fou,spec,correln,undef,walsh,histog,plist,lut
#    20    form: 0,1,2,3,4 = byte,i*2,fp,com,i*4 from Aug08
#    21    wp: non-zero if prot
#  22-27   creation year(-1900?),month,day,hour,min,sec
# 28    v61|v7 # chars in range record | 255
#  29-55   v61: min,max values present (ic chars for decimal repn)
#           v7: min,max vals as two Fp values in B29-36 (LE ordered)
#               followed by 19 unused bytes B37-55
#    56    plist type: 1,2,3 = list,opencurve,closedcurve
#            acc to EXAMPD - code appears to use explicit numbers
# 57,58,59 ncol,nrow,nlay hsb
# 60,61,62 ccol,crow,clay hsb
#    63    RealCoords flag (non-zero -> DX,DY,DZ,X0,Y0,Z0,units held as below)
# 64    v61:0 | v7: # blocks in (this) picture label (incl extn)
#  65-67   0 (free at present)
#  68-71   DATA cmd V7   4-byte Fp values, order LITTLE-ended
#  72-75   DATA cmd V6
#  76-79   RealCoord DZ / V5  RealCoord pars as 4-byte Fp values, LITTLE-ended
#  80-83   ...       Z0 / V4
#  84-87   ...       DY / V3
#  88-91   ...       Y0 / V2
#  92-95   ...       DX / V1
#  96-99   ...       X0 / V0
# 100    # chars in title
# 101-244  title (ic chars)
# 245-248  RealCoord X unit (4 ic chars)
# 249-252  RealCoord Y unit (4 ic chars)
# 253-256  RealCoord Z unit (4 ic chars)
#
# Aug08-Aug14 labels held no RealCoord information, so:
#   B63    'free' and zero - flagging absence of RealCoord information
# 101-256  title (12 chars longer than now)
#
# Before Aug08 labels held no hsb for pic sizes, so:
#  57-99   all free/zero except for use by DATA cmd
# 101-256  title (ic chars)

try:
    from collections import OrderedDict
    ordict = True
except ImportError:  # happens with Python < 2.7
    ordict = False

from time import strftime
import numpy as np
from traits.api import Undefined
import struct
from functools import partial

from hyperspy.misc.array_tools import sarray2dict

import logging


# Plugin characteristics
# ----------------------
format_name = 'SEMPER UNF (unformatted)'
description = 'Read data from SEMPER UNF files.'
full_support = True  # Hopefully?
# Recognised file extension
file_extensions = ('unf', 'UNF')
default_extension = 0
# Writing features
writes = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]  # All up to 3D
# ----------------------


class SemperFormat(object):

    """Class for importing and exporting SEMPER `.unf`-files.

    The :class:`~.SemperFormat` class represents a SEMPER binary file format with a header, which
    holds additional information. `.unf`-files can be saved and read from files.

    Attributes
    ----------
    data : :class:`~numpy.ndarray` (N=3)
        The phase map or magnetization information in a 3D array (with one slice).
    title : string
        Title of the file (not to be confused with the filename).
    offsets : tuple (N=3) of floats
        Offset shifts (in nm) of the grid origin (does not have to start at 0) in x, y, z.
    scales : tuple (N=3) of floats
        Grid spacing (nm per pixel) in x, y, z.
    units : tuple (N=3) of strings
        Units of the grid in x, y, z.
    metadata : dictionary
        A dictionary of all flags and metadata present in the `.unf`-file.

    """

    _log = logging.getLogger(__name__)

    ICLASS_DICT = {1: 'image', 2: 'macro', 3: 'fourier', 4: 'spectrum',
                   5: 'correlation', 6: Undefined, 7: 'walsh', 8: 'position list',
                   9: 'histogram', 10: 'display look-up table'}

    ICLASS_DICT_INV = {v: k for k, v in ICLASS_DICT.iteritems()}

    IFORM_DICT = {
        0: np.byte,
        1: np.int16,
        2: np.float32,
        3: np.complex64,
        4: np.int32}

    IFORM_DICT_INV = {v: k for k, v in IFORM_DICT.iteritems()}

    HEADER_DTYPES = [('NCOL', '<i2'),
                     ('NROW', '<i2'),
                     ('NLAY', '<i2'),
                     ('ICLASS', '<i2'),
                     ('IFORM', '<i2'),
                     ('IFLAG', '<i2'),
                     ('IFORM', '<i2')]

    LABEL_DTYPES = [('SEMPER', ('<i2', 6)),   # Bytes 1-6
                    ('NCOL', ('<i2', 2)),     # Bytes 7,8
                    ('NROW', ('<i2', 2)),     # Bytes 9,10
                    ('NLAY', ('<i2', 2)),     # Bytes 11,12
                    ('ICCOLN', ('<i2', 2)),   # Bytes 13,14
                    ('ICROWN', ('<i2', 2)),   # Bytes 15,16
                    ('ICLAYN', ('<i2', 2)),   # Bytes 17,18
                    ('ICLASS', '<i2'),        # Bytes 19
                    ('IFORM', '<i2'),         # Bytes 20
                    ('IWP', '<i2'),           # Bytes 21
                    ('DATE', ('<i2', 6)),     # Bytes 22-27
                    ('NCRANG', '<i2'),        # Bytes 28
                    ('RANGE', ('<i2', 27)),   # Bytes 29-55
                    ('IPLTYP', '<i2'),        # Bytes 56
                    ('NCOLH', '<i2'),         # Bytes 57
                    ('NROWH', '<i2'),         # Bytes 58
                    ('NLAYH', '<i2'),         # Bytes 59
                    ('ICCOLNH', '<i2'),       # Bytes 60
                    ('ICROWNH', '<i2'),       # Bytes 61
                    ('ICLAYNH', '<i2'),       # Bytes 62
                    ('REALCO', '<i2'),        # Bytes 63
                    ('NBLOCK', '<i2'),        # Bytes 64
                    ('FREE', ('<i2', 3)),     # Bytes 65-67
                    ('DATAV6', ('<i2', 4)),   # Bytes 68-71
                    ('DATAV7', ('<i2', 4)),   # Bytes 72-75
                    ('DZV5', ('<i2', 4)),     # Bytes 76-79
                    ('Z0V4', ('<i2', 4)),     # Bytes 80-83
                    ('DYV3', ('<i2', 4)),     # Bytes 84-87
                    ('Y0V2', ('<i2', 4)),     # Bytes 88-91
                    ('DXV1', ('<i2', 4)),     # Bytes 92-95
                    ('X0V0', ('<i2', 4)),     # Bytes 96-99
                    ('NTITLE', '<i2'),        # Bytes 100
                    ('TITLE', ('<i2', 144)),  # Bytes 101-244
                    ('XUNIT', ('<i2', 4)),    # Bytes 245-248
                    ('YUNIT', ('<i2', 4)),    # Bytes 249-252
                    ('ZUNIT', ('<i2', 4))]    # Bytes 253-256

    def __init__(self, data, title='', offsets=(0., 0., 0.), scales=(1., 1., 1.),
                 units=(Undefined, Undefined, Undefined), metadata=None):
        self._log.debug('Calling __init__')
        if metadata is None:
            metadata = {}
        # Make sure data is 3D!
        data = data[tuple(None for _ in xrange(3 - len(data.shape)))]
        self.data = data
        self.title = title
        self.offsets = offsets
        self.scales = scales
        self.units = units
        self.metadata = metadata
        self._log.debug('Created ' + str(self))

    @classmethod
    def _read_label(cls, unf_file):
        cls._log.debug('Calling _read_label')
        unpack = partial(
            unpack_from_intbytes,
            '<f4')  # Unpacking function for 4 byte floats!
        rec_length = np.fromfile(
            unf_file,
            dtype='<i4',
            count=1)[0]  # length of label
        label = sarray2dict(
            np.fromfile(
                unf_file,
                dtype=cls.LABEL_DTYPES,
                count=1))
        label['SEMPER'] = ''.join([str(unichr(l)) for l in label['SEMPER']])
        assert label['SEMPER'] == 'Semper'
        # Process dimensions:
        for key in ['NCOL', 'NROW', 'NLAY', 'ICCOLN', 'ICROWN', 'ICLAYN']:
            value = 256**2 * \
                label.pop(key + 'H') + 256 * label[key][0] + label[key][1]
            label[key] = value
        # Process date:
        date = '{}-{}-{} {}:{}:{}'.format(label['DATE'][0] +
                                          1900, *
                                          label['DATE'][1:])
        label['DATE'] = date
        # Process range:
        if label['NCRANG'] == 255:
            range_min = unpack(label['RANGE'][:4])
            range_max = unpack(label['RANGE'][4:8])
            range_string = '{:.6g},{:.6g}'.format(range_min, range_max)
        else:
            range_string = ''.join([str(unichr(l))
                                   for l in label['RANGE'][:label['NCRANG']]])
        label['RANGE'] = range_string
        # Process real coords:
        x0 = unpack(label.pop('X0V0'))
        dx = unpack(label.pop('DXV1'))
        y0 = unpack(label.pop('Y0V2'))
        dy = unpack(label.pop('DYV3'))
        z0 = unpack(label.pop('Z0V4'))
        dz = unpack(label.pop('DZV5'))
        if label['REALCO'] == 1:
            label.update({'X0V0': x0, 'Y0V2': y0, 'Z0V4': z0,
                          'DXV1': dx, 'DYV3': dy, 'DZV5': dz})
        # Process additional commands (unused, not sure about the purpose):
        data_v6 = unpack(label['DATAV6'])
        data_v7 = unpack(label['DATAV7'])
        label['DATAV6'] = data_v6
        label['DATAV7'] = data_v7
        # Process title:
        title = ''.join([str(unichr(l))
                        for l in label['TITLE'][:label['NTITLE']]])
        label['TITLE'] = title
        # Process units:
        label['XUNIT'] = ''.join(
            [chr(l) for l in label['XUNIT']]).replace('\x00', '')
        label['YUNIT'] = ''.join(
            [chr(l) for l in label['YUNIT']]).replace('\x00', '')
        label['ZUNIT'] = ''.join(
            [chr(l) for l in label['ZUNIT']]).replace('\x00', '')
        # Sanity check:
        assert np.fromfile(unf_file, dtype='<i4', count=1)[0] == rec_length
        # Return label:
        return label

    def _get_label(self):
        self._log.debug('Calling _get_label')
        pack = partial(
            pack_to_intbytes,
            '<f4')  # Packing function for 4 byte floats!
        nlay, nrow, ncol = self.data.shape
        data, iform = self._check_format(self.data)
        title = self.title
        # Create label:
        label = np.zeros((1,), dtype=self.LABEL_DTYPES)
        # Fill label:
        label['SEMPER'] = [ord(c) for c in 'Semper']
        label['NCOLH'], remain = divmod(ncol, 256**2)
        label['NCOL'] = divmod(remain, 256)
        label['NROWH'], remain = divmod(nrow, 256**2)
        label['NROW'] = divmod(remain, 256)
        label['NLAYH'], remain = divmod(nlay, 256**2)
        label['NLAY'] = divmod(remain, 256)
        iccoln = self.metadata.get('ICCOLN', self.data.shape[2] // 2 + 1)
        label['ICCOLNH'], remain = divmod(iccoln, 256**2)
        label['ICCOLN'] = divmod(remain, 256)
        icrown = self.metadata.get('ICROWN', self.data.shape[1] // 2 + 1)
        label['ICROWNH'], remain = divmod(icrown, 256**2)
        label['ICROWN'] = divmod(remain, 256)
        iclayn = self.metadata.get('ICLAYN', self.data.shape[0] // 2 + 1)
        label['ICLAYNH'], remain = divmod(iclayn, 256**2)
        label['ICLAYN'] = divmod(remain, 256)
        label['ICLASS'] = self.metadata.get('ICLASS', 6)  # 6: Undefined!
        label['IFORM'] = iform
        label['IWP'] = self.metadata.get('IWP', 0)  # seems standard
        date = self.metadata.get('DATE', strftime('%Y-%m-%d %H:%M:%S'))
        year, time = date.split(' ')
        date_ints = map(int, year.split('-')) + map(int, time.split(':'))
        date_ints[0] -= 1900  # Modify year integer!
        label['DATE'] = date_ints
        range_string = '{:.4g},{:.4g}'.format(self.data.min(), self.data.max())
        label['NCRANG'] = len(range_string)
        label['RANGE'][:, :len(range_string)] = [ord(c) for c in range_string]
        label['IPLTYP'] = self.metadata.get('IPLTYP', 248)  # seems standard
        label['REALCO'] = 1  # Real coordinates are used!
        label['NBLOCK'] = 4  # Always 4 64b blocks!
        label['FREE'] = [0, 0, 0]
        label['DATAV6'] = pack(0.)  # Not used!
        label['DATAV7'] = pack(0.)  # Not used!
        label['DZV5'] = pack(self.scales[2])   # DZ
        label['Z0V4'] = pack(self.offsets[2])  # Z0
        label['DYV3'] = pack(self.scales[1])   # DY
        label['Y0V2'] = pack(self.offsets[1])  # Y0
        label['DXV1'] = pack(self.scales[0])   # DX
        label['X0V0'] = pack(self.offsets[0])  # X0
        label['NTITLE'] = len(title)
        label['TITLE'][:, :len(title)] = [ord(s) for s in title]
        xunit = self.units[0] if self.units[0] is not Undefined else ''
        label['XUNIT'][:, :len(xunit)] = [ord(c) for c in xunit]
        yunit = self.units[0] if self.units[0] is not Undefined else ''
        label['YUNIT'][:, :len(yunit)] = [ord(c) for c in yunit]
        zunit = self.units[0] if self.units[0] is not Undefined else ''
        label['ZUNIT'][:, :len(zunit)] = [ord(c) for c in zunit]
        return label

    @classmethod
    def _check_format(cls, data):
        cls._log.debug('Calling _check_format')
        if data.dtype.name == 'int8':
            iform = 0  # byte
        elif data.dtype.name == 'int16':
            iform = 1  # int16
        elif 'float' in data.dtype.name and data.dtype.itemsize <= 4:
            data = data.astype(np.float32)
            iform = 2  # float (4 byte or less)
        elif 'complex' in data.dtype.name and data.dtype.itemsize <= 8:
            data = data.astype(np.complex64)
            iform = 3  # complex (8 byte or less)
        elif data.dtype.name == 'int32':
            iform = 4  # int32
        else:
            supported_formats = [np.dtype(i).name for i in cls.IFORM_DICT.values()]
            msg = 'The SEMPER file format does not support {} data type. '.format(data.dtype.name)
            msg += 'Supported data types are: ' + ', '.join(supported_formats)
            raise IOError(msg)
        return data, iform

    @classmethod
    def load_from_unf(cls, filename):
        """Load a `.unf`-file into a :class:`~.SemperFormat` object.

        Parameters
        ----------
        filename : string
            The name of the unf-file from which to load the data. Standard format is '\*.unf'.

        Returns
        -------
        semper : :class:`~.SemperFormat` (N=1)
            SEMPER file format object containing the loaded information.

        """
        cls._log.debug('Calling load_from_file')
        if ordict:
            metadata = OrderedDict()
        else:
            cls._log.warning(
                'OrderedDict is not available, using a standard dictionary.\n')
            metadata = {}
        with open(filename, 'rb') as f:
            # Read header:
            rec_length = np.fromfile(
                f,
                dtype='<i4',
                count=1)[0]  # length of header
            header = np.fromfile(
                f,
                dtype=cls.HEADER_DTYPES[
                    :rec_length //
                    2],
                count=1)
            metadata.update(sarray2dict(header))
            assert np.frombuffer(f.read(4), dtype=np.int32)[0] == rec_length, \
                'Error while reading the header (length is not correct)!'
            data_format = cls.IFORM_DICT[metadata['IFORM']]
            iversn, remain = divmod(metadata['IFLAG'], 10000)
            ilabel, ntitle = divmod(remain, 1000)
            metadata.update(
                {'IVERSN': iversn, 'ILABEL': ilabel, 'NTITLE': ntitle})
            # Read title:
            title = ''
            if ntitle > 0:
                assert np.fromfile(
                    f,
                    dtype='<i4',
                    count=1)[0] == ntitle  # length of title
                title = ''.join(np.fromfile(f, dtype='c', count=ntitle))
                metadata['TITLE'] = title
                assert np.fromfile(f, dtype='<i4', count=1)[0] == ntitle
            if ilabel:
                try:
                    metadata.update(cls._read_label(f))
                except Exception as e:
                    warning = 'Could not read label, trying to proceed without it!'
                    warning += ' (Error message: {})'.format(str(e))
                    cls._log.warning(warning)
            # Read picture data:
            nlay, nrow, ncol = metadata['NLAY'], metadata[
                'NROW'], metadata['NCOL']
            data = np.empty((nlay, nrow, ncol), dtype=data_format)
            for k in range(nlay):
                for j in range(nrow):
                    rec_length = np.fromfile(f, dtype='<i4', count=1)[0]
                    count = rec_length/np.dtype(data_format).itemsize  # Not always ncol, see below
                    row = np.fromfile(f, dtype=data_format, count=count)
                    # [:ncol] is used because Semper always writes an even number of bytes which
                    # is a problem when reading in single bytes (IFORM = 0, np.byte). If ncol is
                    # odd, an empty byte (0) is added which has to be skipped
                    # during read in:
                    data[k, j, :] = row[:ncol]
                    test = np.fromfile(f, dtype='<i4', count=1)[0]
                    assert test == rec_length
        offsets = (metadata.get('X0V0', 0.),
                   metadata.get('Y0V2', 0.),
                   metadata.get('Z0V4', 0.))
        scales = (metadata.get('DXV1', 1.),
                  metadata.get('DYV3', 1.),
                  metadata.get('DZV5', 1.))
        units = (metadata.get('XUNIT', Undefined),
                 metadata.get('YUNIT', Undefined),
                 metadata.get('ZUNIT', Undefined))
        return cls(data, title, offsets, scales, units, metadata)

    def save_to_unf(self, filename='semper.unf', skip_header=False):
        """Save a :class:`~.SemperFormat` to a file.

        Parameters
        ----------
        filename : string, optional
            The name of the unf-file to which the data should be written.
        skip_header : boolean, optional
            Determines if the header, title and label should be skipped (useful for some other
            programs). Default is False.

        Returns
        -------
        None

        """
        self._log.debug('Calling to_file')
        nlay, nrow, ncol = self.data.shape
        data, iform = self._check_format(self.data)
        title = self.title if self.title is not Undefined else ''
        with open(filename, 'wb') as f:
            if not skip_header:
                # Create header:
                header = np.zeros(
                    (1,), dtype=self.HEADER_DTYPES[
                        :-1])  # IFORMAT is not used!
                # Fill header:
                header['NCOL'] = self.data.shape[2]
                header['NROW'] = self.data.shape[1]
                header['NLAY'] = self.data.shape[0]
                header['ICLASS'] = self.metadata.get(
                    'ICLASS',
                    6)  # 6: Undefined!
                header['IFORM'] = iform
                # IVERSN: 2; ILABEL:1 (True)
                header['IFLAG'] = 2 * 10000 + 1000 + len(title)
                # Write header:
                f.write(
                    struct.pack(
                        '<i4',
                        12))  # record length, 4 byte format!
                f.write(header.tobytes())
                f.write(
                    struct.pack(
                        '<i4',
                        12))  # record length, 4 byte format!
                # Write title:
                if len(title) > 0:
                    f.write(
                        struct.pack(
                            '<i4',
                            len(title)))  # record length, 4 byte format!
                    f.write(title)
                    f.write(
                        struct.pack(
                            '<i4',
                            len(title)))  # record length, 4 byte format!
                # Create label:
                label = self._get_label()
                # Write label:
                f.write(
                    struct.pack(
                        '<i4',
                        2 *
                        256))  # record length, 4 byte format!
                f.write(label.tobytes())
                f.write(
                    struct.pack(
                        '<i4',
                        2 *
                        256))  # record length, 4 byte format!
            # Write picture data:
            for k in range(nlay):
                for j in range(nrow):
                    row = self.data[k, j, :]
                    # Record length = bytes per entry * ncol:
                    record_length = np.dtype(
                        self.IFORM_DICT[iform]).itemsize * ncol
                    f.write(
                        struct.pack(
                            '<i4',
                            record_length))  # record length, 4 byte format!
                    f.write(row.tobytes())
                    # SEMPER always expects an even number of bytes per row, which is only a
                    # problem for writing single byte data (IFORM = 0, np.byte). If ncol is odd,
                    # an empty byte (0) is added:
                    if self.data.dtype == np.byte and ncol % 2 != 0:
                        np.zeros(1, dtype=np.byte).tobytes()
                    f.write(struct.pack('<i4', record_length))  # record length, 4 byte format!

    @classmethod
    def from_signal(cls, signal):
        """Import a :class:`~.SemperFormat` object from a :class:`~hyperspy.signals.Signal` object.

        Parameters
        ----------
        signal: :class:`~hyperspy.signals.Signal`
            The signal which should be imported.

        Returns
        -------
        None

        """
        data = signal.data
        assert len(
            data.shape) <= 3, 'Only up to 3-dimensional datasets can be handled!'
        scales, offsets, units = [1.] * 3, [0.] * \
            3, [Undefined] * 3  # Defaults!
        for i in range(len(data.shape)):
            scales[i] = signal.axes_manager[i].scale
            offsets[i] = signal.axes_manager[i].offset
            units[i] = signal.axes_manager[i].units
        # Make sure data is 3D!
        data = data[tuple(None for _ in xrange(3 - len(data.shape)))]
        iclass = cls.ICLASS_DICT_INV.get(
            signal.metadata.Signal.record_by,
            6)  # 6: undefined
        data, iform = cls._check_format(data)
        title = signal.metadata.General.as_dictionary().get('title', Undefined)
        if ordict:
            metadata = OrderedDict()
        else:
            cls._log.warning(
                'OrderedDict is not available, using a standard dictionary!')
            metadata = {}
        metadata.update({'DATE': strftime('%Y-%m-%d %H:%M:%S'),
                         'ICLASS': iclass,
                         'IFORM': iform,
                         'IVERSN': 2,  # Current standard
                         'ILABEL': 1,  # True
                         'IWP': 0,  # Seems standard
                         'IPLTYP': 248,  # Seems standard
                         'ICCOLN': data.shape[2] // 2 + 1,
                         'ICROWN': data.shape[1] // 2 + 1,
                         'ICLAYN': data.shape[0] // 2 + 1})
        return cls(data, title, offsets, scales, units, metadata)

    def to_signal(self):
        """Export a :class:`~.SemperFormat` object to a :class:`~hyperspy.signals.Signal` object.

        Parameters
        ----------
        None

        Returns
        -------
        signal: :class:`~hyperspy.signals.Signal`
            The exported signal.

        """
        import hyperspy.api as hp
        data = np.squeeze(self.data)  # Reduce unneeded dimensions!
        iclass = self.ICLASS_DICT.get(self.metadata.get('ICLASS'))
        if iclass == 'spectrum':
            signal = hp.signals.Spectrum(data)
        elif iclass == 'image':
            signal = hp.signals.Image(data)
        else:  # Class is not given, but can be determined by the data shape:
            if len(data.shape) == 1:
                signal = hp.signals.Spectrum(data)
            elif len(data.shape) == 2:
                signal = hp.signals.Image(data)
            else:  # 3D data!
                signal = hp.signals.Signal(data)
        for i in range(len(data.shape)):
            signal.axes_manager[i].name = {0: 'x', 1: 'y', 2: 'z'}[i]
            signal.axes_manager[i].scale = self.scales[i]
            signal.axes_manager[i].offset = self.offsets[i]
            signal.axes_manager[i].units = self.units[i]
        signal.metadata.General.title = self.title
        signal.original_metadata.add_dictionary(self.metadata)
        return signal

    def print_info(self):
        """Print important flag information of the :class:`.~SemperFormat` object.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self._log.debug('Calling print_info')
        print '\n------------------------------------------------------'
        print self.title
        print 'dimensions: x: {}, y: {}, z: {}'.format(*reversed(self.data.shape))
        print 'scaling:    x: {:.3g}, y: {:.3g}, z: {:.3g}'.format(*self.scales)
        print 'offsets:    x: {:.3g}, y: {:.3g}, z: {:.3g}'.format(*self.offsets)
        print 'units:      x: {}, y: {}, z: {}'.format(*self.units)
        print 'data range:', (self.data.min(), self.data.max()), '\n'
        print 'metadata:'
        for k, v in self.metadata.iteritems():
            print '    {}: {}'.format(k, v)
        print '------------------------------------------------------\n'


def unpack_from_intbytes(fmt, byte_list):
    """Read in a list of bytes (as int with range 0-255) and unpack them with format `fmt`."""
    return struct.unpack(fmt, ''.join(map(chr, byte_list)))[0]


def pack_to_intbytes(fmt, value):
    """Pack a `value` into a byte list using format `fmt` and represent it as int (range 0-255)."""
    return [ord(c) for c in struct.pack(fmt, value)]


def file_reader(filename, print_info=False, **kwds):
    semper = SemperFormat.load_from_unf(filename)
    if print_info:
        semper.print_info()
    return [semper.to_signal()._to_dictionary()]


def file_writer(filename, signal, **kwds):
    semper = SemperFormat.from_signal(signal)
    semper.save_to_unf(filename)
