import numpy as np
from numpy import memmap
from pathlib import Path
import pandas as pd
from keras.utils import to_categorical

data_dir = Path('/home/abhisek/Documents/RadarBackscatterModel/data_upd/')
# file_names = (
#         'lambda01_2.18.npy',
#         'lambda01_4.2.npy',
#         'lambda01_3.5.npy',
#         'lambda01_12.6.npy',
#         'lambda01_23.98.npy',
#         'lambda01_70.npy'
#     )
#
# fbands = np.arange(1, len(file_names)+1).astype(np.float32)
#
# array_list = list()
# df_rows = list()
# # for fname, fband in zip(file_names, fbands):
# for fname in file_names:
#     array_file = (data_dir / fname).absolute()
#     map_name = (array_file.stem + '.dat')
#     map_file = array_file.parent / map_name
#     arr = np.load(str((data_dir / fname).absolute()))
#     # arr[:, 0] = np.full_like(arr[:, 0], fband).astype(np.float32)
#     mmap = memmap(map_file, dtype=arr.dtype, mode='w+', shape=arr.shape)
#     mmap[:] = arr[:]
#     mmap.flush()
#     array_list.append(mmap)
#     df_rows.append([fname, map_name, str(arr.dtype), str(arr.shape)])
#     arr = None
# merged_array = np.concatenate(array_list, axis=0)
merged_array = np.load(data_dir / 'ALL.npy')
mmap_ = memmap(
        str((data_dir/'ALL.dat').absolute()),
        dtype=merged_array.dtype,
        mode='w+',
    )
mmap_[:] = merged_array[:]
mmap_.flush()
merged_array = None
main_data = mmap_[:, 1:]
# main_data[:, 0] = np.rad2deg(main_data[:, 0])
main_data[:, 0] = main_data[:, 0] / np.deg2rad(90)
main_data[:, 1] = main_data[:, 1] / 10.0
main_data[:, 2] = main_data[:, 2] * 10
main_data[:, 3] = main_data[:, 3] / 5.0
main_data[:, 4] = (main_data[:, 4] + 125) / 100.0
main_data[:, 5] = (main_data[:, 5] + 90.0) / 100.0
cdata = mmap_[:, 0:1]
# cat_data = to_categorical(cdata).astype(mmap_.dtype)
# marray = np.concatenate((cat_data, main_data), axis=-1)
marray = np.concatenate((cdata, main_data), axis=-1)
np.save(file=str((data_dir/'merged.npy').absolute()), arr=marray)
smap = memmap(
        str((data_dir/'merged.dat').absolute()),
        dtype=marray.dtype,
        mode='w+',
        shape=marray.shape
    )
smap[:] = marray[:]
marray = None
smap.flush()
marray_in = np.concatenate((
            smap[:, :-5],
            smap[:, -2:]
        ), axis=-1)
marray_out = main_data[:, -5:-2]
np.save(file=str((data_dir/'merged_in.npy').absolute()), arr=marray_in)
np.save(file=str((data_dir/'merged_out.npy').absolute()), arr=marray_out)
inmap = memmap(
        str((data_dir/'merged_in.dat').absolute()),
        dtype=marray_in.dtype,
        mode='w+',
        shape=marray_in.shape
    )
inmap[:] = marray_in[:]
inmap.flush()
marray_in = None
outmap = memmap(
        str((data_dir/'merged_out.dat').absolute()),
        dtype=marray_out.dtype,
        mode='w+',
        shape=marray_out.shape
    )
outmap[:] = marray_out[:]
outmap.flush()
marray_out = None
df_rows.append([
        'merged.npy',
        'merged.dat',
        str(smap.dtype),
        str(smap.shape)
    ])
df_rows.append([
        'merged_in.npy',
        'merged_in.dat',
        str(inmap.dtype),
        str(inmap.shape)
    ])
df_rows.append([
        'merged_out.npy',
        'merged_out.dat',
        str(outmap.dtype),
        str(outmap.shape)
    ])
df = pd.DataFrame(
        df_rows[:], columns=(
            'Array_Dump', 'Memory_Map', 'DType', 'Array_Shape')
    )
df.to_csv(str((data_dir/'conf.csv').absolute()))
