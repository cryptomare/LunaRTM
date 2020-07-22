import numpy as np
from numpy import memmap
from pathlib import Path
import pandas as pd
from keras.utils import to_categorical

data_dir = Path('RadarBackscatterModel/data_upd/')
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
main_data[:, 0] = main_data[:, 0] / np.deg2rad(90)
main_data[:, 1] = main_data[:, 1] / 10.0
main_data[:, 2] = main_data[:, 2] * 10
main_data[:, 3] = main_data[:, 3] / 5.0
main_data[:, 4] = (main_data[:, 4] + 125) / 100.0
main_data[:, 5] = (main_data[:, 5] + 90.0) / 100.0
cdata = mmap_[:, 0:1]
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
marray_in = np.concatenate(
    (smap[:, :-5], smap[:, -2:]), axis=-1
)
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
