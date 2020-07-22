from pathlib import Path
from sklearn.utils.extmath import cartesian
import numpy as np
from numpy import memmap
from Model import Rover
from joblib import Parallel, delayed
data_dir = Path('RadarBackscatterModel/data_upd/')
main_data = np.load(data_dir / 'data.npy')

inmap = main_data[:, :4]
outmap = main_data[:, 4:-2]
structure = Rover(input_features=4, output_features=2)
structure.ready()
structure.learn(x=inmap, y=outmap, batch_size=1000, epochs=10)
structure.store(
    str((data_dir / 'Model.h5').absolute())
)
