import numpy as np
import os

def main():
    # Load the train and test splits
    train_path = os.path.join('/datahdd_8T/zql/CL-Adapter/index_data_split', 'train', 'task')
    val_path = os.path.join('/datahdd_8T/zql/CL-Adapter/index_data_split', 'val', 'task')
    train_action = np.load(os.path.join(train_path, 'action.npy'))
    val_action = np.load(os.path.join(val_path, 'action.npy'))
    train_frames = np.load(os.path.join(train_path, 'frames.npy'))
    val_frames = np.load(os.path.join(val_path, 'frames.npy'))

    # Check if the splits are disjoint
    if np.intersect1d(train_action, val_action).size > 0:
        print("The train and test actions are not disjoint.")
    else:
        print("The train and test actions are disjoint.")
        
    if np.intersect1d(train_frames, val_frames).size > 0:
        print("The train and test frames are not disjoint.")
    else:
        print("The train and test frames are disjoint.")
        

if __name__ == '__main__':
    main()