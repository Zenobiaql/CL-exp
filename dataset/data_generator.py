import numpy as np
import os

def main():
    os.makedirs('/datahdd_8T/zql/CL-Adapter/index_dataset', exist_ok=True)
    os.makedirs('/datahdd_8T/zql/CL-Adapter/index_dataset/task', exist_ok=True)
    array = [i for i in range(100)]
    dataset = np.array(array)
    np.save('/datahdd_8T/zql/CL-Adapter/index_dataset/task/action.npy', dataset)
    np.save('/datahdd_8T/zql/CL-Adapter/index_dataset/task/frames.npy', dataset)
    with open('/datahdd_8T/zql/CL-Adapter/index_dataset/task/instruction.txt', 'w') as f:
        f.write('This is a instruction file.')


if __name__ == '__main__':
    main()