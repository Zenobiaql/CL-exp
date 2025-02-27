from multiprocessing import Pool

def add(a, b):
    return a + b

def main():
    # 创建一个进程池
    with Pool(processes=4) as pool:
        # 参数元组列表
        args = [(1, 2), (3, 4), (5, 6), (7, 8)]
        
        # 使用 starmap 并行处理任务
        results = pool.starmap(add, args)
        
    print(results)

if __name__ == "__main__":
    main()