import numpy as np
mem_len = 100
batch = np.random.choice(mem_len, 10, replace=False)
print(batch)