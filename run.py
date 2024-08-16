import os

n_estimator = [110, 100, 150, 200, 210]
max_depth = [20,15,25,10,5]
for n in n_estimator:
    for m in max_depth:
        # print(n, m)
        os.system(f"python basic_ml_model.py -n{n} -m{m}")
