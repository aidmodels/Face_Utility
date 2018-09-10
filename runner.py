from face_utility.bundle import Bundle

print(Bundle.__SOLVERS__)

for each in Bundle.__SOLVERS__:
    print(each)
    task_solver = each
    task_solver.start()