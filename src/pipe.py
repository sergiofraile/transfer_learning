from subprocess import call

print('Initializasing pipe...')

call("./pre.sh", shell=True)
call(["python", "main.py"])

print('Pipe process completed')
