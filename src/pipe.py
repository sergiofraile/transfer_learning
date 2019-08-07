from subprocess import call

print('Initializasing pipe...')

call("src/pre.sh", shell=True)
call(["python", "src/main.py"])

print('Pipe process completed')
