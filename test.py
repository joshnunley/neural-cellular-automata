import subprocess

remove_old_directory = "rm -rf train_log/*"
subprocess.run(remove_old_directory.split(), cwd="/N/u/joshnunl/BigRed200/neural-cellular-automata/replicate-lizard")
make_new_directory = "mkdir -p train_log" 
subprocess.run(make_new_directory.split(), cwd="/N/u/joshnunl/BigRed200/neural-cellular-automata/replicate-lizard")