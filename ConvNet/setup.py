from cx_Freeze import setup, Executable

base = None    

executables = [Executable("TestDigit.py", base=base)]

packages = ["idna","numpy"]
options = {
    'build_exe': {    
        'packages':packages,
    },    
}

setup(
    name = "Digits Prediction",
    options = options,
    version = "1.0",
    description = 'Test trained convolutional neural network in digit recognition',
    executables = executables
)
