import argparse
import os
import sys

# Create the parser
diagnostics = argparse.ArgumentParser(description='Generate diagnostics and submissions files for Numerai')

# Add the arguments
diagnostics.add_argument('model',
                       metavar='model',
                       type=str,
                       help='path to the model')



# Execute the parse_args() method
args = diagnostics.parse_args()

model_path = args.model

if not os.path.isdir(input_path):
    print('The path specified does not exist')
    sys.exit()

print('\n'.join(os.listdir(input_path)))

TARGET_NAME = f"target"
PREDICTION_NAME = f"prediction"

MODEL_FILE = Path(model_path)