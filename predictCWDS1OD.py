#======================================================================================================================================
#======================================================================================================================================
#======================================================================================================================================
# 2023Sep10 mcvogt 
# USE CASE
# the goal was to develop a script that could be executed at the command line, accept an input [image] file, and pass
# it to a local Azure Custom Vision model [Object Detector] exported as an ONNX model

# HISTORY/REVISIONS
#======================================================================================
# 2023Nov05 mcvogt 
# updates to code to rename .ONNX model folder from long cryptic auto-generated name to something
# meaningful to humans, update folder name, then update .py code to include that new folder name in example calls to local models.
# (venv-python3113-datascience) C:\Development\GitHub\AIVisionExercises>python predictONNXS1OD.py CWDS1OD.ONNX/model.onnx TestImageHealthyDeerDayBuck.jpg

# 2023Oct17 mcvogt
# follow up and documenting.  migrating now to a VM hosted by VirtualBox under Wind11Pro
# Win11Pro(host)\VirtualBox7.0\Ubuntu20.04LTS(guest)\Python3.11.03\this Script

# 2023Oct09 mcvogt
# later improved code to format Category/Class Label in human-friendly form... 
# (venv-python3113-datascience) C:\Development\GitHub\AIVisionExercises>python predictONNXS1OD.py 41c20ea0f5414a3381bfab173c1b97a2.ONNX/model.onnx TestImageHealthyDeerDayBuck.jpg
# Label: Healthy Deer, Probability: 0.99009, box: (0.29941, 0.50782) (0.61335, 0.81000)
# Label: UnHealthy Deer, Probability: 0.02532, box: (0.34874, 0.00170) (0.58188, 0.55292)
# Label: Healthy Deer, Probability: 0.01254, box: (0.00925, 0.15533) (0.87207, 0.98420)

# (venv-python3113-datascience) C:\Development\GitHub\AIVisionExercises>python predictONNXS1OD.py 41c20ea0f5414a3381bfab173c1b97a2.ONNX\model.onnx TestImageUnHealthyDeerDayBuck.jpg
# Label: UnHealthy Deer, Probability: 0.13009, box: (0.07849, 0.59303) (0.71738, 0.99858)
# Label: Healthy Deer, Probability: 0.03130, box: (0.04158, 0.04641) (0.94205, 0.97782)
# Label: UnHealthy Deer, Probability: 0.01827, box: (0.55772, 0.04974) (0.78629, 0.40335)
# Label: UnHealthy Deer, Probability: 0.01671, box: (0.22403, 0.21533) (0.77413, 0.71315)

# (venv-python3113-datascience) C:\Development\GitHub\AIVisionExercises>
#
# 2023Oct02 mcvogt
# first an example of UnHealthy animal... 
# WORKING!!!!!! ========================== example ====================================
# |<-venv that was activated->| |<------- current directory -------->| python  |<-predict script->| |<----relative path from script to model->|    |<--image to be processed-->|
# (venv-python3113-datascience) C:\Development\GitHub\AIVisionExercises>python predictONNXS1OD.py 41c20ea0f5414a3381bfab173c1b97a2.ONNX/model.onnx TestImageUnHealthyDeerDayBuck.jpg
# Label: 1, Probability: 0.13009, box: (0.07849, 0.59303) (0.71738, 0.99858)
# Label: 0, Probability: 0.03130, box: (0.04158, 0.04641) (0.94205, 0.97782)
# Label: 1, Probability: 0.01827, box: (0.55772, 0.04974) (0.78629, 0.40335)
# Label: 1, Probability: 0.01671, box: (0.22403, 0.21533) (0.77413, 0.71315)

# (venv-python3113-datascience) C:\Development\GitHub\AIVisionExercises>

# second, an example of a Healthy animal...  
# (venv-python3113-datascience) C:\Development\GitHub\AIVisionExercises>python predictONNXS1OD.py 41c20ea0f5414a3381bfab173c1b97a2.ONNX/model.onnx TestImageHealthyDeerDayBuck.jpg
# Label: 0, Probability: 0.99009, box: (0.29941, 0.50782) (0.61335, 0.81000)
# Label: 1, Probability: 0.02532, box: (0.34874, 0.00170) (0.58188, 0.55292)
# Label: 0, Probability: 0.01254, box: (0.00925, 0.15533) (0.87207, 0.98420)

# (venv-python3113-datascience) C:\Development\GitHub\AIVisionExercises>

# labels to be read in from labels.txt
# 1 deer-day-healthy                    # index 0
# 2 deer-day-unhealthy                  # index 1

# 2023Sep11 mike evaluating approaches
# from https://github.com/Azure-Samples/customvision-export-samples/blob/main/samples/python/onnx/object_detection_s1/predict.py

# How to use...  python predict.py <model_filepath> <image_filepath>   <----   NO actual examples...   mike is frustrated.  
#  "C:\Development\GitHub\AIVisionExercises\41c20ea0f5414a3381bfab173c1b97a2.ONNX\model.onnx"
# python predictONNXS1OD.py 41c20ea0f5414a3381bfab173c1b97a2.ONNX/model.onnx TestImageHealthyDeerDayBuck.jpg
#========================================= example ====================================
#======================================================================================

# IMPORTS
import argparse
import pathlib
import numpy as np
import onnx
import onnxruntime
import PIL.Image

PROB_THRESHOLD = 0.01  # Minimum probably to show results.


class Model:
    def __init__(self, model_filepath):
        self.session = onnxruntime.InferenceSession(str(model_filepath))
        assert len(self.session.get_inputs()) == 1
        self.input_shape = self.session.get_inputs()[0].shape[2:]
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}[self.session.get_inputs()[0].type]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.is_bgr = False
        self.is_range255 = False
        onnx_model = onnx.load(model_filepath)
        for metadata in onnx_model.metadata_props:
            if metadata.key == 'Image.BitmapPixelFormat' and metadata.value == 'Bgr8':
                self.is_bgr = True
            elif metadata.key == 'Image.NominalPixelRange' and metadata.value == 'NominalRange_0_255':
                self.is_range255 = True

    def predict(self, image_filepath):
        image = PIL.Image.open(image_filepath).resize(self.input_shape)
        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array.transpose((0, 3, 1, 2))  # => (N, C, H, W)
        if self.is_bgr:
            input_array = input_array[:, (2, 1, 0), :, :]
        if not self.is_range255:
            input_array = input_array / 255  # => Pixel values should be in range [0, 1]

        outputs = self.session.run(self.output_names, {self.input_name: input_array.astype(self.input_type)})
        return {name: outputs[i] for i, name in enumerate(self.output_names)}

# https://www.freecodecamp.org/news/python-switch-statement-switch-case-example/
def switch(class_id):
    if class_id == 0:
        return "Healthy Deer"
    elif class_id == 1:
        return "UnHealthy Deer"

def print_outputs(outputs):
    assert set(outputs.keys()) == set(['detected_boxes', 'detected_classes', 'detected_scores'])
    for box, class_id, score in zip(outputs['detected_boxes'][0], outputs['detected_classes'][0], outputs['detected_scores'][0]):
        if score > PROB_THRESHOLD:
            clabel = switch(class_id)
#           print(f"Label: {class_id}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")
            print(f"Label: {clabel}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_filepath', type=pathlib.Path)
    parser.add_argument('image_filepath', type=pathlib.Path)

    args = parser.parse_args()

    model = Model(args.model_filepath)
    outputs = model.predict(args.image_filepath)
    print_outputs(outputs)


if __name__ == '__main__':
    main()
    
#======================================================================================================================================
#======================================================================================================================================
#======================================================================================================================================
    
