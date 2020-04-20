import colorsys
import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
import shutil
from numba import cuda
from os import listdir


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def load_models(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        person_boxes = []
        for i, c in reversed(list(enumerate(out_classes))):
            box = out_boxes[i]
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            height = bottom-top
            width = right-left
            top = top - int(height*0.1)
            bottom = bottom + int(height*0.1)
            left = left - int(width*0.1)
            right = right + int(width*0.1)
            top = max(0, top)
            left = max(0, left)
            right = min(image.size[0],right)
            bottom = min(image.size[1],bottom)
            coordinates= [top,bottom, left, right]
            person_boxes += [coordinates]
        return image, person_boxes

    def close_session(self):
        self.sess.close()

    def detect_video(self,frame):
        image = Image.fromarray(frame)
        image, person_boxes = self.detect_image(image)
        return person_boxes


class GenerateFaceCrops:
    def __init__(self,video_directory, sample_rate):
        self.video_directory = video_directory
        self.sample_rate = sample_rate
        self.skip_rate = 300//sample_rate

    def face_crops(self):
        os.mkdir("FaceCrops")
        from os.path import isfile, join
        videos = [f for f in listdir(self.video_directory) if isfile(join(self.video_directory, f))]
        video_captures = []
        first_frame_faces = []
        first_frame_vids = []

        yolo_object = YOLO()
        yolo_object.load_models()
        for v in videos:
            video_path = self.video_directory+"/"+v
            video_name = video_path.split("/")[-1].split(".")[0]
            os.mkdir("FaceCrops/" + video_name)
            vid = cv2.VideoCapture(video_path)
            ret, frame = vid.read()
            first_frame_vids.append(frame)
            person_boxes = yolo_object.detect_video(frame)
            video_captures.append(vid)
            first_frame_faces.append(person_boxes)
            for i in range(0, len(person_boxes)):
                os.mkdir("FaceCrops/" + video_name + "/" + "Person" + str(i + 1))
            person_no = 1
            for bbox in person_boxes:
                x, y, w, h = int(bbox[2]), int(bbox[0]), int(bbox[3] - bbox[2]), int(bbox[1] - bbox[0])
                img = frame[y:y + h, x:x + w]
                cv2.imwrite("FaceCrops/" + video_name + "/" + "Person" + str(person_no) + "/" + "Frame0.jpg", img)
                person_no += 1
        cuda.close()
        for index, video_name in enumerate(videos):
            print("Video Number is :- ", index+1)
            frame = first_frame_vids.pop(0)
            vid = video_captures.pop(0)
            current_boxes = first_frame_faces.pop(0)

            video_name = video_name.split(".")[0]
            multiTracker = cv2.MultiTracker_create()

            for i in range(0, len(current_boxes)):
                bbox = current_boxes[i]
                x, y, w, h = int(bbox[2]), int(bbox[0]), int(bbox[3] - bbox[2]), int(bbox[1] - bbox[0])
                bbox = [x, y, w, h]
                multiTracker.add(cv2.TrackerKCF_create(), frame, tuple(bbox))
            frame_number = 1
            total_frames = 1
            while vid.isOpened():
                ret, frame = vid.read()
                if not ret:
                    break
                ret, boxes = multiTracker.update(frame)
                if total_frames==self.sample_rate:
                    break
                if frame_number%self.skip_rate!=0:
                    frame_number +=1
                    continue
                total_frames+=1
                person_no = 1
                for i, newbox in enumerate(boxes):
                    x, y, w, h = newbox
                    img = frame[int(y):int(y + h), int(x):int(x + w)]
                    cv2.imwrite("FaceCrops/" + video_name + "/" + "Person" + str(person_no) + "/" + "Frame" + str(frame_number) + ".jpg", img)
                    person_no += 1
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                frame_number += 1
                print("Current Video :-", index+1, "Frame No.:- ", frame_number)


class RunXceptionNet:
    pass

class RunResNext:
    pass


if __name__ == "__main__":
    frames_per_video = 30

    video_directory = "/media/kenil/Kenil/Users/Kenil/Downloads/Final_Project/check"
    face_crop_object = GenerateFaceCrops(video_directory, frames_per_video)
    face_crop_object.face_crops()
    # shutil.rmtree("/media/kenil/Kenil/Users/Kenil/Downloads/Final_Project/Main/FaceCrops")