import colorsys
import cv2
import numpy as np
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model,Model, Sequential
from keras.layers import Input, Dropout, Flatten, Dense,Conv2D, MaxPool2D, GlobalMaxPooling2D,Activation,GlobalAveragePooling2D
from keras.layers import BatchNormalization,multiply,Lambda
from keras import optimizers
from PIL import Image
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
import time
from numba import cuda
from os import listdir
from os.path import isfile, join
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Normalize
from sklearn import metrics
from efficientnet.keras import EfficientNetB5, EfficientNetB6
from keras.applications.xception import Xception, preprocess_input
import json

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MyResNeXt(models.resnet.ResNet):
    def __init__(self, training=True):
        super(MyResNeXt, self).__init__(block=models.resnet.Bottleneck,
                                        layers=[3, 4, 6, 3],
                                        groups=32,
                                        width_per_group=4)
        self.fc = nn.Linear(2048, 1)


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
    def __init__(self, video_directory, sample_rate):
        self.video_directory = video_directory
        self.sample_rate = sample_rate

    def face_crops(self):
        try:
            os.mkdir("FaceCrops")
        except Exception as e:
            print(e)
        videos = [f for f in listdir(self.video_directory) if isfile(join(self.video_directory, f))]
        yolo_object = YOLO()
        yolo_object.load_models()
        for v in videos:
            try:
                video_path = self.video_directory+"/"+v
                video_name = video_path.split("/")[-1].split(".")[0]
                os.mkdir("FaceCrops/" + video_name)
                print(video_name)
                vid = cv2.VideoCapture(video_path)
                ret, frame = vid.read()
                person_boxes = yolo_object.detect_video(frame)
                for i in range(0, len(person_boxes)):
                    os.mkdir("FaceCrops/" + video_name + "/" + "Person" + str(i + 1))
                person_no = 1
                for bbox in person_boxes:
                    x, y, w, h = int(bbox[2]), int(bbox[0]), int(bbox[3] - bbox[2]), int(bbox[1] - bbox[0])
                    img = frame[y:y + h, x:x + w]
                    cv2.imwrite("FaceCrops/" + video_name + "/" + "Person" + str(person_no) + "/" + "Frame0.jpg", img)
                    person_no += 1
                multiTracker = cv2.MultiTracker_create()

                for i in range(0, len(person_boxes)):
                    bbox = person_boxes[i]
                    x, y, w, h = int(bbox[2]), int(bbox[0]), int(bbox[3] - bbox[2]), int(bbox[1] - bbox[0])
                    bbox = [x, y, w, h]
                    multiTracker.add(cv2.TrackerKCF_create(), frame, tuple(bbox))
                frame_number = 1
                total_frames = 1
                while vid.isOpened():
                    skip_rate = vid.get(cv2.CAP_PROP_FRAME_COUNT)//self.sample_rate
                    ret, frame = vid.read()
                    if not ret:
                        break
                    ret, boxes = multiTracker.update(frame)
                    if total_frames == self.sample_rate:
                        break
                    if frame_number % skip_rate != 0:
                        frame_number += 1
                        continue
                    total_frames += 1
                    person_no = 1
                    for i, newbox in enumerate(boxes):
                        x, y, w, h = newbox
                        img = frame[int(y):int(y + h), int(x):int(x + w)]
                        cv2.imwrite("FaceCrops/" + video_name + "/" + "Person" + str(person_no) + "/" + "Frame" + str(frame_number) + ".jpg", img)
                        person_no += 1
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    frame_number += 1
            except Exception as e:
                print(e)
                pass
        cuda.close()


class Utils:
    def __init__(self, frames_per_video):
        self.frames_per_video = frames_per_video
        self.input_size = 150
        self.input_size2 = 224

    """
    Function :- isotropically_resize_image
    Resize the face image as per the input size of the inception model.
    Input :
           img :- Face Crops generated from the main frame of the video.
           size :- Width/Height of the image. (Image is supposed to be a square)
    Returns :
            resized :- Resized image with it's higher dimension equal to size.
    """

    def isotropically_resize_image(self, img, size, resample=cv2.INTER_AREA):
        h, w = img.shape[:2]
        if w > h:
            h = h * size // w
            w = size
        else:
            w = w * size // h
            h = size

        resized = cv2.resize(img, (w, h), interpolation=resample)
        return resized

    """
    Function :- make_square_image
    Add borders to the image in order to change the shape of image to a square.
    Input :
           img :- Resized face crops from the isotropically_resize_image function.
    Returns :
            square_image :- Square image with a border of height (size-h) and width (size-w). 
    """

    def make_square_image(self, img):
        h, w = img.shape[:2]
        size = max(h, w)
        t = 0
        b = size - h
        l = 0
        r = size - w
        return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)


    """
    Function :- find_number_of_person
    Return the number of person id found after completion of detection and tracking algotihms.
    Input :
           video_face_path :- Path of the folder of a specific video where different person 
                              ID are stored.
    Returns :
            arr :- Number of different people in the entire video. 
    """

    def find_number_of_person(self, video_face_path):
        arr = os.listdir(video_face_path)
        return len(arr)

    def generate_image_tensor(self, video_face_path):
        person_count = self.find_number_of_person(video_face_path)
        person_x = []
        for i in range(1,person_count+1):
            mypath = video_face_path + "/Person" + str(i)
            onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            x = np.zeros((self.frames_per_video, self.input_size, self.input_size, 3), dtype=np.uint8)
            n = 0
            for f in onlyfiles:
                files = mypath + "/" + f
                img = cv2.imread(files)
                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                resized_face = self.isotropically_resize_image(frame, self.input_size)
                resized_face = self.make_square_image(resized_face)
                x[n] = resized_face
                n += 1
            person_x.append(x)
        return person_x

    def generate_image_tensor2(self,video_face_path):
        person_count = self.find_number_of_person(video_face_path)
        person_x = []
        for i in range(1, person_count+1):
            mypath = video_face_path + "/Person" + str(i)
            onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            x = np.zeros((self.frames_per_video, self.input_size2, self.input_size2, 3), dtype=np.uint8)
            n = 0
            for f in onlyfiles:
                files = mypath + "/" + f
                img = image.load_img(files, target_size=(224, 224))
                img_tensor = image.img_to_array(img)
                img_tensor /= 255.
                x[n] = img_tensor
                n += 1
            person_x.append(x)
        return person_x


class results:

    def find_best_threshold(self, probability_scores):
        accuracy = []
        threshold = 0.0
        threshold_graph = []
        while threshold < 1:
            y_pred = []
            y_real = []
            for keys in probability_scores.keys():
                label = int(keys.split("_")[-1])
                y_real.append(label)
                if probability_scores[keys] > threshold:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
            threshold_graph.append(threshold)
            accuracy.append(metrics.accuracy_score(y_real, y_pred))
            threshold = threshold + 0.05

        print((accuracy))
        print("+====================================================+")
        print((threshold_graph))
        print("+====================================================+")


    """
    Function :- generate_report
    This function is called in the after the predicted scores of every video in the directory has been
    generated. It is mainly used to generate confusion matrix, classification report and accuracy.
    Input :
            probability_scores :- Dictionary with video names as their keys and there predicted 
                                  probability score as the value of that specific key.
            threshold :- Real value in between 0 and 1. If probability is more than threshold, then 
                         we will consider video as fake.
    Returns :
            probability_scores :- Dictionary with video names as their keys and there predicted 
            probability score as the value of that specific key.  
    """
    def generate_report(self, probability_scores, threshold):
        y_real = []
        y_pred = []
        y_prob = []
        for keys in probability_scores.keys():
            label = int(keys.split("_")[-1])
            y_real.append(label)
            if probability_scores[keys] > threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
            y_prob.append(probability_scores[keys])

        print(metrics.confusion_matrix(y_real, y_pred))
        print(metrics.accuracy_score(y_real, y_pred))
        print(metrics.classification_report(y_real, y_pred))
        print("Log Loss value is :-", metrics.log_loss(y_real, y_prob))

    def ensemble(self,scores_resnext, scores_efficient):
        ensembled = {}
        for keys in scores_resnext.keys():
            ensembled[keys] = ((0.4 * scores_efficient[keys] + 0.6 * scores_resnext[keys]))
        return ensembled

    def ensemble3(self,scores_resnext, scores_xception, scores_efficient):
        ensembled = {}
        for keys in scores_resnext.keys():
            ensembled[keys] = ((0.3 * scores_efficient[keys] + 0.2*scores_xception[keys] + 0.5 * scores_resnext[keys]))
        return ensembled

    def dump_json(self, file_name, dictionary):
        with open(file_name+'.json', 'w') as fp:
            json.dump(dictionary, fp)

    def read_json(self,file_name):
        with open(file_name+".json") as f:
            scores1 = json.load(f)
        return scores1

class RunClassifier1:
    def __init__(self, video_directory_path, frames_per_video):
        self.utils = Utils(frames_per_video)
        self.video_directory_path = video_directory_path
        self.frames_per_video = frames_per_video

    def load_model_Xception(self):
        base_model = Xception(input_shape=(224, 224, 3),
                              weights='imagenet',
                              include_top=False,
                              pooling='avg')

        for layer in base_model.layers:
            layer.trainable = True
        x = base_model.output
        x = Dense(512)(x)
        x = Dropout(0.1)(x)
        x = Activation('relu')(x)
        x = Dense(64)(x)
        x = Dropout(0.1)(x)
        x = Activation('relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
        model.load_weights("Xception_DeepFakeFull.h5")
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        return model,test_datagen

    def load_model_efficient(self):
        base_model = EfficientNetB5(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalMaxPooling2D()(x)
        x = Dense(512)(x)
        x = Dropout(0.1)(x)
        x = Activation('relu')(x)
        x = Dense(64)(x)
        x = Dropout(0.1)(x)
        x = Activation('relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
        model.load_weights("EFB5DeepFakes.h5")
        test_datagen = ImageDataGenerator(rescale=1./255)
        return model,test_datagen

    def load_efficient_attention(self):
        print("MODEL LOADING..........")
        in_lay = Input(shape=(224, 224, 3))
        base_model = EfficientNetB6(
            input_shape=(224, 224, 3),
            weights='imagenet',
            include_top=False
        )

        pt_depth = base_model.get_output_shape_at(0)[-1]
        pt_features = base_model(in_lay)
        bn_features = BatchNormalization()(pt_features)
        # here we do an attention mechanism to turn pixels in the GAP on an off
        attn_layer = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(Dropout(0.5)(bn_features))
        attn_layer = Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
        attn_layer = Conv2D(8, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
        attn_layer = Conv2D(1, kernel_size=(1, 1), padding='valid', activation='sigmoid')(attn_layer)
        # fan it out to all of the channels
        up_c2_w = np.ones((1, 1, 1, pt_depth))
        up_c2 = Conv2D(pt_depth, kernel_size=(1, 1), padding='same',
                       activation='linear', use_bias=False, weights=[up_c2_w])
        up_c2.trainable = False
        attn_layer = up_c2(attn_layer)

        mask_features = multiply([attn_layer, bn_features])
        gap_features = GlobalAveragePooling2D()(mask_features)
        gap_mask = GlobalAveragePooling2D()(attn_layer)
        # to account for missing values from the attention model
        gap = Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_features, gap_mask])
        gap_dr = Dropout(0.25)(gap)
        dr_steps = Dropout(0.25)(Dense(128, activation='relu')(gap_dr))
        out_layer = Dense(1, activation='sigmoid')(dr_steps)
        model = Model(inputs=[in_lay], outputs=[out_layer])
        model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
        model.load_weights("EFB6withAttentionDeepFakes.h5")
        model.summary()
        test_datagen = ImageDataGenerator(rescale=1./255)
        return model,test_datagen

    def predict(self):
        model, test_datagen = self.load_efficient_attention()
        print("Model has been loaded")
        arr = os.listdir(self.video_directory_path)
        probability_scores = {}

        for video in arr:
            start = time.time()
            val_set_dir = self.video_directory_path+"/"+video
            person_count = len(os.listdir(val_set_dir))

            test_generator = test_datagen.flow_from_directory(val_set_dir,
                                                              target_size=(224, 224),
                                                              batch_size=10, class_mode='binary',
                                                              color_mode='rgb', shuffle=False)
            test_generator.reset()
            predictions = model.predict_generator(test_generator, steps=test_generator.samples//10)
            per_person = len(predictions)//person_count
            y_person = []
            for i in range(person_count):
                y_person.append(predictions[i*per_person:i*per_person + per_person].mean().item())
            probability_scores[video] = max(y_person)
            end = time.time()
            print(video, ":-", probability_scores[video], "Time Taken :-", end-start)

        return probability_scores



class RunResNext:
    def __init__(self, video_directory_path, frames_per_video):
        self.utils = Utils(frames_per_video)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize_transform = Normalize(self.mean, self.std)
        self.model = self.load_model()
        self.video_directory_path = video_directory_path
        self.frames_per_video = frames_per_video

    """
    Function :- load_model
    This function is called in the when the object of RunResNext is created. Inception ResNet model is
    loaded using the function.
    Input :
           None
    Returns :
            model :- Instance of the loaded model.
    """
    def load_model(self):
        checkpoint = torch.load("resnext.pth", map_location=gpu)
        model = MyResNeXt().to(gpu)
        model.load_state_dict(checkpoint)
        _ = model.eval()
        del checkpoint
        return model

    """
    Function :- predict
    This function is called in the after the model has been loaded. It is used predict the probability
    score of each video present in the self.video_directory_path.
    Input :
           None
    Returns :
            probability_scores :- Dictionary with video names as their keys and there predicted 
                                  probability score as the value of that specific key.  
    """
    def predict(self):
        arr = os.listdir(self.video_directory_path)
        probability_scores = {}
        for video in arr:
            video_path = self.video_directory_path+"/"+video
            face_tensor = self.utils.generate_image_tensor(video_path)
            y_person = []
            for i in range(len(face_tensor)):
                x = face_tensor[i]
                x = torch.tensor(x, device=gpu).float()
                x = x.permute((0, 3, 1, 2))
                for i in range(len(x)):
                    x[i] = self.normalize_transform(x[i] / 255.)
                with torch.no_grad():
                    y_pred = self.model(x)
                    y_pred = torch.sigmoid(y_pred.squeeze())
                    y_person.append(y_pred[:self.frames_per_video].mean().item())
            probability_scores[video] = max(y_person)
            print(video, ":-", probability_scores[video])
        return probability_scores


if __name__ == "__main__":
    frames_per_video = 30
    Result = results()

    
    video_directory = "/media/kenil/Kenil/Users/Kenil/Downloads/Final_Project/test_videos"
    #Result = results()
    face_crop_object = GenerateFaceCrops(video_directory, frames_per_video)
    face_crop_object.face_crops()
    
    image_directory = "/media/kenil/Kenil/Users/Kenil/Downloads/Final_Project/Main/FaceCrops"

    resnext = RunResNext(image_directory , frames_per_video)
    scores1 = resnext.predict()

    EfficientNet = RunClassifier1(image_directory frames_per_video)
    EfficientNet.load_efficient_attention()
    scores2 = EfficientNet.predict()

    print("************************************************")
    print("ResNext Results")
    # Result.find_best_threshold(scores1)
    Result.generate_report(scores1, 0.6)
    print("************************************************")

    print("************************************************")
    print("EfficientNet Attention Results")
    Result.find_best_threshold(scores4  )
    Result.generate_report(scores4, 0.5)
    print("************************************************")

    print("************************************************")
    print("EfficientNet Attention + ResNext Ensemble Results")
    ensembled = Result.ensemble(scores4, scores1)
    Result.find_best_threshold(ensembled)
    Result.generate_report(ensembled, 0.5)
    print("************************************************")
