"""

"""

import cv2
import math
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
import hnswlib
import threading
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}


class Index:
    def __init__(self, space, dim):
        self.index = hnswlib.Index(space, dim)
        self.lock = threading.Lock()
        self.dict_labels = {}
        self.cur_ind = 0

    def init_index(self, max_elements, ef_construction=200, M=16):
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)

    def add_items(self, data, ids=None):
        if ids is not None:
            assert len(data) == len(ids)
        num_added = len(data)
        with self.lock:
            start = self.cur_ind
            self.cur_ind += num_added
        int_labels = []

        if ids is not None:
            for dl in ids:
                int_labels.append(start)
                self.dict_labels[start] = dl
                start += 1
        else:
            for _ in range(len(data)):
                int_labels.append(start)
                self.dict_labels[start] = start
                start += 1
        self.index.add_items(data=data, ids=np.asarray(int_labels))

    def set_ef(self, ef):
        self.index.set_ef(ef)

    def load_index(self, path):
        self.index.load_index(path)
        with open(path + ".pkl", "rb") as f:
            self.cur_ind, self.dict_labels = pickle.load(f)

    def save_index(self, path):
        self.index.save_index(path)
        with open(path + ".pkl", "wb") as f:
            pickle.dump((self.cur_ind, self.dict_labels), f)

    def set_num_threads(self, num_threads):
        self.index.set_num_threads(num_threads)

    def knn_query(self, data, k=1):
        labels_int, distances = self.index.knn_query(data=data, k=k)
        labels = []
        for li in labels_int:
            line = []
            for l in li:
                line.append(self.dict_labels[l])
            labels.append(line)
        return labels, distances


def train(train_dir, model_save_path=None, num_threads=4, verbose=False):
    """
    Trains a knn classifier for face recognition.
    :param train_dir: directory that contains a sub-directory for each known person, with its name coded by DIGITS.
     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
    :param model_save_path: (optional) path to save model on disk
    :param num_threads: Set number of threads used during batch search/construction
    :param verbose: verbose mode for training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Create and train the KNN classifier
    model = Index(space='l2', dim=128)
    model.init_index(len(X))
    model.set_ef(10)
    model.set_num_threads(num_threads)
    model.add_items(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        model.save_index(model_save_path)

    return model

# TODO: change prediction function


def load_model(model, model_path):
    """
    Loads saved index of the created model
    :param model: model instance
    :param model_path: path to index
    """
    model.load_index(model_path)


def predict(model, X_frame, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_frame: frame to do the prediction on.
    :param model: a knn classifier object.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if model is None:
        raise Exception("Must supply knn_clf")

    X_face_locations = face_recognition.face_locations(X_frame)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    labels, distances = model.knn_query(faces_encodings, k=1)
    # TODO: type of answer investigate
    are_matches = [distances[i] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(labels, X_face_locations, are_matches)]


def show_prediction_labels_on_image(frame, predictions):
    """
    Shows the face recognition results visually.
    :param frame: frame to show the predictions on
    :param predictions: results of the predict function
    :return opencv suited image to be fitting with cv2.imshow fucntion:
    """
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # enlarge the predictions for the full sized image.
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs.
    del draw
    # Save image in open-cv format to be able to show it.

    opencvimage = np.array(pil_image)
    return opencvimage


if __name__ == "__main__":
    print("Training KNN classifier...")
    # classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")
    # process one frame in every 30 frames for speed
    process_this_frame = 29
    print('Setting cameras up...')
    # multiple cameras can be used with the format url = 'http://username:password@camera_ip:port'
    url = 'http://admin:admin@192.168.0.106:8081/'
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if ret:
            # Different resizing options can be chosen based on desired program runtime.
            # Image resizing for more stable streaming
            img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            process_this_frame = process_this_frame + 1
            if process_this_frame % 30 == 0:
                # TODO: make predictions
                predictions = predict(img, model_path="trained_knn_model.clf")
            frame = show_prediction_labels_on_image(frame, predictions)
            cv2.imshow('camera', frame)
            if ord('q') == cv2.waitKey(10):
                cap.release()
                cv2.destroyAllWindows()
                exit(0)
