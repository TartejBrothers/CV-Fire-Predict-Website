import os
import cv2
from PIL import Image
import numpy as np

import tensorflow as tf
from django.conf import settings
from django.template.response import TemplateResponse
from django.utils.datastructures import MultiValueDictKeyError
from django.core.files.storage import FileSystemStorage


class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name


def index(request):
    message = ""
    prediction = ""
    fss = CustomFileSystemStorage()

    try:
        image = request.FILES["image"]
        _image = fss.save(image.name, image)

        # Use full system path instead of fss.url(_image)
        path = os.path.join(settings.MEDIA_ROOT, _image)
        print("Image Path:", path)

        # Read the image
        imag = cv2.imread(path)

        if imag is None:
            raise ValueError(
                "Failed to read the image. Make sure the file path is correct."
            )

        img_from_ar = Image.fromarray(imag, "RGB")

        # Resize the image to match the model's input shape (256, 256)
        resized_image = img_from_ar.resize((256, 256))

        # Convert the resized image to a NumPy array
        test_image = np.expand_dims(np.array(resized_image), axis=0)

        # Load the model
        model = tf.keras.models.load_model(os.path.join(settings.BASE_DIR, "model.h5"))

        # Make predictions
        result = model.predict(test_image)
        print("Prediction: " + str(np.argmax(result)))

        if np.argmax(result) == 0:
            prediction = "Fire"
        elif np.argmax(result) == 1:
            prediction = "No Fire"
        else:
            prediction = "Unknown"

        return TemplateResponse(
            request,
            "index.html",
            {
                "message": message,
                "image": image,
                "image_url": fss.url(_image),
                "prediction": prediction,
            },
        )

    except MultiValueDictKeyError:
        return TemplateResponse(
            request,
            "index.html",
            {"message": "No Image Selected"},
        )
    except Exception as e:
        return TemplateResponse(
            request,
            "index.html",
            {"message": str(e)},
        )
