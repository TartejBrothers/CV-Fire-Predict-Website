import os
from django.conf import settings
from django.template.response import TemplateResponse
from django.utils.datastructures import MultiValueDictKeyError
from django.core.files.storage import FileSystemStorage
import matplotlib.image as mpimg
import onnxruntime
import numpy as np
import cv2


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

        path = os.path.join(settings.MEDIA_ROOT, _image)

        image = mpimg.imread(path)
        if image is None:
            raise ValueError(
                "Failed to read the image. Make sure the file path is correct."
            )

        resized_image = cv2.resize(image, (256, 256))
        normalized_image = resized_image / 256.0
        input_image = np.expand_dims(normalized_image.astype(np.float32), axis=0)

        model_path = os.path.join(settings.BASE_DIR, "model.onnx")
        ort_session = onnxruntime.InferenceSession(model_path)

        input_name = ort_session.get_inputs()[0].name
        result = ort_session.run(None, {input_name: input_image})

        Fire = result[0][0][0] >= 0.5
        if Fire:
            prediction = "Fire"
        else:
            prediction = "No Fire"

        filename = _image
        return TemplateResponse(
            request,
            "index.html",
            {
                "message": message,
                "filename": filename,
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
