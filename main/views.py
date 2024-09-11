from django.shortcuts import render, redirect
from .models import UploadImg
from django.contrib import messages
from keras.preprocessing import image
import numpy as np
from keras.models import Model
from keras.applications.densenet import DenseNet121
from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
            BatchNormalization, Input, Conv2D, GlobalAveragePooling2D, concatenate, Concatenate)
from keras.applications.resnet50 import ResNet50

import threading

def index(request):
    return render(request, 'index.html')

def contact(request):
    return render(request, 'contact.html')

def manage(request):
    return render(request, 'manage.html', {})

def about(request):
    return render(request, 'about.html', {})

def dashboard(request):
    image_tbl = UploadImg.objects.all()
    return render(request, 'dashboard.html', {'image_tbl': image_tbl})

def upload_images(request):
    if request.method == "POST":
        patient_id = request.POST['pid']
        patient_name = request.POST['pname']
        patient_age = request.POST['page']
        chestpics = request.FILES['chestpics']
        img = UploadImg(pid=patient_id, patient_name=patient_name, page=patient_age,
                        chestpics=chestpics)
        img.save()
        print("Newly saved item id:", img.id)
        messages.success(request, "Detail has been uploaded to Database.Please go to report section.")

        ## Start New Thread
        t = threading.Thread(target=run_prediction, args=(img.id,))
        t.setDaemon(True)
        t.start()
        return render(request, 'manage.html')
    else:
        return render(request, 'manage.html')

def delete_pics(request, patient_id):
    item = UploadImg.objects.filter(pk=patient_id)
    item.delete()
    messages.success(request, "Deleted Successfully!!!")
    return redirect('dashboard')

def update_pics(request, patient_id):
    item = UploadImg.objects.filter(pk=patient_id)
    return render(request, 'update.html', {'items': item})

def create_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = ResNet50(input_tensor=input_tensor, include_top=False, pooling='average')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(n_out, activation='softmax')(x)
    model = Model(base_model.input, x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def run_prediction(record_id):
    item = UploadImg.objects.filter(pk=record_id)
    chest_files = item[0].chestpics
    chest_pred = {0: 'Normal', 1: 'Pneumonia'}

    # Load Model
    SIZE = 100
    NUM_CLASSES = 2
    model = create_model(
        input_shape=(SIZE, SIZE, 3),
        n_out=NUM_CLASSES)
    model.load_weights('models/model_weights.h5')

    chest_img = image.load_img(chest_files, target_size=(SIZE, SIZE))
    chest_img = image.img_to_array(chest_img)
    chest_img = np.expand_dims(chest_img, axis=0)
    chest_ig_predict = model.predict(chest_img)
    chest_score_predict = chest_pred[np.argmax(chest_ig_predict)]
    chest_score = np.max(chest_ig_predict)
    print('Prediction: ', chest_score_predict, chest_score)

    get_patient_rec = UploadImg.objects.get(pk=record_id)
    get_patient_rec.chest_prediction = chest_score_predict
    get_patient_rec.chest_score = round(chest_score * 100, 2)
    get_patient_rec.save()

    # Clear backend for next prediction
    from keras import backend as K
    K.clear_session()
    return redirect('dashboard')

def run_report(request, patient_id):
    item = UploadImg.objects.filter(pk=patient_id)

    return render(request, 'report.html', {'items': item})