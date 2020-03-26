import numpy as np
from keras.preprocessing import image
from keras.models import load_model

classifier=load_model("mymodel_family.h5")
# test_image = image.load_img('D:\\Personal\\backup\\cat\\5614.jpg', target_size = (64, 64))
test_image = image.load_img('D:\\Personal\\CNN_Images\\IMG_20170608_073824.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
# training_set.class_indices
if result[0][0] == 1:
    prediction = 'Nitesh'
    print(prediction)
else:
    prediction = 'Dhairya'
    print(prediction)