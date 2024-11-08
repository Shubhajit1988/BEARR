import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import random
import cv2

model_base = tf.keras.models.load_model('/content/M_B.h5')
model_rand1 = tf.keras.models.load_model('/content/M_1.h5')
model_rand2 = tf.keras.models.load_model('/content/M_2.h5')
model_patch = tf.keras.models.load_model('/content/M_P.h5')
model_int = Model(inputs=model_base.inputs, outputs=model_base.layers[40].output)
model_det = tf.keras.models.load_model('/content/M_D.h5')


inputs = tf.keras.Input(shape=(32, 32, 3))
x = model_int(inputs)
outputs = model_det(x)

model_binary_det = tf.keras.Model(inputs, outputs)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images=train_images.astype(np.float32)
test_images=test_images.astype(np.float32)
num_classes=10
train_labels=tf.keras.utils.to_categorical(train_labels,num_classes)
test_labels=tf.keras.utils.to_categorical(test_labels,num_classes)


train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255.0
test_images /= 255.0

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern_prim(input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model_base(input_image)
        loss = loss_object(input_label, prediction)
    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad

def random_transform(image):
  image=np.array(image)
  si=random.randint(16,32)
                #si=24
  img=np.zeros((1,si,si,3))
  for ii in range(3):
    img[0,:,:,ii] = cv2.resize(image[0,:,:,ii], dsize=(si, si))
  resize_index=32-si
  img2=np.zeros((1,32,32,3))
  for ij in range(3):
    for ii_ in range(si):
      for jj_ in range(si):
        img2[0][ii_+resize_index][jj_+resize_index][ij]=img[0][ii_][jj_][ij]
  trans_image=np.array(img2)
  return trans_image

def diifgsm(image):
    n_classes = 10 #n_classes=10 for MNIST and CIFAR-10, n_classes=100 for CIFAR-100
    epsilon = 8/255    # attack strength
    iterations = 10
    alpha = 1/255
    image = tf.reshape(image,[1, 32, 32, 3])
    labels=np.argmax(model_base(image))
    labels=tf.one_hot(labels,10)
    labels=tf.reshape(labels,(1,10))
    adv_img = image
    for iters in range(iterations):
        imgv = tf.Variable(adv_img)
        with tf.GradientTape() as tape:
            tape.watch(imgv)
            predictions = model_base(imgv)
            loss = tf.keras.losses.CategoricalCrossentropy()(labels, predictions)
            grads = tape.gradient(loss,imgv)
        signed_grads1 = tf.sign(grads)

        transformed_imgv=random_transform(imgv)
        transformed_imgv=tf.convert_to_tensor(transformed_imgv)

        with tf.GradientTape() as tape:
            tape.watch(transformed_imgv)
            predictions = model_base(transformed_imgv)

            loss = tf.keras.losses.CategoricalCrossentropy()(labels, predictions)

            grads = tape.gradient(loss,transformed_imgv)

        signed_grads2 = tf.sign(grads)
        signed_grads1=np.array(signed_grads1,dtype='float32')
        signed_grads2=np.array(signed_grads2,dtype='float32')

        adv_img = adv_img + (0.5*alpha)*signed_grads1 +(0.5*alpha)*signed_grads2
        adv_img = tf.clip_by_value(adv_img, image-epsilon, image+epsilon)
        adv_img = tf.clip_by_value(adv_img, 0, 1)
    return adv_img

t=0
s=0
m=0
BDERP=0
eps=8/255
for im in range(100):
    output=np.zeros(10)
    count=0
    count_=0
    x=test_images[im]
    x=tf.convert_to_tensor(x)
    x=tf.reshape(x,[1,32,32,3])
    y=test_labels[im]
    label=np.argmax(y)
    label=tf.one_hot(label,10)
    label=tf.reshape(label,(1,10))

    adv_x=diifgsm(x)
    adv_x=tf.clip_by_value(adv_x,0,1)

    y_=np.argmax(model_base(adv_x))
    label_=tf.one_hot(y_,10)
    label_=tf.reshape(label_,(1,10))

    adv_img=np.array(adv_x)
    adv_img=adv_img.astype(np.float32)

    adv_x_rotate1=rotate_image(adv_img[0],1)
    adv_x_rotate1=tf.reshape(adv_x_rotate1,[1,32,32,3])
    adv_x_rotate2=rotate_image(adv_img[0],2)
    adv_x_rotate2=tf.reshape(adv_x_rotate2,[1,32,32,3])
    adv_x_rotate3=rotate_image(adv_img[0],3)
    adv_x_rotate3=tf.reshape(adv_x_rotate3,[1,32,32,3])
    adv_x_rotate4=rotate_image(adv_img[0],-1)
    adv_x_rotate4=tf.reshape(adv_x_rotate4,[1,32,32,3])
    adv_x_rotate5=rotate_image(adv_img[0],-2)
    adv_x_rotate5=tf.reshape(adv_x_rotate5,[1,32,32,3])
    adv_x_rotate6=rotate_image(adv_img[0],-3)
    adv_x_rotate6=tf.reshape(adv_x_rotate6,[1,32,32,3])
    adv_x_rotate7=rotate_image(adv_img[0],0)
    adv_x_rotate7=tf.reshape(adv_x_rotate7,[1,32,32,3])


    perturbation_1=create_adversarial_pattern_prim(adv_x_rotate1,label_)
    perturbation_2=create_adversarial_pattern_prim(adv_x_rotate2,label_)
    perturbation_3=create_adversarial_pattern_prim(adv_x_rotate3,label_)
    perturbation_4=create_adversarial_pattern_prim(adv_x_rotate4,label_)
    perturbation_5=create_adversarial_pattern_prim(adv_x_rotate5,label_)
    perturbation_6=create_adversarial_pattern_prim(adv_x_rotate6,label_)
    perturbation_7=create_adversarial_pattern_prim(adv_x_rotate7,label_)
    epsilon_=0.0004
    adv_x_1=adv_x_rotate1+epsilon_*perturbation_1
    adv_x_2=adv_x_rotate2+epsilon_*perturbation_2
    adv_x_3=adv_x_rotate3+epsilon_*perturbation_3
    adv_x_4=adv_x_rotate4+epsilon_*perturbation_4
    adv_x_5=adv_x_rotate5+epsilon_*perturbation_5
    adv_x_6=adv_x_rotate6+epsilon_*perturbation_6
    adv_x_7=adv_x_rotate7+epsilon_*perturbation_7

    if np.argmax(model_base(adv_x))!=np.argmax(model_base(adv_x_1)):
        count=count+1
    if np.argmax(model_base(adv_x))!=np.argmax(model_base(adv_x_2)):
        count=count+1
    if np.argmax(model_base(adv_x))!=np.argmax(model_base(adv_x_3)):
        count=count+1
    if np.argmax(model_base(adv_x))!=np.argmax(model_base(adv_x_4)):
        count=count+1
    if np.argmax(model_base(adv_x))!=np.argmax(model_base(adv_x_5)):
        count=count+1
    if np.argmax(model_base(adv_x))!=np.argmax(model_base(adv_x_6)):
        count=count+1
    if np.argmax(model_base(adv_x))!=np.argmax(model_base(adv_x_7)):
        count=count+1

    adv=1
    if np.argmax(model_binary_det(adv_x))==0:
        if np.argmax(model_base(adv_x))==np.argmax(model_rand1(adv_x)):
            if count<4:
                BDERP=BDERP+1
                adv=0


    if adv==0:
        output_ = (model_base(adv_x)+model_rand1(adv_x)+model_rand2(adv_x))/3
        if np.argmax(output_)==np.argmax(y):
            m=m+1
    elif adv==1:
        t=t+1
        for iiii in range(10):
            adv_img__=np.array(adv_x)
            si=random.randint(16,32)
                #si=24
            img=np.zeros((1,si,si,3))
            for ii in range(3):
                img[0,:,:,ii] = cv2.resize(adv_img__[0,:,:,ii], dsize=(si, si))
            img2=np.zeros((1,32,32,3))
            for ij in range(3):
                img2[0,:,:,ij] = cv2.resize(img[0,:,:,ij], dsize=(32, 32))



            adv_img_=np.array(img2)
            for l in range(10):
                i_=random.randint(1,6)
                j_=random.randint(1,6)
                for i in range(4):
                    for j in range(4):
                        for k in range(3):
                            adv_img_[0][i+i_*4][j+j_*4][k]=1


            output=output+model_patch(adv_img_)
        output=output/10

        output_=(0.73)*output + (0.27)*model_base(adv_x)


        if np.argmax(output)==np.argmax(model_base(adv_x)):


            if output[0][np.argsort(output)[0][-1]]>model_base(adv_x)[0][np.argsort(output)[0][-1]]:
                out_= np.argmax(output)

            elif output[0][np.argsort(output)[0][-1]]>0.7:
                out_=np.argmax(output)
                
            else:
                out_=np.argsort(output_)[0][-2]
                
        elif np.argsort(output)[0][-1]!=np.argmax(model_base(adv_x)):
            out_=np.argsort(output_)[0][-1]

        if out_==np.argmax(y):
            s=s+1
total_input=BDERP+t
correctly_classified=m+s
classification_accuracy=(correctly_classified/total_input)*100
print('classification accuracy:',classification_accuracy)