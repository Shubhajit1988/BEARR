import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import random
import cv2

!pip install cleverhans
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent as pgd
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method as bim

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



t=0
s=0
m=0
BDERP=0
eps=8/255
for im in range(1000):
    output=np.zeros(10)
    count=0
    count_=0
    x=test_images[im]
    x=tf.reshape(x,[1,32,32,3])
    y=test_labels[im]

    adv_x_base=bim(model_base,x,eps,1/255,10,np.inf)
    adv_noise_base=adv_x_base-x
    adv_x_det=bim(model_binary_det,adv_x_base,eps,1/255,10,np.inf)
    adv_noise_det=adv_x_det-adv_x_base
    adv_x_rand1=bim(model_rand1,x,eps,1/255,10,np.inf)
    adv_noise_rand1=adv_x_rand1-x
    adv_x_rand2=bim(model_rand2,x,eps,1/255,10,np.inf)
    adv_noise_rand2=adv_x_rand2-x

    imgv1=np.array(x)
    imgv1=imgv1.astype(np.float32)
    imgv1=rotate_image(imgv1[0],1)
    imgv1=tf.reshape(imgv1,[1,32,32,3])

    adv_img4_1 = bim(model_base, imgv1,eps,1/255,10,np.inf)
    adv_noise4_1=adv_img4_1-imgv1
    adv_noise4_1=np.array(adv_noise4_1)
    adv_noise4_1=adv_noise4_1.astype(np.float32)
    adv_noise4_1=rotate_image(adv_noise4_1[0],-1)
    adv_noise4_1=tf.reshape(adv_noise4_1,[1,32,32,3])

    imgv1=np.array(x)
    imgv1=imgv1.astype(np.float32)
    imgv1=rotate_image(imgv1[0],2)
    imgv1=tf.reshape(imgv1,[1,32,32,3])

    adv_img4_2 = bim(model_base, imgv1,eps,1/255,10,np.inf)
    adv_noise4_2=adv_img4_1-imgv1
    adv_noise4_2=np.array(adv_noise4_2)
    adv_noise4_2=adv_noise4_2.astype(np.float32)
    adv_noise4_2=rotate_image(adv_noise4_2[0],-2)
    adv_noise4_2=tf.reshape(adv_noise4_2,[1,32,32,3])


    imgv1=np.array(x)
    imgv1=imgv1.astype(np.float32)
    imgv1=rotate_image(imgv1[0],3)
    imgv1=tf.reshape(imgv1,[1,32,32,3])

    adv_img4_3 = bim(model_base, imgv1,eps,1/255,10,np.inf)
    adv_noise4_3=adv_img4_3-imgv1
    adv_noise4_3=np.array(adv_noise4_3)
    adv_noise4_3=adv_noise4_3.astype(np.float32)
    adv_noise4_3=rotate_image(adv_noise4_3[0],-3)
    adv_noise4_3=tf.reshape(adv_noise4_3,[1,32,32,3])


    imgv1=np.array(x)
    imgv1=imgv1.astype(np.float32)
    imgv1=rotate_image(imgv1[0],-1)
    imgv1=tf.reshape(imgv1,[1,32,32,3])

    adv_img4_4 = bim(model_base, imgv1,eps,1/255,10,np.inf)
    adv_noise4_4=adv_img4_4-imgv1
    adv_noise4_4=np.array(adv_noise4_4)
    adv_noise4_4=adv_noise4_4.astype(np.float32)
    adv_noise4_4=rotate_image(adv_noise4_4[0],1)
    adv_noise4_4=tf.reshape(adv_noise4_4,[1,32,32,3])

    imgv1=np.array(x)
    imgv1=imgv1.astype(np.float32)
    imgv1=rotate_image(imgv1[0],-2)
    imgv1=tf.reshape(imgv1,[1,32,32,3])

    adv_img4_5 = bim(model_base,imgv1,eps,1/255,10,np.inf)
    adv_noise4_5=adv_img4_5-imgv1
    adv_noise4_5=np.array(adv_noise4_5)
    adv_noise4_5=adv_noise4_5.astype(np.float32)
    adv_noise4_5=rotate_image(adv_noise4_5[0],2)
    adv_noise4_5=tf.reshape(adv_noise4_5,[1,32,32,3])


    imgv1=np.array(x)
    imgv1=imgv1.astype(np.float32)
    imgv1=rotate_image(imgv1[0],-3)
    imgv1=tf.reshape(imgv1,[1,32,32,3])

    adv_img4_6 = bim(model_base, imgv1,eps,1/255,10,np.inf)
    adv_noise4_6=adv_img4_6-imgv1
    adv_noise4_6=np.array(adv_noise4_6)
    adv_noise4_6=adv_noise4_6.astype(np.float32)
    adv_noise4_6=rotate_image(adv_noise4_6[0],3)
    adv_noise4_6=tf.reshape(adv_noise4_6,[1,32,32,3])

    adv_noise_rotate=(adv_noise4_1+adv_noise4_2+adv_noise4_3+adv_noise4_4+adv_noise4_5+adv_noise4_6)/6


    imgv1=np.array(x)
    img_=np.zeros((1,24,24,3))
    for ii in range(3):
        img_[0,:,:,ii] = cv2.resize(imgv1[0,:,:,ii], dsize=(24, 24))
        img2_=np.zeros((1,32,32,3))
        for ij in range(3):
            img2_[0,:,:,ij] = cv2.resize(img_[0,:,:,ij], dsize=(32, 32))
            img2__=tf.reshape(img2_,[1,32,32,3])


    imgv1=np.array(img2__)
    for l in range(10):
        i_=random.randint(1,6)
        j_=random.randint(1,6)
        for i__ in range(4):
            for j__ in range(4):
                for k__ in range(3):
                    imgv1[0][i__+i_*4][j__+j_*4][k__]=1

    imgv1=np.float32(imgv1)

    imgv1=tf.reshape(imgv1,[1,32,32,3])

    adv_x_patch = bim(model_patch, imgv1,eps,1/255,10,np.inf)
    adv_noise_patch1=adv_x_patch-imgv1

    imgv1=np.array(x)
    img_=np.zeros((1,16,16,3))
    for ii in range(3):
        img_[0,:,:,ii] = cv2.resize(imgv1[0,:,:,ii], dsize=(16, 16))
        img2_=np.zeros((1,32,32,3))
        for ij in range(3):
            img2_[0,:,:,ij] = cv2.resize(img_[0,:,:,ij], dsize=(32, 32))
            img2__=tf.reshape(img2_,[1,32,32,3])


    imgv1=np.array(img2__)
    for l in range(10):
        i_=random.randint(1,6)
        j_=random.randint(1,6)
        for i__ in range(4):
            for j__ in range(4):
                for k__ in range(3):
                    imgv1[0][i__+i_*4][j__+j_*4][k__]=1

    imgv1=np.float32(imgv1)

    imgv1=tf.reshape(imgv1,[1,32,32,3])

    adv_x_patch = bim(model_patch, imgv1,eps,1/255,10,np.inf)
    adv_noise_patch2=adv_x_patch-imgv1

    imgv1=np.array(x)
    for l in range(10):
        i_=random.randint(1,6)
        j_=random.randint(1,6)
        for i__ in range(4):
            for j__ in range(4):
                for k__ in range(3):
                    imgv1[0][i__+i_*4][j__+j_*4][k__]=1

    imgv1=np.float32(imgv1)

    imgv1=tf.reshape(imgv1,[1,32,32,3])

    adv_x_patch = bim(model_patch, imgv1,eps,1/255,10,np.inf)
    adv_noise_patch3=adv_x_patch-imgv1

    adv_noise_patch=(adv_noise_patch1+adv_noise_patch2+adv_noise_patch3)/3

    adv_x = x + (1/6)*(adv_noise_base) + (1/6)*(adv_noise_det) + (1/6)*(adv_noise_rand1) + (1/6)*(adv_noise_rand2)  + (1/6)*(adv_noise_patch) + (1/6)*(adv_noise_rotate)
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