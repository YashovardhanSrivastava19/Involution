import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf

""" 
    Credits: https://keras.io/examples/vision/involution/
    arXiv : https://arxiv.org/abs/2103.06255

"""

""" While Convolution NN is the go-to for most computer vision related experiments,the C Kernel is spatial agnostic and 
    channel-specific. Involution NN (Inverting the Inherence of Convolution for VisualRecognition) addresses these issues-
    through Involution Kernel that is location-specific and channel-agnostic. Self attention falls under the design paradigm
    of involution.

    To make kernel location-specific, the authors have considered generating each kernel conditioned on 
    specific spatial positions.  

"""

EPOCHS = 2

class InvolutionLayer(tf.keras.layers.Layer):
    def __init__(self,Channel,Group_Number,Kernel_Size,Stride,Reduction_Ratio,**kwargs):
        super().__init__(**kwargs)
        self.channel = Channel
        self.group_number = Group_Number
        self.kernel_size = Kernel_Size
        self.stride = Stride
        self.red_ratio = Reduction_Ratio

    def build(self, input_shape):
        (_,height,width,num_channels) = input_shape
        height = height // self.stride
        width = width // self.stride

        self.stride_layer = (tf.keras.layers.AveragePooling2D(
            pool_size=self.stride,strides=self.stride,padding="same"
            )
            if self.stride > 1
            else tf.identity
        )

        self.kernel_gen = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.channel // self.red_ratio,kernel_size=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=self.kernel_size*self.kernel_size*self.group_number,kernel_size=1)
        ])


        self.kerenl_reshape = tf.keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size*self.kernel_size,
                1,
                self.group_number,
            )
        )

        self.input_pathches_reshape = tf.keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size*self.kernel_size,
                num_channels//self.group_number,
                self.group_number,
            )
        )
        
        self.output_reshape = tf.keras.layers.Reshape(
            target_shape=(height,width,num_channels)
        )

    def call(self,x):
        kernel_input = self.stride_layer(x)
        kernel = self.kernel_gen(kernel_input)
        kernel = self.kerenl_reshape(kernel)
        
        input_patches = tf.image.extract_patches(
            images=x,
            sizes=[1,self.kernel_size,self.kernel_size,1],
            strides=[1,self.stride,self.stride,1],
            rates=[1,1,1,1],
            padding='SAME',
        ) 

        input_patches = self.input_pathches_reshape(input_patches)

        output = tf.multiply(kernel,input_patches)

        output = tf.reduce_sum(output,axis=3)

        output = self.output_reshape(output)

        return output,kernel

(xTrain,yTrain),(xTest,yTest) = tf.keras.datasets.cifar10.load_data()

(xTrain,xTest) = (xTrain/255.0 , xTest/255.0)

TrainDataset = tf.data.Dataset.from_tensor_slices((xTrain,yTrain)).shuffle(256).batch(256)

TestDataset = tf.data.Dataset.from_tensor_slices((xTest,yTest)).batch(256)

print("Convolution Neural Network:\n")

conv_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(32,32,3)),
    tf.keras.layers.Conv2D(32,(3,3),padding="same"),
    tf.keras.layers.ReLU(name="ReLU_1"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64,(3,3),padding="same"),
    tf.keras.layers.ReLU(name="ReLU_2"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64,(3,3),padding="same"),
    tf.keras.layers.ReLU(name="ReLU_3"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dense(10),    
])

conv_model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

""" fit method returns an history object which is a record of meterics """
conv_hist = conv_model.fit(TrainDataset,epochs=EPOCHS,validation_data=TestDataset)

Cscore = conv_model.evaluate(TestDataset)
print("\nConvolution--> Loss:{} Accuracy:{}\n".format(Cscore[0],Cscore[1]))

print("Involution Neural Network:\n")

inputs = tf.keras.layers.Input(shape=(32,32,3))

x,_ = InvolutionLayer(Channel=3,Group_Number=1,Kernel_Size=3,Stride=1,Reduction_Ratio=2,name="INV_1")(inputs)

x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPooling2D()(x)

x,_ = InvolutionLayer(Channel=3,Group_Number=1,Kernel_Size=3,Stride=1,Reduction_Ratio=2,name="INV_2")(x)

x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPooling2D()(x)

x,_ = InvolutionLayer(Channel=3,Group_Number=1,Kernel_Size=3,Stride=1,Reduction_Ratio=2,name="INV_3")(x)

x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Flatten()(x)


x = tf.keras.layers.Dense(64,activation="relu")(x)
outputs = tf.keras.layers.Dense(10)(x)

inv_model = tf.keras.Model(inputs=[inputs],outputs=[outputs],name="INV_MODEL")


inv_model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

inv_hist = inv_model.fit(TrainDataset,epochs=EPOCHS,validation_data=TestDataset)


Iscore = inv_model.evaluate(TestDataset)
print("\nInvolution--> Loss:{} Accuracy:{}\n".format(Iscore[0],Iscore[1]))

import matplotlib.pyplot as plt

plt.figure(figsize=(20,5))

plt.subplot(1,2,1)
plt.title("Convolution Loss")
plt.plot(conv_hist.history['loss'],label="Loss")
plt.plot(conv_hist.history['val_loss'],label="Value Loss")
plt.legend()


plt.subplot(1,2,2)
plt.title("Involution Loss")
plt.plot(inv_hist.history['loss'],label="Loss")
plt.plot(inv_hist.history['val_loss'],label="Value Loss")
plt.legend()

plt.show()

plt.figure(figsize=(20,5))

plt.subplot(1,2,1)
plt.title("Convolution Accuracy")
plt.plot(conv_hist.history['accuracy'],label="Accuracy")
plt.plot(conv_hist.history['val_accuracy'],label="Value Accuracy")
plt.legend()


plt.subplot(1,2,2)
plt.title("Involution Accuracy")
plt.plot(inv_hist.history['accuracy'],label="Accuracy")
plt.plot(inv_hist.history['val_accuracy'],label="Value Accuracy")
plt.legend()

plt.show()
