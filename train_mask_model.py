{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "b8b507c0-48af-425a-b941-b477970127e1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import cv2\n",
    "import numpy as np\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "fe858e55-a696-4103-9517-aff62271d8a8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dataset loaded: 7553 images\n"
     ]
    }
   ],
   "source": [
    "data = []\n",
    "labels = []\n",
    "categories = [\"with_mask\", \"without_mask\"]\n",
    "\n",
    "for category in categories:\n",
    "    path = os.path.join(\"C:/Users/JAAVANIKA L/fall semester 22-23/Downloads/face-mask-detector/dataset\", category)\n",
    "    class_num = categories.index(category)\n",
    "    for img in os.listdir(path):\n",
    "        try:\n",
    "            img_array = cv2.imread(os.path.join(path, img))\n",
    "            resized = cv2.resize(img_array, (100, 100))\n",
    "            data.append(resized)\n",
    "            labels.append(class_num)\n",
    "        except Exception as e:\n",
    "            print(f\"Error loading image: {e}\")\n",
    "\n",
    "X = np.array(data) / 255.0\n",
    "y = to_categorical(labels)\n",
    "\n",
    "print(f\"Dataset loaded: {X.shape[0]} images\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "a4f2f3b8-2ea0-40d4-a4ff-8a6151143f60",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " conv2d (Conv2D)             (None, 98, 98, 32)        896       \n",
      "                                                                 \n",
      " max_pooling2d (MaxPooling2D  (None, 49, 49, 32)       0         \n",
      " )                                                               \n",
      "                                                                 \n",
      " conv2d_1 (Conv2D)           (None, 47, 47, 64)        18496     \n",
      "                                                                 \n",
      " max_pooling2d_1 (MaxPooling  (None, 23, 23, 64)       0         \n",
      " 2D)                                                             \n",
      "                                                                 \n",
      " flatten (Flatten)           (None, 33856)             0         \n",
      "                                                                 \n",
      " dense (Dense)               (None, 64)                2166848   \n",
      "                                                                 \n",
      " dense_1 (Dense)             (None, 2)                 130       \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 2,186,370\n",
      "Trainable params: 2,186,370\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model = Sequential([\n",
    "    Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),\n",
    "    MaxPooling2D(2,2),\n",
    "    Conv2D(64, (3,3), activation='relu'),\n",
    "    MaxPooling2D(2,2),\n",
    "    Flatten(),\n",
    "    Dense(64, activation='relu'),\n",
    "    Dense(2, activation='softmax')\n",
    "])\n",
    "\n",
    "model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "d8c2f9d0-25a8-4d5d-9019-a6f9b73a04db",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "189/189 [==============================] - 20s 98ms/step - loss: 0.5200 - accuracy: 0.7921 - val_loss: 0.6758 - val_accuracy: 0.8908\n",
      "Epoch 2/10\n",
      "189/189 [==============================] - 18s 93ms/step - loss: 0.3497 - accuracy: 0.8937 - val_loss: 0.5582 - val_accuracy: 0.9087\n",
      "Epoch 3/10\n",
      "189/189 [==============================] - 18s 93ms/step - loss: 0.2925 - accuracy: 0.9113 - val_loss: 0.4596 - val_accuracy: 0.9265\n",
      "Epoch 4/10\n",
      "189/189 [==============================] - 18s 94ms/step - loss: 0.2537 - accuracy: 0.9242 - val_loss: 0.5238 - val_accuracy: 0.8736\n",
      "Epoch 5/10\n",
      "189/189 [==============================] - 18s 97ms/step - loss: 0.2140 - accuracy: 0.9383 - val_loss: 0.6332 - val_accuracy: 0.7750\n",
      "Epoch 6/10\n",
      "189/189 [==============================] - 17s 92ms/step - loss: 0.1448 - accuracy: 0.9495 - val_loss: 0.2815 - val_accuracy: 0.9060\n",
      "Epoch 7/10\n",
      "189/189 [==============================] - 17s 93ms/step - loss: 0.0987 - accuracy: 0.9667 - val_loss: 0.3101 - val_accuracy: 0.8968\n",
      "Epoch 8/10\n",
      "189/189 [==============================] - 18s 97ms/step - loss: 0.0695 - accuracy: 0.9762 - val_loss: 0.1293 - val_accuracy: 0.9583\n",
      "Epoch 9/10\n",
      "189/189 [==============================] - 18s 96ms/step - loss: 0.0570 - accuracy: 0.9805 - val_loss: 0.3105 - val_accuracy: 0.9146\n",
      "Epoch 10/10\n",
      "189/189 [==============================] - 18s 97ms/step - loss: 0.0357 - accuracy: 0.9887 - val_loss: 0.3866 - val_accuracy: 0.9040\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x19910bd2d50>"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X, y, epochs=10, validation_split=0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "89bc97ff-00cc-496b-9abf-07f212a893a8",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _update_step_xla while saving (showing 3 of 3). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: mask_detector.model\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: mask_detector.model\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Model saved as mask_detector.model\n"
     ]
    }
   ],
   "source": [
    "model.save(\"mask_detector.model\")\n",
    "print(\"✅ Model saved as mask_detector.model\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5b45cce6-3dc0-44cd-96b0-e787d879d73d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0rc2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
