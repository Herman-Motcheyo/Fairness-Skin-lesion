import tensorflow as tf



def augment_image(image):
    # Flip horizontal et vertical
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Zoom central aléatoire (recadrage puis resize)
    scale = tf.random.uniform([], 0.9, 1.0)
    crop_height = tf.cast(scale * tf.cast(tf.shape(image)[0], tf.float32), tf.int32)
    crop_width = tf.cast(scale * tf.cast(tf.shape(image)[1], tf.float32), tf.int32)
    image = tf.image.random_crop(image, size=[crop_height, crop_width, 3])
    image = tf.image.resize(image, [224, 224])

    # Brightness, contrast, saturation, hue
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.image.random_hue(image, 0.05)

    # Bruit gaussien
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=2.0, dtype=tf.float32)
    image = tf.cast(image, tf.float32) + noise
    image = tf.clip_by_value(image, 0.0, 255.0)

    return tf.cast(image, tf.float32)

def make_multimodal_dataset(images, metadata, labels, batch_size=32, shuffle=True, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices(((images, metadata), labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))

    if augment:
        def augment_fn(inputs, label):
            img, meta = inputs
            img = augment_image(img)
            return (img, meta), label

        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
