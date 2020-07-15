#yolov3.py
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, \
    Input, ZeroPadding2D, LeakyReLU, UpSampling2D

def parse_cfg(cfgfile):
    with open(cfgfile, 'r') as file:
        lines = [line.rstrip('\n') for line in file if line != '\n' and line[0] != '#']
    holder = {}
    blocks = []
    for line in lines:
        if line[0] == '[':
            line = 'type=' + line[1:-1].rstrip()
            if len(holder) != 0:
                blocks.append(holder)
                holder = {}
        key, value = line.split("=")
        holder[key.rstrip()] = value.lstrip()
    blocks.append(holder)
    return blocks

"""
Let’s explain this code.

Lines 11-12, we open the cfgfile and read it, then remove unnecessary characters like ‘\n’ and ‘#’.

The variable lines in line 12 is now holding all the lines of the file yolov3.cfg. So, we need to loop over it in order to read every single line from it.

Lines 15-23, loop over the variable lines and read every single attribute from it and store them all in the list blocks. This process is performed by reading the attributes block per block. The block’s attributes and their values are firstly stored as the key-value pairs in a dictionary holder. After reading each block, all attributes are then appended to the list blocks and the holder is then made empty and ready to read another block. Loop until all blocks are read before returning the content of the list blocks.
"""

def YOLOv3Net(cfgfile, model_size, num_classes):

    blocks = parse_cfg(cfgfile)

    outputs = {}
    output_filters = []
    filters = []
    out_pred = []
    scale = 0

    inputs = input_image = Input(shape=model_size)
    inputs = inputs / 255.0

    for i, block in enumerate(blocks[1:]):
        # If it is a convolutional layer
        if (block["type"] == "convolutional"):

            activation = block["activation"]
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            strides = int(block["stride"])

            if strides > 1:
                inputs = ZeroPadding2D(((1, 0), (1, 0)))(inputs)

            inputs = Conv2D(filters,
                            kernel_size,
                            strides=strides,
                            padding='valid' if strides > 1 else 'same',
                            name='conv_' + str(i),
                            use_bias=False if ("batch_normalize" in block) else True)(inputs)

            if "batch_normalize" in block:
                inputs = BatchNormalization(name='bnorm_' + str(i))(inputs)
                inputs = LeakyReLU(alpha=0.1, name='leaky_' + str(i))(inputs)
        
        elif (block["type"] == "upsample"):
            stride = int(block["stride"])
            inputs = UpSampling2D(stride)(inputs)

                # If it is a route layer
        elif (block["type"] == "route"):
            block["layers"] = block["layers"].split(',')
            start = int(block["layers"][0])

            if len(block["layers"]) > 1:
                end = int(block["layers"][1]) - i
                filters = output_filters[i + start] + output_filters[end]  # Index negatif :end - index
                inputs = tf.concat([outputs[i + start], outputs[i + end]], axis=-1)
            else:
                filters = output_filters[i + start]
                inputs = outputs[i + start]


#What we’re going to do in this layer block is to backward 3 layers (-3) as indicated in from value, then take the feature map from that layer, 
# and add it with the feature map from the previous layer. Here is the code for that
        elif block["type"] == "shortcut":
            from_ = int(block["from"])
            inputs = outputs[i - 1] + outputs[i + from_]


#As we did to other layers, just check whether we’re in the yolo layer.
                # Yolo detection layer
        elif block["type"] == "yolo":
#If it is true, then take all the necessary attributes associated with it. In this case, we just need mask and anchors attributes.
            mask = block["mask"].split(",")
            mask = [int(x) for x in mask]
            anchors = block["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            n_anchors = len(anchors)
#Then we need to reshape the YOLOv3 output to the form of [None, B * grid size * grid size, 5 + C]. The B is the number of anchors and C is the number of classes.
            out_shape = inputs.get_shape().as_list()
            inputs = tf.reshape(inputs, [-1, n_anchors * out_shape[1] * out_shape[2], \
										 5 + num_classes])
#Then access all boxes attributes by this way:
            box_centers = inputs[:, :, 0:2]
            box_shapes = inputs[:, :, 2:4]
            confidence = inputs[:, :, 4:5]
            classes = inputs[:, :, 5:num_classes + 5]
#Use the sigmoid function to convert box_centers, confidence, and classes values into range of 0 – 1.
            box_centers = tf.sigmoid(box_centers)
            confidence = tf.sigmoid(confidence)
            classes = tf.sigmoid(classes)
#Then convert box_shapes as the following:
            anchors = tf.tile(anchors, [out_shape[1] * out_shape[2], 1])
            box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)
#Use a meshgrid to convert the relative positions of the center boxes into the real positions.
            x = tf.range(out_shape[1], dtype=tf.float32)
            y = tf.range(out_shape[2], dtype=tf.float32)

            cx, cy = tf.meshgrid(x, y)
            cx = tf.reshape(cx, (-1, 1))
            cy = tf.reshape(cy, (-1, 1))
            cxy = tf.concat([cx, cy], axis=-1)
            cxy = tf.tile(cxy, [1, n_anchors])
            cxy = tf.reshape(cxy, [1, -1, 2])

            strides = (input_image.shape[1] // out_shape[1], \
                       input_image.shape[2] // out_shape[2])
            box_centers = (box_centers + cxy) * strides
#Then, concatenate them all together.
            prediction = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)
#Take the prediction result for each scale and concatenate it with the others.
            if scale:
                out_pred = tf.concat([out_pred, prediction], axis=1)
            else:
                out_pred = prediction
                scale = 1
#Since the route and shortcut layers need output feature maps from previous layers, 
# so for every iteration, we always keep the track of the feature maps and output filters.
        outputs[i] = inputs
        output_filters.append(filters)
# Finally, we can return our model.
    model = Model(input_image, out_pred)
    model.summary()
    return model
