from keras.applications.imagenet_utils import decode_predictions

# Get class index → human-readable labels
imagenet_classes = decode_predictions([[0]*1000], top=1000)[0]

# Extract only class names.
class_names = [name for (_, name, _) in imagenet_classes]


