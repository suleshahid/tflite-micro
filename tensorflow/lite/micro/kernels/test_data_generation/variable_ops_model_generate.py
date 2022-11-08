"""Simple resource variable model generate"""




def create_model()

def convert_to_tflite(model):
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.allow_custom_ops = True
  return converter.convert()

def save_model(save_dir, save_name, tflite_model):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + "/" + save_name, "wb") as f:
        f.write(tflite_model)
    logging.info("Tflite model saved to %s", save_dir)


def main(_):


if __name__ == "__main__":
  app.run(main)