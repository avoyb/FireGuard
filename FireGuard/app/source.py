import numpy as np
from tensorflow.keras.models import Sequential
from keras import backend as K
from PIL import Image

def augment_image(input_image, im_width=720, im_height=480, color=0):
  height, width, vec = input_image.shape
  padded_image = np.pad(input_image, ((0, im_height-height), (0, im_width-width), (0, 0)), 'constant', constant_values=color)
  return padded_image

def image_segmentation(input_image, im_height=480, im_width=720):
  input_image = input_image / np.max(input_image.astype('float'))
  height, width, vec = input_image.shape
  row_count = height//im_height if height%im_height==0 else height//im_height+1
  col_count = width//im_width if width%im_width==0 else width//im_width+1

  new_image_array = np.zeros((row_count*col_count, im_height, im_width, vec))

  for i in range(row_count):
    for j in range(col_count):
      cropped_image = input_image[im_height*i:im_height*(i+1), im_width*j:im_width*(j+1), :]
      new_image_array[i*col_count+j] = augment_image(cropped_image, im_width=im_width, im_height=im_height, color=0)
  return new_image_array, row_count, col_count

def stitch_images(input_image, row_count, col_count, original_height, original_width, im_height=480, im_width=720, remove_ghost=True):
  num, height, width, vec = input_image.shape
  stitched_image = np.zeros((height*row_count, width*col_count, vec))
  for i in range(row_count):
    for j in range(col_count):
      if remove_ghost:
        input_image[i*col_count+j, :, :, :] = np.pad(input_image[i*col_count+j, 4:height-4, 4:width-4, :], ((4, 4), (4, 4), (0, 0)), 'edge')
          
      stitched_image[im_height*i:im_height*(i+1), im_width*j:im_width*(j+1), :] = input_image[i*col_count+j, :, :, :]  
  return stitched_image[:original_height, :original_width, :]

def predict_batch(input_image_array, model):
  num, height, width, vec = input_image_array.shape
  preds_array = np.zeros((num, height, width, 1))
  for ii in range(input_image_array.shape[0]):
    preds_array[ii] = model.predict(np.expand_dims(input_image_array[ii, :, :, :], axis=0), verbose=1)
  return preds_array
  

def convert_float_to_int(image):
  if not np.any(image): # check if the array is all zero 
    return image.astype(int)
  return ((image-np.min(image))/(np.max(image)-np.min(image))*255).astype(int)


def load_model_from_file(model_location):
    model = Sequential().load_model(model_location)
    session = K.get_session()
    return model, session

def calculate_burnt_area(output_mask, resolution, forest_type):
  biomass_type = {'Tropical Forest': 28076,
                   'Temperate Forest':10492,
                   'Boreal Forest': 25000,
                   'Shrublands': 5705,
                   'Grasslands': 976
                  }
  area = np.count_nonzero(output_mask) * resolution**2
  biomass_burnt =  area * biomass_type[forest_type]/1e3 * 1624 #unit in g
  
  ca_co2_daily = 4.24e8 / 365. # Califorlia annual CO2 emission from power generating, 424 million metric tons of CO2 per year
  
  equivalent_days = biomass_burnt /1e6 / ca_co2_daily
  
  print('The total burnt area is:', "{:.4e}".format(area/1e6), 'km^2 \n')
  print('The total CO2 emitted is:', "{:.4e}".format(biomass_burnt/1e6), 'tons \n')
  print("Equivalent to:", "{:.4e}".format(equivalent_days), " days of Califorlia's  daily electricity power emission \n")
  return area, biomass_burnt, equivalent_days