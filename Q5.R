library(zip)
library(fs)
library(keras)
library(tfdatasets)

# Preliminary steps to produce folders

# unlink("dogs-vs-cats", recursive = TRUE)
# zip::unzip('dogs-vs-cats.zip', exdir = "dogs-vs-cats", files = "train.zip")
# zip::unzip("dogs-vs-cats/train.zip", exdir = "dogs-vs-cats")
# unlink("cats_vs_dogs_small", recursive = TRUE)

original_dir <- path("dogs-vs-cats/train")
new_base_dir <- path("cats_vs_dogs_small")

make_subset <- function(subset_name, start_index, end_index) {
  for (category in c("dog", "cat")) {
    file_name <- glue::glue("{category}.{ start_index:end_index }.jpg")
    dir_create(new_base_dir / subset_name / category)
    file_copy(original_dir / file_name,
              new_base_dir / subset_name / category / file_name)
  }
}

# create training, validation and test sets

make_subset("train", start_index = 1, end_index = 1000)
make_subset("validation", start_index = 1001, end_index = 1500)
make_subset("test", start_index = 1501, end_index = 2500)

train_dataset <-
  image_dataset_from_directory(new_base_dir / "train",
                               image_size = c(180, 180),
                               batch_size = 32)
validation_dataset <-
  image_dataset_from_directory(new_base_dir / "validation",
                               image_size = c(180, 180),
                               batch_size = 32)
test_dataset <-
  image_dataset_from_directory(new_base_dir / "test",
                               image_size = c(180, 180),
                               batch_size = 32)

# Read image function to process images into tensor data
tf_read_image <- 
  function(path, format="image", resize = NULL, ...){
    img <- path %>%
      tf$io$read_file() %>%
      tf$io[[paste0("decode_", format)]](...)
    if (!is.null(resize))
      img <- img %>%
        tf$image$resize(as.integer(resize))
    img
  }

# Function to display an image tensor
display_image_tensor <- function(x, ..., max = 255,
                                 plot_margins = c(0, 0, 0, 0)) {
  if (!is.null(plot_margins))
    withr::local_par(mar = plot_margins)
  
  x %>%
    as.array() %>%
    drop() %>%
    as.raster(max = max) %>%
    plot(..., interpolate = FALSE)
}

# Find the image path for the cropped picture of Bella
img_path <- normalizePath(("crop_bella.jpg"))

# Get an image tensor of the picture
img_tensor <- img_path %>%
  tf_read_image(resize = c(180, 180))
display_image_tensor(img_tensor)

# Allows us to check if a layer is for convolution or for pooling
conv_layer_s3_classname <- class(layer_conv_2d(NULL, 1, 1))[1]
pooling_layer_s3_classname <- class(layer_max_pooling_2d(NULL))[1]
is_conv_layer <- function(x) inherits(x, conv_layer_s3_classname)
is_pooling_layer <- function(x) inherits(x, pooling_layer_s3_classname)

# Store all layer outputs that we need (convolutions and pooling layers)
layer_outputs <- list()
for (layer in model$layers)
  if (is_conv_layer(layer) || is_pooling_layer(layer))
    layer_outputs[[layer$name]] <- layer$output

# Produce the activation model and predict the img tensor data across all axes
activation_model <- keras_model(inputs = model$input,
                                outputs = layer_outputs)
activations <- activation_model %>%
  predict(img_tensor[tf$newaxis, , , ])

# Get first layer
first_layer_activation <- activations[[ names(layer_outputs)[1] ]]

# Produce function to plot all activations produced
plot_activations <- function(x, ...) {
  
  x <- as.array(x)
  
  if(sum(x) == 0)
    return(plot(as.raster("gray")))
  
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(x), asp = 1, axes = FALSE, useRaster = TRUE,
        col = terrain.colors(256), ...)
}

plot_activations(first_layer_activation[, , , 5])

# Plot each desired layer's plot activation and observe differences per layer.
par(mfrow=c(5,2))
for (layer_name in names(layer_outputs)) {
  layer_output <- activations[[layer_name]]
  
  n_features <- dim(layer_output) %>% tail(1)
  par(mfrow = n2mfrow(n_features, asp = 1.75),
      mar = rep(.1, 4), oma = c(0, 0, 1.5, 0))
  for (j in 1:n_features)
    plot_activations(layer_output[, , ,j])
  title(main = layer_name, outer = TRUE)
}
