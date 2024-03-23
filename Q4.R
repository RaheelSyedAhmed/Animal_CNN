library(keras)

# Load in cropped image of Bella
img <- image_load("crop_bella.jpg", target_size = c(224, 224))
# Ensure the dimensions of the input are correct 
# (1 for providing only 1 image and 224, 224, 3 as the dimensions required for each image)
x <- array(dim = c(1, 224, 224, 3))
# Convert the image to an array
x[1,,,] <- (image_to_array(img))

# Plot for visualization prior to evaluation
plot(as.raster(x[1,,,]/255))

# Preprocess the picture with imagenet
x <- imagenet_preprocess_input(x)

# Develop resnet50 model
model <- application_resnet50(weights = "imagenet")
summary(model)

# Predict the model and display the top 5 predictions for our picture of Bella
pred <- model %>% predict(x) %>%
  imagenet_decode_predictions(top = 5)
#names(pred) <- image_names
print(pred)
