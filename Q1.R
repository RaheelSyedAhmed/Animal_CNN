library(keras)
# Load in Mnist data
mnist <- dataset_mnist()

# Assign variables to work with training and test data from the mnist dataset
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

# Reshape data into a matrix form and divide by 255 to scale
train_images <- array_reshape(train_images, c(60000, 28, 28, 1)) # matrix
train_images <- train_images/255 # ensures all values are in [0, 1]
test_images <- array_reshape(test_images, c(10000, 28, 28, 1))
test_images <- test_images/255

# Set response to categorical for multinomial response
cat_train_labels <- to_categorical(train_labels, 10)
cat_test_labels <- to_categorical(test_labels, 10)

# Build convolutional model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3),
                padding = "same", activation = "relu",
                input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3),
                padding = "same", activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3),
                padding = "same", activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")
summary(model)

# Compile the model
model %>% compile(loss = "categorical_crossentropy",
                  optimizer = optimizer_rmsprop(), metrics = c("accuracy"))

# Fit the model to train data with categorical labels
history <- model %>% fit(train_images, cat_train_labels, epochs = 30,
                         batch_size = 128, validation_split = 0.2)
# Print training and validation accuracy
history
# Print test accuracy
(model %>% evaluate(test_images, cat_test_labels, verbose = F))["accuracy"]

# Produce a better model (with dropout) and print out metrics of evaluation
better_model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3),
                padding = "same", activation = "relu",
                input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3),
                padding = "same", activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3),
                padding = "same", activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")
summary(better_model)

better_model %>% compile(loss = "categorical_crossentropy",
                  optimizer = optimizer_rmsprop(), metrics = c("accuracy"))
history2 <- better_model %>% fit(train_images, cat_train_labels, epochs = 30,
                         batch_size = 128, validation_split = 0.2)
history2
(better_model %>% evaluate(test_images, cat_test_labels, verbose = F))["accuracy"]
